"""Judge requests and answers: prompts, JSON schemas, cache keys, parsing and cost.

Three kinds of setup:
  single     one clip, overall plausibility rubric      (setups: default, dense)
  checklist  one clip, explicit questions about time     (setup: checklist)
  paired     the clip next to its reversed copy, in random order; the judge picks
             the plausible one                           (setup: paired)

A request is a plain dict in the google-genai SDK's shape, so the same object is
sent by the sync runner (`client.models.generate_content(**request)`) and inside a
Batch API job (an inlined request).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources

from .corrupt import rng_for

PROBLEM_TYPES = [
    "none", "gravity_or_motion", "collision_or_contact", "liquid_or_material",
    "object_appearance", "time_direction_or_continuity", "other",
]
CHECKLIST_ITEMS = ["backward_motion", "unexplained_pause", "jump_or_teleport", "impossible_order"]

_SCORE = {"type": "integer", "minimum": 1, "maximum": 5}
_VERDICT = {"type": "string", "enum": ["plausible", "implausible"]}
_REASON = {"type": "string"}

# Field order matters: Gemini fills fields in schema order, so the reason comes first.
SCHEMAS = {
    "single": {
        "type": "object",
        "properties": {
            "reason": _REASON,
            "problem_type": {"type": "string", "enum": PROBLEM_TYPES},
            "plausibility_score": _SCORE,
            "verdict": _VERDICT,
        },
        "required": ["reason", "problem_type", "plausibility_score", "verdict"],
    },
    "checklist": {
        "type": "object",
        "properties": {
            "reason": _REASON,
            **{item: {"type": "boolean"} for item in CHECKLIST_ITEMS},
            "plausibility_score": _SCORE,
            "verdict": _VERDICT,
        },
        "required": ["reason", *CHECKLIST_ITEMS, "plausibility_score", "verdict"],
    },
    "paired": {
        "type": "object",
        "properties": {
            "reason": _REASON,
            "plausible_video": {"type": "string", "enum": ["A", "B", "neither"]},
            "confidence": _SCORE,
        },
        "required": ["reason", "plausible_video", "confidence"],
    },
}


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:12]


def load_prompt(name: str) -> Prompt:
    text = resources.files("tbj.prompts").joinpath(f"{name}.txt").read_text()
    return Prompt(name, text.strip())


@dataclass(frozen=True)
class Job:
    condition_id: str
    setup: str
    repeat: int = 0


def judge_settings(cfg) -> dict:
    j = cfg["judge"]
    return {
        "model": j["model"],
        "media_resolution": j.get("media_resolution"),
        "thinking_level": j.get("thinking_level"),
        "temperature": j.get("temperature"),
    }


def job_key(job: Job, cond: dict, setup_cfg: dict, prompt: Prompt, settings: dict) -> str:
    """Cache key: changes whenever anything that could change the answer changes
    (the clip bytes, the prompt text, the frame rate, the model settings, the repeat)."""
    blob = {
        "clip": cond["sha256"],
        "rev": cond["rev_sha256"] if setup_cfg["kind"] == "paired" else None,
        "kind": setup_cfg["kind"],
        "prompt": prompt.hash,
        "fps": setup_cfg["fps"],
        "settings": settings,
        "repeat": job.repeat,
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:24]


def pair_order(job: Job) -> str:
    """'A' if the judged clip goes in slot A, else 'B'. Random but reproducible per job."""
    return "A" if rng_for(0, job.condition_id, job.setup, job.repeat).random() < 0.5 else "B"


def _video_part(uri: str, fps: float) -> dict:
    return {"file_data": {"file_uri": uri, "mime_type": "video/mp4"}, "video_metadata": {"fps": fps}}


def build_request(job: Job, setup_cfg: dict, prompt: Prompt, settings: dict,
                  clip_uri: str, rev_uri: str | None = None) -> tuple[dict, dict]:
    """Returns (request, meta). `request` has keys model, contents, config."""
    kind, fps = setup_cfg["kind"], setup_cfg["fps"]
    meta = {"kind": kind, "prompt": prompt.name, "prompt_hash": prompt.hash, "fps": fps}
    if kind == "paired":
        if rev_uri is None:
            raise ValueError("paired setup needs the reversed clip")
        slot = pair_order(job)
        a, b = (clip_uri, rev_uri) if slot == "A" else (rev_uri, clip_uri)
        parts = [{"text": "Video A:"}, _video_part(a, fps), {"text": "Video B:"}, _video_part(b, fps),
                 {"text": prompt.text}]
        meta["clip_slot"] = slot
    else:
        parts = [_video_part(clip_uri, fps), {"text": prompt.text}]
    config = {"response_mime_type": "application/json", "response_json_schema": SCHEMAS[kind]}
    if settings.get("media_resolution"):
        config["media_resolution"] = settings["media_resolution"]
    if settings.get("thinking_level"):
        config["thinking_config"] = {"thinking_level": settings["thinking_level"]}
    if settings.get("temperature") is not None:
        config["temperature"] = settings["temperature"]
    request = {"model": settings["model"], "contents": [{"role": "user", "parts": parts}], "config": config}
    return request, meta


class ParseError(ValueError):
    pass


def parse_answer(text: str | None, meta: dict) -> dict:
    """Turn the judge's JSON into a flag and a score.

    flag   True if the judge says the judged clip is implausible
    score  implausibility in [0, 1], used for AUROC
    """
    if not text:
        raise ParseError("empty response")
    try:
        ans = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(f"not JSON: {text[:200]!r}") from e
    kind = meta["kind"]
    try:
        if kind in ("single", "checklist"):
            score = int(ans["plausibility_score"])
            verdict = ans["verdict"]
            if verdict not in ("plausible", "implausible") or not 1 <= score <= 5:
                raise ParseError(f"bad values: {ans}")
            out = {"flag": verdict == "implausible", "score": (5 - score) / 4, "answer": ans}
            if kind == "checklist":
                out["any_check"] = any(bool(ans[i]) for i in CHECKLIST_ITEMS)
            return out
        choice, conf = ans["plausible_video"], int(ans["confidence"])
        if choice not in ("A", "B", "neither") or not 1 <= conf <= 5:
            raise ParseError(f"bad values: {ans}")
        if choice == meta["clip_slot"]:          # prefers the judged clip: not flagged
            flag, score = False, 0.5 - 0.1 * conf
        elif choice == "neither":                # rejects both: flagged
            flag, score = True, 0.5 + 0.05 * conf
        else:                                    # prefers the reversed copy: flagged
            flag, score = True, 0.5 + 0.1 * conf
        return {"flag": flag, "score": round(score, 3), "answer": ans}
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, ParseError):
            raise
        raise ParseError(f"missing or bad field ({e}): {text[:200]!r}") from e


def usage_from(usage) -> dict:
    """Token counts from an SDK usage_metadata object or a REST usageMetadata dict."""
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    get = (lambda k, c: usage.get(k, usage.get(c)) or 0) if isinstance(usage, dict) else \
          (lambda k, c: getattr(usage, k, None) or 0)
    return {
        "input_tokens": int(get("prompt_token_count", "promptTokenCount")),
        "output_tokens": int(get("candidates_token_count", "candidatesTokenCount")),
        "thought_tokens": int(get("thoughts_token_count", "thoughtsTokenCount")),
    }


def cost_usd(tokens: dict, pricing: dict, batch: bool) -> float:
    """Thinking tokens are billed as output."""
    cost = (tokens["input_tokens"] * pricing["input"]
            + (tokens["output_tokens"] + tokens["thought_tokens"]) * pricing["output"]) / 1e6
    return cost * (pricing.get("batch_discount", 0.5) if batch else 1.0)


def estimate_tokens(cond: dict, setup_cfg: dict, prompt: Prompt, tokens_per_frame: int,
                    output_guess: int) -> tuple[int, int]:
    """Rough (input, output) tokens for one call, for pre-run estimates only."""
    frames = max(1, int(float(cond["duration_s"]) * setup_cfg["fps"]) + 1)
    videos = 2 if setup_cfg["kind"] == "paired" else 1
    return frames * tokens_per_frame * videos + len(prompt.text) // 4 + 20, output_guess
