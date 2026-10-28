"""A stand-in for the Gemini client, for `tbj demo`, tests and dry runs.

It mimics the parts of the google-genai client this project uses (files, models,
batches). Its answers are random, with detection rates set in RATES below. These
numbers are invented to exercise the pipeline; they say nothing about Gemini.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import threading
from pathlib import Path
from types import SimpleNamespace

# Probability that the fake judge flags a clip, by setup and condition. Invented.
RATES = {
    "default":   {"clean": 0.10, "reversal": (0.15, 0.25, 0.40), "shuffle": (0.35, 0.50, 0.65),
                  "freeze": (0.12, 0.15, 0.25), "drop": (0.15, 0.25, 0.35)},
    "dense":     {"clean": 0.12, "reversal": (0.25, 0.40, 0.60), "shuffle": (0.45, 0.60, 0.75),
                  "freeze": (0.20, 0.30, 0.45), "drop": (0.25, 0.40, 0.55)},
    "checklist": {"clean": 0.18, "reversal": (0.30, 0.45, 0.65), "shuffle": (0.50, 0.65, 0.80),
                  "freeze": (0.25, 0.35, 0.50), "drop": (0.30, 0.45, 0.60)},
    "paired":    {"clean": 0.08, "reversal": (0.40, 0.60, 0.90), "shuffle": (0.30, 0.40, 0.50),
                  "freeze": (0.10, 0.12, 0.15), "drop": (0.12, 0.15, 0.20)},
}
STRENGTH_INDEX = {"mild": 0, "medium": 1, "strong": 2}


class FakeError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"{code} {message}")
        self.code = code


def _condition_from_uri(uri: str) -> tuple[str, str, bool]:
    """What the file actually shows. Uploads are shared between byte-identical files, so a
    clip may arrive under another file's name: clean played backward is the same video as
    reversal_strong, and reversal_strong played backward is the clean clip."""
    stem = Path(uri.removeprefix("fake://")).stem
    rev = stem.endswith("__rev")
    name = stem.removesuffix("__rev")
    if "/external/" in uri:
        return "external", "", rev
    if rev and name == "clean":
        return "reversal", "strong", rev
    if rev and name == "reversal_strong":
        return "clean", "", rev
    if name == "clean":
        return "clean", "", rev
    corruption, strength = name.rsplit("_", 1)
    return corruption, strength, rev


def _setup_of(request: dict) -> tuple[str, list[str], float]:
    parts = request["contents"][0]["parts"]
    uris = [p["file_data"]["file_uri"] for p in parts if "file_data" in p]
    fps = [p["video_metadata"]["fps"] for p in parts if "file_data" in p][0]
    props = request["config"]["response_json_schema"]["properties"]
    if "plausible_video" in props:
        return "paired", uris, fps
    if "backward_motion" in props:
        return "checklist", uris, fps
    return ("dense" if fps > 1 else "default"), uris, fps


class _Files:
    def __init__(self, root: Path):
        self.root = root

    def upload(self, file, config=None):
        rel = os.path.relpath(file, self.root)
        return SimpleNamespace(name=f"files/{hashlib.sha1(rel.encode()).hexdigest()[:12]}",
                               uri=f"fake://{rel}", state="ACTIVE",
                               expiration_time=dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=47))

    def get(self, name):
        return SimpleNamespace(name=name, state="ACTIVE")

    def download(self, file):
        raise NotImplementedError


class _Models:
    def __init__(self, error_rate: float, seed: int):
        self.error_rate = error_rate
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def _draw(self) -> float:
        with self._lock:
            return self._rng.random()

    def generate_content(self, model, contents, config=None, raise_errors: bool = True):
        request = {"contents": contents, "config": config}
        if raise_errors and self._draw() < self.error_rate:
            raise FakeError(503, "UNAVAILABLE (simulated)")
        setup, uris, fps = _setup_of(request)
        conf = 2 + int(self._draw() * 4)
        if setup == "paired":
            shows = [_condition_from_uri(u)[:2] for u in uris]
            if ("clean", "") in shows and ("reversal", "strong") in shows:
                # One slot shows the real clip: the fake picks it with probability 0.9.
                forward = "AB"[shows.index(("clean", ""))]
                pick_forward = self._draw() < 0.9
                choice = forward if pick_forward else ("B" if forward == "A" else "A")
            else:
                # Both slots are flawed (or outside videos): the judged clip is the non-__rev file.
                clip_slot = "A" if not Path(uris[0]).stem.endswith("__rev") else "B"
                corruption, strength = shows["AB".index(clip_slot)]
                p = 0.5 if corruption == "external" else RATES["paired"][corruption][STRENGTH_INDEX[strength]]
                other = "B" if clip_slot == "A" else "A"
                flag = self._draw() < p
                choice = (other if self._draw() < 0.5 else "neither") if flag else clip_slot
            answer = {"reason": "simulated", "plausible_video": choice, "confidence": conf}
        else:
            corruption, strength, _ = _condition_from_uri(uris[0])
            if corruption == "external":
                p = 0.5
            elif corruption == "clean":
                p = RATES[setup]["clean"]
            else:
                p = RATES[setup][corruption][STRENGTH_INDEX[strength]]
            flag = self._draw() < p
            score = (1 + int(self._draw() * 2)) if flag else (4 + int(self._draw() * 2))
            answer = {"reason": "simulated",
                      "problem_type": "time_direction_or_continuity" if flag and self._draw() < 0.6
                      else ("other" if flag else "none"),
                      "plausibility_score": score, "verdict": "implausible" if flag else "plausible"}
            if setup == "checklist":
                for item in ("backward_motion", "unexplained_pause", "jump_or_teleport", "impossible_order"):
                    answer[item] = self._draw() < (0.4 if flag else 0.08)
        frames = int(8 * fps) + 1
        usage = SimpleNamespace(prompt_token_count=frames * 70 * len(uris) + 400,
                                candidates_token_count=60 + int(self._draw() * 40),
                                thoughts_token_count=150 + int(self._draw() * 250))
        return SimpleNamespace(text=json.dumps(answer), usage_metadata=usage)


class _Batches:
    def __init__(self, models: _Models, folder: Path):
        self.models, self.folder = models, folder

    def create(self, model, src, config=None):
        name = f"batches/fake-{hashlib.sha1(json.dumps(src, sort_keys=True).encode()).hexdigest()[:12]}"
        responses = []
        for req in src:
            r = self.models.generate_content(model, req["contents"], req["config"], raise_errors=False)
            responses.append({"key": req["metadata"]["key"], "text": r.text,
                              "usage": vars(r.usage_metadata)})
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / f"{name.split('/')[-1]}.json").write_text(json.dumps(responses))
        return SimpleNamespace(name=name, state="JOB_STATE_PENDING")

    def get(self, name):
        data = json.loads((self.folder / f"{name.split('/')[-1]}.json").read_text())
        inline = [SimpleNamespace(metadata={"key": r["key"]}, error=None,
                                  response=SimpleNamespace(text=r["text"], usage_metadata=SimpleNamespace(**r["usage"])))
                  for r in data]
        return SimpleNamespace(name=name, state="JOB_STATE_SUCCEEDED",
                               dest=SimpleNamespace(inlined_responses=inline, file_name=None))


class FakeClient:
    def __init__(self, cfg, error_rate: float = 0.02, seed: int = 7):
        self.files = _Files(cfg.root)
        self.models = _Models(error_rate, seed)
        self.batches = _Batches(self.models, cfg.path("store").parent / "fake_batches")
