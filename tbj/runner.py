"""Runs judge jobs: synchronously (pilot, small runs) or through the Batch API (half price).

Both paths share the cache: a job whose key already has a successful answer is skipped,
and a job sitting in an uncollected batch is not submitted again.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

import pandas as pd

from .config import Config
from .judge import (Job, ParseError, Prompt, build_request, cost_usd, estimate_tokens, job_key,
                    judge_settings, load_prompt, parse_answer, usage_from)
from .store import Store
from .uploads import Uploader

RETRY_CODES = {408, 429, 500, 502, 503, 504}
DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def make_client(cfg: Config, fake: bool = False):
    if fake:
        from .fake import FakeClient
        return FakeClient(cfg)
    try:
        from google import genai
    except ImportError as e:
        raise SystemExit("google-genai is not installed: pip install -e .") from e
    return genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment


def _state(job) -> str:
    return str(getattr(job.state, "name", job.state)).rsplit(".", 1)[-1]


@dataclass
class Item:
    job: Job
    key: str
    cond: dict
    setup_cfg: dict
    prompt: Prompt

    def meta(self, settings: dict, mode: str, extra: dict) -> dict:
        return {
            "condition_id": self.job.condition_id, "setup": self.job.setup, "repeat": self.job.repeat,
            "clip_sha": self.cond["sha256"], "model": settings["model"], "mode": mode, **extra,
        }


class Runner:
    def __init__(self, cfg: Config, client, store: Store, cond: pd.DataFrame):
        self.cfg, self.client, self.store = cfg, client, store
        self.cond = cond.set_index("condition_id", drop=False).to_dict("index")
        self.settings = judge_settings(cfg)
        self.uploader = Uploader(client, cfg.path("uploads"), cfg.root)
        self._prompts: dict[str, Prompt] = {}

    # ------------------------------------------------------------ preparation
    def prompt(self, name: str) -> Prompt:
        if name not in self._prompts:
            self._prompts[name] = load_prompt(name)
        return self._prompts[name]

    def prepare(self, jobs: list[Job]) -> list[Item]:
        items = []
        for job in jobs:
            cond = self.cond[job.condition_id]
            setup_cfg = self.cfg.setup(job.setup)
            prompt = self.prompt(setup_cfg["prompt"])
            items.append(Item(job, job_key(job, cond, setup_cfg, prompt, self.settings), cond, setup_cfg, prompt))
        return items

    def pending(self, items: list[Item]) -> list[Item]:
        done = self.store.ok_keys() | self.store.in_flight_keys()
        seen, out = set(), []
        for it in items:
            if it.key not in done and it.key not in seen:
                seen.add(it.key)
                out.append(it)
        return out

    def estimate(self, items: list[Item], batch: bool) -> tuple[int, float]:
        j = self.cfg["judge"]
        total = 0.0
        for it in items:
            tin, tout = estimate_tokens(it.cond, it.setup_cfg, it.prompt, j["tokens_per_frame"], j["output_tokens_guess"])
            total += cost_usd({"input_tokens": tin, "output_tokens": tout, "thought_tokens": 0}, self.cfg["pricing"], batch)
        return len(items), total

    def _uploads(self, items: list[Item], min_hours: float) -> dict[str, str]:
        files = []
        for it in items:
            files.append((it.cond["path"], it.cond["sha256"]))
            if it.setup_cfg["kind"] == "paired":
                files.append((it.cond["rev_path"], it.cond["rev_sha256"]))
        return self.uploader.ensure(files, min_hours=min_hours)

    def _request(self, it: Item, uris: dict[str, str]) -> tuple[dict, dict]:
        rev = uris.get(it.cond["rev_sha256"]) if it.setup_cfg["kind"] == "paired" else None
        return build_request(it.job, it.setup_cfg, it.prompt, self.settings, uris[it.cond["sha256"]], rev)

    # ------------------------------------------------------------ sync
    def _call(self, it: Item, uris: dict[str, str], max_retries: int) -> tuple[dict, dict]:
        request, meta = self._request(it, uris)
        spent, last_error = 0.0, ""
        tokens = {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
        parse_failures = 0
        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(**request)
            except Exception as e:  # noqa: BLE001 - classify any SDK or network error
                code = getattr(e, "code", None)
                last_error = f"{type(e).__name__}: {e}"[:500]
                if isinstance(code, int) and code not in RETRY_CODES:
                    break
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            tokens = usage_from(getattr(resp, "usage_metadata", None))
            spent += cost_usd(tokens, self.cfg["pricing"], batch=False)
            text = getattr(resp, "text", None)
            try:
                parsed = parse_answer(text, meta)
                return meta, {"status": "ok", **parsed, "raw_text": text, **tokens, "cost_usd": spent}
            except ParseError as e:
                last_error = str(e)[:500]
                parse_failures += 1
                if parse_failures >= 2:
                    return meta, {"status": "parse_error", "raw_text": text, "error": last_error,
                                  **tokens, "cost_usd": spent}
        return meta, {"status": "error", "error": last_error, **tokens, "cost_usd": spent}

    def run_sync(self, items: list[Item], budget: float, concurrency: int = 4, max_retries: int = 6) -> dict:
        if not items:
            return {"ok": 0, "failed": 0, "stopped": False}
        uris = self._uploads(items, min_hours=1.0)
        counts = {"ok": 0, "failed": 0, "stopped": False}
        queue = iter(items)
        in_flight = {}
        start = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            def submit_next() -> bool:
                if self.store.total_cost() >= budget:
                    counts["stopped"] = True
                    return False
                it = next(queue, None)
                if it is None:
                    return False
                in_flight[pool.submit(self._call, it, uris, max_retries)] = it
                return True

            for _ in range(concurrency * 2):
                if not submit_next():
                    break
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    it = in_flight.pop(fut)
                    meta, result = fut.result()
                    self.store.record(it.key, it.meta(self.settings, "sync", meta), result)
                    counts["ok" if result["status"] == "ok" else "failed"] += 1
                    n = counts["ok"] + counts["failed"]
                    if n % 25 == 0 or n == len(items):
                        rate = n / max(time.time() - start, 1e-6)
                        print(f"  {n}/{len(items)} done ({counts['failed']} failed), "
                              f"${self.store.total_cost():.2f} spent in total, {rate:.1f} calls/s")
                    submit_next()
        return counts

    # ------------------------------------------------------------ batch
    def submit_batches(self, items: list[Item], plan: str, chunk_size: int, min_upload_hours: float) -> list[str]:
        if not items:
            return []
        uris = self._uploads(items, min_hours=min_upload_hours)
        names = []
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            src, records = [], []
            for it in chunk:
                request, meta = self._request(it, uris)
                src.append({"contents": request["contents"], "config": request["config"],
                            "metadata": {"key": it.key}})
                records.append((it.key, it.meta(self.settings, "batch", meta)))
            display = f"tbj-{plan}-{stamp}-{i // chunk_size + 1}"
            job = self.client.batches.create(model=self.settings["model"], src=src,
                                             config={"display_name": display})
            self.store.add_batch(job.name, plan, display, records)
            names.append(job.name)
            print(f"  submitted {display}: {len(chunk)} requests ({job.name})")
        return names

    def batch_status(self) -> list[dict]:
        rows = []
        for b in self.store.open_batches():
            job = self.client.batches.get(name=b["name"])
            state = _state(job)
            self.store.set_batch_state(b["name"], state)
            rows.append({**b, "state": state})
        return rows

    def collect(self) -> dict:
        counts = {"ok": 0, "failed": 0, "batches_done": 0, "batches_running": 0}
        for b in self.store.open_batches():
            job = self.client.batches.get(name=b["name"])
            state = _state(job)
            if state not in DONE_STATES:
                counts["batches_running"] += 1
                self.store.set_batch_state(b["name"], state)
                continue
            items = self.store.batch_items(b["name"])
            if state == "JOB_STATE_SUCCEEDED":
                for key, resp, error in self._batch_responses(job):
                    meta = items.get(key)
                    if meta is None:
                        continue
                    result = self._batch_result(meta, resp, error)
                    self.store.record(key, meta, result)
                    counts["ok" if result["status"] == "ok" else "failed"] += 1
            else:
                print(f"  {b['display_name']} ended as {state}; its jobs will be resubmitted next time")
            self.store.set_batch_state(b["name"], state, collected=True)
            counts["batches_done"] += 1
        return counts

    def _batch_responses(self, job):
        """Yields (key, response, error) from inline responses or a results file."""
        dest = job.dest
        inline = getattr(dest, "inlined_responses", None)
        if inline:
            for r in inline:
                key = (getattr(r, "metadata", None) or {}).get("key")
                yield key, getattr(r, "response", None), getattr(r, "error", None)
            return
        file_name = getattr(dest, "file_name", None)
        if file_name:
            data = self.client.files.download(file=file_name)
            for line in data.decode().splitlines():
                if line.strip():
                    obj = json.loads(line)
                    yield obj.get("key"), obj.get("response"), obj.get("error")

    def _batch_result(self, meta: dict, resp, error) -> dict:
        if error is not None or resp is None:
            return {"status": "error", "error": str(error)[:500]}
        if isinstance(resp, dict):  # REST JSON from a results file
            parts = (resp.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            tokens = usage_from(resp.get("usageMetadata"))
        else:
            text = getattr(resp, "text", None)
            tokens = usage_from(getattr(resp, "usage_metadata", None))
        cost = cost_usd(tokens, self.cfg["pricing"], batch=True)
        try:
            return {"status": "ok", **parse_answer(text, meta), "raw_text": text, **tokens, "cost_usd": cost}
        except ParseError as e:
            return {"status": "parse_error", "raw_text": text, "error": str(e)[:500], **tokens, "cost_usd": cost}
