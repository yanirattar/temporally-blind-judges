"""Uploads clips to the Gemini Files API, once per file content.

Uploaded files expire after about 48 hours. The cache (uploads.json) records each
file's URI and expiry and re-uploads when a file would expire too soon for the
work at hand: a sync run needs an hour, a batch job up to a day.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _state_name(state) -> str:
    return str(getattr(state, "name", state)).rsplit(".", 1)[-1]


class Uploader:
    def __init__(self, client, cache_path: str | Path, root: str | Path):
        self.client = client
        self.cache_path = Path(cache_path)
        self.root = Path(root)
        self._lock = threading.Lock()
        self.cache: dict = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=1))
        tmp.replace(self.cache_path)

    def _fresh(self, sha: str, min_hours: float) -> str | None:
        entry = self.cache.get(sha)
        if not entry:
            return None
        expires = dt.datetime.fromisoformat(entry["expires"])
        return entry["uri"] if expires - _now() >= dt.timedelta(hours=min_hours) else None

    def _upload(self, rel_path: str, sha: str, timeout_s: float = 600) -> str:
        f = self.client.files.upload(file=str(self.root / rel_path),
                                     config={"mime_type": "video/mp4", "display_name": sha[:16]})
        start = time.time()
        while _state_name(f.state) == "PROCESSING":
            if time.time() - start > timeout_s:
                raise TimeoutError(f"File still processing after {timeout_s}s: {rel_path}")
            time.sleep(2)
            f = self.client.files.get(name=f.name)
        if _state_name(f.state) != "ACTIVE":
            raise RuntimeError(f"Upload failed for {rel_path}: state {f.state}")
        expires = getattr(f, "expiration_time", None) or (_now() + dt.timedelta(hours=47))
        if isinstance(expires, str):
            expires = dt.datetime.fromisoformat(expires.replace("Z", "+00:00"))
        with self._lock:
            self.cache[sha] = {"uri": f.uri, "name": f.name, "path": rel_path, "expires": expires.isoformat()}
            self._save()
        return f.uri

    def uri(self, rel_path: str, sha: str, min_hours: float = 1.0) -> str:
        return self._fresh(sha, min_hours) or self._upload(rel_path, sha)

    def ensure(self, items: list[tuple[str, str]], min_hours: float = 1.0, workers: int = 8) -> dict[str, str]:
        """Upload (rel_path, sha) pairs that are missing or expiring; returns sha -> uri."""
        items = list(dict.fromkeys(items))
        todo = [(p, s) for p, s in items if not self._fresh(s, min_hours)]
        if todo:
            print(f"  uploading {len(todo)} clips ({len(items) - len(todo)} already uploaded)")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._upload, p, s) for p, s in todo]
                for i, fut in enumerate(as_completed(futures), 1):
                    fut.result()
                    if i % 50 == 0 or i == len(todo):
                        print(f"    {i}/{len(todo)} uploaded")
        return {s: self.cache[s]["uri"] for _, s in items}
