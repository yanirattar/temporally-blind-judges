"""SQLite store of every judge call: the cache, the raw answers and the cost log.

A call is stored once per cache key; reruns skip keys that already succeeded, so you
never pay twice for the same request. Batch jobs are tracked here too.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  key TEXT PRIMARY KEY,
  condition_id TEXT, setup TEXT, repeat INTEGER, clip_sha TEXT,
  model TEXT, prompt TEXT, prompt_hash TEXT, fps REAL, kind TEXT, clip_slot TEXT,
  mode TEXT, status TEXT, flag INTEGER, score REAL, any_check INTEGER,
  answer TEXT, raw_text TEXT, error TEXT,
  input_tokens INTEGER, output_tokens INTEGER, thought_tokens INTEGER, cost_usd REAL,
  created_at REAL
);
CREATE INDEX IF NOT EXISTS calls_setup ON calls(setup, repeat);
CREATE TABLE IF NOT EXISTS batches (
  name TEXT PRIMARY KEY, plan TEXT, display_name TEXT, created_at REAL,
  state TEXT, n_requests INTEGER, collected INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS batch_items (
  batch_name TEXT, key TEXT, meta TEXT, PRIMARY KEY (batch_name, key)
);
"""


class Store:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(SCHEMA)

    # ---------------------------------------------------------- calls
    def ok_keys(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT key FROM calls WHERE status = 'ok'")}

    def record(self, key: str, meta: dict, result: dict) -> None:
        """meta: job info; result: status, flag, score, answer, raw_text, error, tokens, cost."""
        row = {
            "key": key,
            "condition_id": meta["condition_id"], "setup": meta["setup"], "repeat": meta["repeat"],
            "clip_sha": meta["clip_sha"], "model": meta["model"], "prompt": meta["prompt"],
            "prompt_hash": meta["prompt_hash"], "fps": meta["fps"], "kind": meta["kind"],
            "clip_slot": meta.get("clip_slot"), "mode": meta["mode"],
            "status": result["status"],
            "flag": None if result.get("flag") is None else int(result["flag"]),
            "score": result.get("score"),
            "any_check": None if result.get("any_check") is None else int(result["any_check"]),
            "answer": json.dumps(result["answer"]) if result.get("answer") is not None else None,
            "raw_text": result.get("raw_text"), "error": result.get("error"),
            "input_tokens": result.get("input_tokens", 0), "output_tokens": result.get("output_tokens", 0),
            "thought_tokens": result.get("thought_tokens", 0), "cost_usd": result.get("cost_usd", 0.0),
            "created_at": time.time(),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._lock:
            # Keep a successful answer if one exists; otherwise store the latest attempt.
            existing = self.db.execute("SELECT status, cost_usd FROM calls WHERE key = ?", (key,)).fetchone()
            if existing and existing[0] == "ok" and row["status"] != "ok":
                return
            if existing:  # keep the cost of earlier failed attempts in the total
                row["cost_usd"] = (row["cost_usd"] or 0) + (existing[1] or 0)
            self.db.execute(f"INSERT OR REPLACE INTO calls ({cols}) VALUES ({marks})", tuple(row.values()))
            self.db.commit()

    def total_cost(self) -> float:
        return float(self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()[0])

    def calls(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM calls", self.db)

    # ---------------------------------------------------------- batches
    def add_batch(self, name: str, plan: str, display_name: str, items: list[tuple[str, dict]]) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO batches (name, plan, display_name, created_at, state, n_requests) VALUES (?,?,?,?,?,?)",
                (name, plan, display_name, time.time(), "SUBMITTED", len(items)),
            )
            self.db.executemany(
                "INSERT INTO batch_items (batch_name, key, meta) VALUES (?,?,?)",
                [(name, k, json.dumps(m)) for k, m in items],
            )
            self.db.commit()

    def open_batches(self) -> list[dict]:
        cur = self.db.execute("SELECT name, plan, display_name, created_at, state, n_requests FROM batches "
                              "WHERE collected = 0 ORDER BY created_at")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def in_flight_keys(self) -> set[str]:
        return {r[0] for r in self.db.execute(
            "SELECT i.key FROM batch_items i JOIN batches b ON b.name = i.batch_name WHERE b.collected = 0")}

    def batch_items(self, name: str) -> dict[str, dict]:
        return {k: json.loads(m) for k, m in
                self.db.execute("SELECT key, meta FROM batch_items WHERE batch_name = ?", (name,))}

    def set_batch_state(self, name: str, state: str, collected: bool = False) -> None:
        with self._lock:
            self.db.execute("UPDATE batches SET state = ?, collected = ? WHERE name = ?",
                            (state, int(collected), name))
            self.db.commit()
