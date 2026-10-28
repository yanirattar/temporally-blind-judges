"""Turns a plan in config.yaml into a list of judge jobs.

A plan picks clips (all, or N spread evenly across scenes), conditions (name patterns
such as 'clean' or '*_strong'), setups, and a repeat index. Repeat 1 asks the same
question again, to measure how consistent the judge is.
"""

from __future__ import annotations

import fnmatch

import numpy as np
import pandas as pd

from .config import Config
from .judge import Job


def select_videos(cond: pd.DataFrame, n, seed: int = 0) -> list[str]:
    """All videos, or n of them spread round-robin across scenes in a fixed random order."""
    videos = cond[["video_id", "scenario"]].drop_duplicates()
    if n in (None, "all") or int(n) >= len(videos):
        return sorted(videos["video_id"])
    rng = np.random.default_rng(seed)
    by_scene = {s: list(rng.permutation(sorted(g["video_id"]))) for s, g in videos.groupby("scenario")}
    order = list(rng.permutation(sorted(by_scene)))
    picked: list[str] = []
    while len(picked) < int(n):
        for s in order:
            if by_scene[s] and len(picked) < int(n):
                picked.append(by_scene[s].pop(0))
    return sorted(picked)


def expand(cfg: Config, plan_name: str, cond: pd.DataFrame) -> list[Job]:
    plan = cfg.plan(plan_name)
    patterns = plan.get("conditions", ["*"])
    external = patterns == ["external"]
    pool = cond[cond["corruption"] == "external"] if external else cond[cond["corruption"] != "external"]
    if pool.empty:
        raise SystemExit(f"Plan '{plan_name}' matches no clips. "
                         + ("Add videos with `tbj add-external`." if external else "Run `tbj build` first."))
    videos = set(select_videos(pool, plan.get("clips", "all"), seed=plan.get("seed", 0)))
    pool = pool[pool["video_id"].isin(videos)]
    names = sorted(pool["condition"].unique())
    keep = {n for n in names if any(fnmatch.fnmatch(n, p) for p in patterns)}
    pool = pool[pool["condition"].isin(keep)]
    for s in plan["setups"]:
        cfg.setup(s)  # validates the name
    repeat = int(plan.get("repeat", 0))
    return [Job(cid, s, repeat) for cid in sorted(pool["condition_id"]) for s in plan["setups"]]
