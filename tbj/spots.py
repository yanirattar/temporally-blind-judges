"""A screen for clips whose motion detection caught the wrong thing.

Corruptions are placed inside the motion window, from the first to the last frame above 20%
of the clip's strongest motion. If that strongest motion is a side event (a reflection on the
whiteboard in marble-run-x, a door seen in the glass), the window and the corruptions follow it.
This measures, for every freeze, drop and partial reversal, the clean clip's strongest motion
among the affected frames, as a share of the clip's peak:

  freeze             where the picture stops: the frozen frame +-2 frames
  drop               the skipped frames
  partial reversal   the reversed stretch
  shuffle, full reversal   not defined: they span the whole clip

A clip with several freezes and drops on still frames (under 0.05) gets a look by eye.

It is a screen, not a measure of visibility. Checked against the eye it fails both ways: water
glints change many pixels without anything visible happening (0332's invisible drop scores 0.43),
and in clips that move the whole time the median, used as the noise level, is itself motion,
so an obvious brush movement can score 0 (0130). Visibility is estimated by hand instead
(tbj check --visibility).

Cached next to conditions.csv; a clip is measured again only if its clean video or this method changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .motion import motion_profile, relative_motion
from .video import read_frames

COLUMNS = ["condition_id", "video_id", "corruption", "strength", "clean_sha", "method", "motion_at_spot"]
METHOD = "max-v2"  # change when the measure changes, so cached values are recomputed
STILL = 0.05       # below this share of the peak, nothing in the affected frames moved above the clip's typical level


def spot_motion(rel: np.ndarray, corruption: str, start: int, length: int) -> float:
    """Strongest motion among the affected frames, as a share of the clip's peak (nan where not defined)."""
    n = len(rel)
    if corruption == "freeze":
        lo, hi = start - 2, start + 3
    elif corruption == "drop" or (corruption == "reversal" and length < n):
        lo, hi = start, start + length
    else:
        return float("nan")
    lo, hi = max(0, lo), min(n, hi)
    return float(rel[lo:hi].max()) if hi > lo else float("nan")


def _cache_path(cfg: Config):
    return cfg.path("conditions_csv").with_name("motion_at_spot.csv")


def spot_table(cfg: Config, cond: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """motion_at_spot for every corrupted condition of every clip in conditions.csv."""
    cache = _cache_path(cfg)
    old = pd.read_csv(cache, dtype={"video_id": str}) if cache.exists() else pd.DataFrame(columns=COLUMNS)
    corrupted = cond[~cond["corruption"].isin(["clean", "external"])]
    clean = cond[cond["condition"] == "clean"].set_index("video_id")
    videos = sorted(corrupted["video_id"].unique())
    parts, measured = [], 0
    for i, vid in enumerate(videos, 1):
        g = corrupted[corrupted["video_id"] == vid]
        sha = clean.loc[vid, "sha256"]
        hit = old[(old["video_id"] == vid) & (old["clean_sha"] == sha)
                  & (old.get("method", pd.Series(index=old.index, dtype=object)) == METHOD)]
        if set(hit["condition_id"]) >= set(g["condition_id"]):
            parts.append(hit[hit["condition_id"].isin(g["condition_id"])])
            continue
        frames, _ = read_frames(cfg.root / clean.loc[vid, "path"])
        rel = relative_motion(motion_profile(frames))
        del frames
        parts.append(pd.DataFrame([{
            "condition_id": r["condition_id"], "video_id": vid, "corruption": r["corruption"],
            "strength": r["strength"], "clean_sha": sha, "method": METHOD,
            "motion_at_spot": round(spot_motion(rel, r["corruption"], int(r["start_frame"]), int(r["length_frames"])), 4),
        } for _, r in g.iterrows()]))
        measured += 1
        if verbose and (measured % 25 == 0 or i == len(videos)):
            print(f"  motion at the spot: {i}/{len(videos)} clips")
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    if measured:
        table.to_csv(cache, index=False)
    return table


def describe(table: pd.DataFrame, cond: pd.DataFrame, min_still: int = 3) -> str:
    """Text summary: still spots per corruption, and the clips to look at (min_still or more of
    their freezes and drops on still frames)."""
    t = table.dropna(subset=["motion_at_spot"])
    if t.empty:
        return "No freezes, drops or partial reversals found."
    lines = [f"{len(t)} corruptions with a spot (freezes, drops, partial reversals) in {t['video_id'].nunique()} clips.",
             f"Still spots (under {STILL:g}: nothing moved above the clip's typical level): "
             f"{(t['motion_at_spot'] < STILL).sum()} of {len(t)} ({(t['motion_at_spot'] < STILL).mean():.0%}).", "",
             f"{'corruption':<10} {'strength':<8} {'n':>4} {'median':>7} {'still':>6}"]
    for (c, s_), g in t.groupby(["corruption", "strength"], sort=False):
        v = g["motion_at_spot"]
        lines.append(f"{c:<10} {s_:<8} {len(v):>4} {v.median():>7.3f} {(v < STILL).mean():>6.0%}")
    m = t[t["corruption"].isin(["freeze", "drop"])].merge(
        cond[["condition_id", "scenario", "perspective", "take", "start_frame", "fps", "motion_start", "motion_end"]],
        on="condition_id")
    still = m[m["motion_at_spot"] < STILL].groupby("video_id").size()
    flagged = still[still >= min_still].index
    lines += ["", f"Clips to look at ({min_still} or more of their freezes and drops on still frames): {len(flagged)}"]
    for vid in flagged:
        d = m[m["video_id"] == vid].sort_values("start_frame")
        r = d.iloc[0]
        fps = float(r["fps"])
        spots = ", ".join(f"{int(x['start_frame']) / fps:.1f}s={x['motion_at_spot']:.2f}" for _, x in d.iterrows())
        lines.append(f"  {vid} {r['scenario']} {r['perspective']} take {r['take']}: window "
                     f"{int(r['motion_start']) / fps:.1f}-{int(r['motion_end']) / fps:.1f} s; {spots}")
    lines += ["  In each, open clean.mp4 and look at the times scoring 1.00 and those scoring 0.00: if the",
              "  strongest motion is a side event (a reflection, a person, a door), exclude the clip."]
    return "\n".join(lines)
