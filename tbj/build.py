"""Render every condition of every selected clip, and the check page.

For each video of an 'irreversible' scene this writes 13 clips (clean + 4 corruptions
x 3 strengths) and the reversed copy of each (for the paired setup), all downscaled,
H.264 and without audio, plus one row per condition in conditions.csv.
Re-running skips clips that are already built with the same settings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .corrupt import condition_names, make_corruption
from .motion import motion_profile, motion_span
from .physiq import raw_path, read_manifest, selected_rows
from .video import read_frames, sha256_file, write_video

CONDITION_FIELDS = [
    "condition_id", "video_id", "scenario", "category", "perspective", "take", "condition",
    "corruption", "strength", "frames", "fps", "duration_s", "start_frame", "length_frames",
    "motion_overlap", "motion_start", "motion_end", "has_motion", "detail",
    "path", "rev_path", "sha256", "rev_sha256", "human_label",
]


def _settings_hash(cfg: Config) -> str:
    # Bump "v" whenever corruption or motion code changes, so clips are re-rendered.
    blob = json.dumps({"pre": cfg["preprocess"], "cor": cfg["corruptions"], "v": 2}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _rel(cfg: Config, p: Path) -> str:
    return os.path.relpath(p, cfg.root)


def build_clip(cfg: Config, row: dict) -> list[dict]:
    """Render all conditions for one video; returns its conditions.csv rows."""
    out_dir = cfg.path("conditions_dir") / row["video_id"]
    stamp = out_dir / "build.json"
    settings = _settings_hash(cfg)
    if stamp.exists():
        saved = json.loads(stamp.read_text())
        if saved.get("settings") == settings:
            return saved["rows"]

    pre, cor = cfg["preprocess"], cfg["corruptions"]
    frames, fps = read_frames(raw_path(cfg, row), width=pre["width"])
    n = len(frames)
    span = motion_span(motion_profile(frames), threshold=cor["motion_threshold"])
    rows = []
    for name in condition_names():
        c = make_corruption(name, n, fps, span, cor, seed=cor["seed"], video_id=row["video_id"])
        path, rev_path = out_dir / f"{name}.mp4", out_dir / f"{name}__rev.mp4"
        write_video(path, frames[c.index], fps, crf=pre["crf"], preset=pre["preset"])
        write_video(rev_path, frames[c.reversed_index], fps, crf=pre["crf"], preset=pre["preset"])
        rows.append({
            "condition_id": f"{row['video_id']}__{name}",
            "video_id": row["video_id"], "scenario": row["scenario"], "category": row["category"],
            "perspective": row["perspective"], "take": row["take"], "condition": name,
            "corruption": c.corruption, "strength": c.strength,
            "frames": n, "fps": fps, "duration_s": round(n / fps, 3),
            "start_frame": c.start, "length_frames": c.length, "motion_overlap": round(c.motion_overlap, 3),
            "motion_start": span.start, "motion_end": span.end, "has_motion": span.has_motion,
            "detail": c.detail,
            "path": _rel(cfg, path), "rev_path": _rel(cfg, rev_path),
            "sha256": sha256_file(path), "rev_sha256": sha256_file(rev_path), "human_label": "",
        })
    stamp.write_text(json.dumps({"settings": settings, "rows": rows}, indent=1))
    return rows


def _build_clip_worker(config_path: str, row: dict) -> list[dict]:
    from .config import load_config
    return build_clip(load_config(config_path), row)


def build_all(cfg: Config, config_path: str, workers: int = 2, limit: int | None = None) -> Path:
    rows = [r for r in selected_rows(read_manifest(cfg)) if raw_path(cfg, r).exists()]
    missing = len(selected_rows(read_manifest(cfg))) - len(rows)
    if missing:
        print(f"  {missing} selected videos are not downloaded yet; run `tbj download --set selected`.")
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit("Nothing to build: label scenes as 'irreversible' and download them first.")
    all_rows: list[dict] = []
    if workers <= 1:
        for i, r in enumerate(rows, 1):
            all_rows += build_clip(cfg, r)
            print(f"  [{i}/{len(rows)}] {r['filename']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_build_clip_worker, config_path, r): r for r in rows}
            for i, fut in enumerate(as_completed(futures), 1):
                all_rows += fut.result()
                print(f"  [{i}/{len(rows)}] {futures[fut]['filename']}")
    write_conditions(cfg, all_rows, keep_external=True)
    return cfg.path("conditions_csv")


def read_conditions(cfg: Config) -> pd.DataFrame:
    path = cfg.path("conditions_csv")
    if not path.exists():
        raise SystemExit(f"No {path}. Run `tbj build` first.")
    df = pd.read_csv(path, dtype={"video_id": str, "detail": str, "human_label": str}, keep_default_na=False)
    return df


def write_conditions(cfg: Config, rows: list[dict], keep_external: bool = True) -> None:
    path = cfg.path("conditions_csv")
    external = []
    if keep_external and path.exists():
        old = read_conditions(cfg)
        external = old[old["corruption"] == "external"].to_dict("records")
    ids = {r["condition_id"] for r in rows}
    rows = sorted(rows, key=lambda r: r["condition_id"]) + [r for r in external if r["condition_id"] not in ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CONDITION_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def add_external(cfg: Config, csv_path: str | Path) -> int:
    """Register outside videos with human labels (e.g. generated videos from ImplausiBench).

    CSV columns: path (video file), label ('plausible' or 'implausible'), optional id.
    Each video is re-encoded like the Physics-IQ clips, with a reversed copy for the paired setup.
    """
    pre = cfg["preprocess"]
    with open(csv_path, newline="") as f:
        items = list(csv.DictReader(f))
    existing = read_conditions(cfg).to_dict("records") if cfg.path("conditions_csv").exists() else []
    by_id = {r["condition_id"]: r for r in existing}
    out_dir = cfg.path("conditions_dir") / "external"
    added = 0
    for i, it in enumerate(items):
        label = it["label"].strip().lower()
        if label not in ("plausible", "implausible"):
            raise SystemExit(f"Row {i + 2}: label must be 'plausible' or 'implausible', got {it['label']!r}")
        ext_id = (it.get("id") or Path(it["path"]).stem).strip()
        cid = f"ext__{ext_id}"
        if cid in by_id:
            continue
        frames, fps = read_frames(it["path"], width=pre["width"])
        path, rev_path = out_dir / f"{ext_id}.mp4", out_dir / f"{ext_id}__rev.mp4"
        write_video(path, frames, fps, crf=pre["crf"], preset=pre["preset"])
        write_video(rev_path, frames[::-1], fps, crf=pre["crf"], preset=pre["preset"])
        by_id[cid] = {
            "condition_id": cid, "video_id": f"ext-{ext_id}", "scenario": it.get("scenario", ext_id),
            "category": it.get("category", "external"), "perspective": "", "take": 0,
            "condition": "external", "corruption": "external", "strength": "",
            "frames": len(frames), "fps": fps, "duration_s": round(len(frames) / fps, 3),
            "start_frame": 0, "length_frames": 0, "motion_overlap": 0, "motion_start": 0,
            "motion_end": 0, "has_motion": True, "detail": "",
            "path": _rel(cfg, path), "rev_path": _rel(cfg, rev_path),
            "sha256": sha256_file(path), "rev_sha256": sha256_file(rev_path), "human_label": label,
        }
        added += 1
    write_conditions(cfg, [r for r in by_id.values() if r["corruption"] != "external"]
                     + [r for r in by_id.values() if r["corruption"] == "external"], keep_external=False)
    return added


def write_check_page(cfg: Config, per_condition: int = 2, seed: int = 0) -> Path:
    """Side-by-side clean vs corrupted clips, to confirm by eye that each corruption looks wrong."""
    df = read_conditions(cfg)
    df = df[df["corruption"] != "external"]
    rng = np.random.default_rng(seed)
    videos = sorted(df["video_id"].unique())
    page_dir = cfg.path("conditions_dir")
    blocks = []
    for name in condition_names()[1:]:
        picks = rng.choice(videos, size=min(per_condition, len(videos)), replace=False)
        for vid in picks:
            row = df[(df["video_id"] == vid) & (df["condition"] == name)].iloc[0]
            clean = df[(df["video_id"] == vid) & (df["condition"] == "clean")].iloc[0]
            fps = float(row["fps"])
            where = (f"affects frames {row['start_frame']}–{row['start_frame'] + row['length_frames'] - 1} "
                     f"({row['start_frame'] / fps:.2f}–{(row['start_frame'] + row['length_frames']) / fps:.2f} s)")
            if row["corruption"] == "shuffle":
                where = f"segment order {row['detail']}"
            rel = lambda p: os.path.relpath(cfg.root / p, page_dir)
            blocks.append(f"""
<section><h2>{name} · {row['scenario']} · {row['perspective']} take {row['take']}</h2>
<p>{where}; motion detected {int(row['motion_start']) / fps:.2f}–{int(row['motion_end']) / fps:.2f} s; overlap {row['motion_overlap']}</p>
<div class="pair"><figure><video src="{rel(clean['path'])}" controls muted loop playsinline></video><figcaption>clean</figcaption></figure>
<figure><video src="{rel(row['path'])}" controls muted loop playsinline></video><figcaption>{name}</figcaption></figure></div></section>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corruption check</title><style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#dedcd6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#f0efec; --muted:#c3c2b7; --line:#3a3a37; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }}
section {{ border-top:1px solid var(--line); padding:12px 0; max-width:1100px; }}
h2 {{ font-size:16px; margin:0; }} p, figcaption {{ color:var(--muted); font-size:13px; margin:4px 0; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }} figure {{ margin:0; }}
video {{ width:100%; background:#000; border-radius:4px; }}
</style></head><body><h1>Corruption check</h1>
<p>Each corrupted clip should look clearly wrong next to its clean version. If one does not (for example a freeze placed where nothing moves), note it and adjust config.yaml before the full run.</p>
{''.join(blocks)}</body></html>"""
    path = page_dir / "check.html"
    path.write_text(page)
    return path
