"""Physics-IQ: the clip manifest, downloads and the scene review page.

Physics-IQ has 66 scenes, each filmed from 3 perspectives (left, centre, right) in
2 takes, for 396 videos. File names look like
`0001_perspective-left_take-1_trimmed-ball-and-block-fall.mp4`.

Only scenes whose events are irreversible are usable: if a reversed clip can look
physically fine (a rotating teapot, a rolling ball), "reversed" is not a reliable
ground truth for "implausible". You decide this per scene by watching each scene
forward and backward on the review page (`tbj review`).
"""

from __future__ import annotations

import csv
import html
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources
from pathlib import Path

import numpy as np

from .config import Config
from .video import read_frames, write_video

FILENAME_RE = re.compile(r"^(\d{4})_perspective-(left|center|right)_take-(\d)_trimmed-(.+)\.mp4$")

LABELS = ("irreversible", "reversible", "static", "exclude")

# First-pass hints from the scene descriptions, NOT ground truth. Confirm every
# scene by watching it forward and backward on the review page.
SUGGESTED = {
    "likely_irreversible": [
        "ball-and-block-fall", "ball-in-basket", "ball-in-sand", "ball-rolls-off", "block-domino",
        "cut-orange", "cut-paper", "domino-in-juice", "dominos-with-space", "duck-and-dominos",
        "duck-falls-in-box", "fill-glass-red-drink", "juice-in-water", "liquid-on-duck",
        "liquid-overfill", "marble-run-x", "marble-run-y", "match", "match-blows-balloon",
        "mug-breaks", "napkin-soak", "paint-on-glass", "paper-fall-water", "paper-in-water",
        "paper-smoke", "potato-in-water", "siphon", "water-in-juice", "weight-on-ceramic",
        "weight-on-paper",
    ],
    "uncertain": [
        "ball-hits-duck", "ball-ramp", "ball-train", "balls-collide", "blow-balloon", "double-cradle",
        "lit-candle", "magnet-wrench", "roll-in-box", "rolling-reflection", "silk-cover",
        "single-cradle", "unstable-block-stack", "weight-on-pillow", "weight-protects-duck",
    ],
    "likely_reversible": [
        "ball-behind-rotating-paper", "ball-hits-nothing", "ball-rolls-on-glass", "light-on-block",
        "light-on-mug", "light-on-mug-block", "light-on-statue", "magnet-domino",
        "magnet-transparent-peakaboo", "mirror-ball-fall", "mirror-ball-rotate",
        "mirror-teapot-rotate", "roll-behind-box", "roll-front-box", "smiley-ball-rotates",
        "solid-ball-peakaboo", "stable-blocks", "teapot-rotates", "two-balls-pass",
    ],
    "static": ["duck-static", "glass-stays-same"],
}
SUGGESTION_BY_SCENE = {scene: hint for hint, scenes in SUGGESTED.items() for scene in scenes}

MANIFEST_FIELDS = [
    "video_id", "scenario", "perspective", "take", "category", "filename",
    "suggested", "label", "description",
]


def parse_filename(name: str) -> dict:
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Not a Physics-IQ file name: {name}")
    return {"video_id": m.group(1), "perspective": m.group(2), "take": int(m.group(3)), "scenario": m.group(4)}


# ---------------------------------------------------------------- manifest

def read_manifest(cfg: Config) -> list[dict]:
    path = cfg.path("manifest")
    if not path.exists():
        raise SystemExit(f"No manifest at {path}. Run `tbj init` first.")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["take"] = int(r["take"])
    return rows


def write_manifest(cfg: Config, rows: list[dict]) -> None:
    path = cfg.path("manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def init_manifest(cfg: Config, force: bool = False) -> Path:
    path = cfg.path("manifest")
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists. Use --force to rebuild it (labels are kept).")
    old_labels = {}
    if path.exists():
        old_labels = {r["video_id"]: r.get("label", "") for r in read_manifest(cfg)}
    with resources.files("tbj.resources").joinpath("physiq_descriptions.csv").open(newline="") as f:
        source = list(csv.DictReader(f))
    rows = []
    for s in source:
        meta = parse_filename(s["scenario"])
        rows.append({
            **meta,
            "category": s["category"],
            "filename": s["scenario"],
            "suggested": SUGGESTION_BY_SCENE.get(meta["scenario"], "uncertain"),
            "label": old_labels.get(meta["video_id"], ""),
            "description": s["description"],
        })
    write_manifest(cfg, rows)
    return path


def scenes(rows: list[dict]) -> list[str]:
    return sorted({r["scenario"] for r in rows})


def review_rows(rows: list[dict]) -> list[dict]:
    """One video per scene (centre perspective, take 1) for the review page."""
    return [r for r in rows if r["perspective"] == "center" and r["take"] == 1]


def selected_rows(rows: list[dict]) -> list[dict]:
    """All videos of scenes labelled 'irreversible'."""
    return [r for r in rows if r.get("label") == "irreversible"]


def apply_labels(cfg: Config, labels_csv: str | Path) -> dict:
    """Set the label of every video of each scene from a scene,label CSV."""
    with open(labels_csv, newline="") as f:
        labels = {r["scenario"].strip(): r["label"].strip() for r in csv.DictReader(f) if r.get("label")}
    bad = {s: l for s, l in labels.items() if l not in LABELS}
    if bad:
        raise SystemExit(f"Unknown labels {bad}. Allowed: {', '.join(LABELS)}")
    rows = read_manifest(cfg)
    known = scenes(rows)
    unknown = sorted(set(labels) - set(known))
    if unknown:
        raise SystemExit(f"Unknown scenes in {labels_csv}: {unknown}")
    for r in rows:
        if r["scenario"] in labels:
            r["label"] = labels[r["scenario"]]
    write_manifest(cfg, rows)
    counts: dict[str, int] = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    return counts


# ---------------------------------------------------------------- downloads

def raw_path(cfg: Config, row: dict) -> Path:
    return cfg.path("raw_dir") / f"take-{row['take']}" / row["filename"]


def remote_url(cfg: Config, row: dict) -> str:
    p = cfg["physiq"]
    vid, rest = row["filename"].split("_", 1)
    name = f"{vid}_full-videos_{p['fps']}FPS_{rest}"
    return f"{p['base_url'].rstrip('/')}/full-videos/take-{row['take']}/{p['fps']}FPS/{name}"


def _download_one(url: str, dest: Path, retries: int = 4) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return dest.stat().st_size
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise RuntimeError(
                    f"HTTP {e.code} for {url}. If the bucket needs sign-in, install the Google Cloud CLI and run:\n"
                    f"  gcloud storage cp {url.replace('https://storage.googleapis.com/', 'gs://')} {dest}"
                ) from e
            err: Exception = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 ** attempt)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Download failed after {retries} tries: {url} ({err})")


def download(cfg: Config, rows: list[dict], workers: int = 4) -> tuple[int, int]:
    """Download the given manifest rows; skips files already on disk. Returns (new, skipped)."""
    todo = [r for r in rows if not raw_path(cfg, r).exists()]
    skipped = len(rows) - len(todo)
    if not todo:
        return 0, skipped
    done, total_bytes = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, remote_url(cfg, r), raw_path(cfg, r)): r for r in todo}
        for fut in as_completed(futures):
            total_bytes += fut.result()
            done += 1
            print(f"  [{done}/{len(todo)}] {futures[fut]['filename']}  ({total_bytes / 1e6:.0f} MB so far)")
    return done, skipped


# ---------------------------------------------------------------- review page

def build_review_previews(cfg: Config, rows: list[dict], width: int = 480) -> list[dict]:
    """Small forward and reversed previews of one video per scene."""
    out_dir = cfg.path("review_dir")
    items = []
    for r in rows:
        src = raw_path(cfg, r)
        if not src.exists():
            print(f"  missing {src.name}: run `tbj download --set review` first")
            continue
        fwd, rev = out_dir / f"{r['scenario']}_fwd.mp4", out_dir / f"{r['scenario']}_rev.mp4"
        if not (fwd.exists() and rev.exists()):
            frames, fps = read_frames(src, width=width)
            write_video(fwd, frames, fps, crf=28)
            write_video(rev, frames[::-1], fps, crf=28)
            print(f"  previews for {r['scenario']}")
        items.append({**r, "fwd": fwd.name, "rev": rev.name})
    return items


def write_review_page(cfg: Config, items: list[dict]) -> Path:
    """An HTML page to label each scene; it downloads a scene,label CSV for `tbj apply-labels`."""
    cards = []
    for it in items:
        current = it.get("label", "")
        options = "".join(
            f'<option value="{l}"{" selected" if l == current else ""}>{l}</option>' for l in ("",) + LABELS
        ).replace('<option value=""></option>', '<option value="">choose…</option>')
        cards.append(f"""
<section class="card" data-scene="{html.escape(it['scenario'])}">
  <h2>{html.escape(it['scenario'])} <span class="cat">{html.escape(it['category'])}</span></h2>
  <p class="desc">{html.escape(it['description'])}</p>
  <div class="pair">
    <figure><video src="{it['fwd']}" controls muted loop preload="metadata" playsinline></video><figcaption>Forward (real)</figcaption></figure>
    <figure><video src="{it['rev']}" controls muted loop preload="metadata" playsinline></video><figcaption>Reversed</figcaption></figure>
  </div>
  <p class="hint">Suggestion from the description: <b>{html.escape(it['suggested'])}</b></p>
  <label>Label <select>{options}</select></label>
</section>""")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scene review</title>
<style>
  :root {{ --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#dedcd6; --accent:#2a78d6; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#f0efec; --muted:#c3c2b7; --line:#3a3a37; --accent:#3987e5; }} }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }}
  header {{ position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; z-index:1; }}
  header p {{ margin:4px 0; color:var(--muted); max-width:900px; }}
  main {{ padding:16px; display:grid; gap:16px; grid-template-columns:repeat(auto-fill, minmax(min(100%, 620px), 1fr)); }}
  .card {{ border:1px solid var(--line); border-radius:8px; padding:12px 16px; }}
  h2 {{ font-size:16px; margin:0 0 4px; }} .cat {{ color:var(--muted); font-weight:400; font-size:13px; }}
  .desc, .hint {{ color:var(--muted); font-size:13px; margin:4px 0 8px; }}
  .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
  figure {{ margin:0; }} video {{ width:100%; border-radius:4px; background:#000; }}
  figcaption {{ font-size:12px; color:var(--muted); }}
  select, button {{ font:inherit; padding:4px 8px; }}
  button {{ background:var(--accent); color:#fff; border:0; border-radius:6px; padding:6px 12px; cursor:pointer; }}
  #count {{ margin-left:12px; color:var(--muted); }}
</style></head><body>
<header>
  <strong>Scene review</strong>
  <p>For each scene, watch both clips. Label it <b>irreversible</b> only if the reversed clip looks clearly physically impossible
  (spills un-spilling, objects rising to where they were dropped, pieces reassembling). Use <b>reversible</b> when the reversed clip could pass as real,
  <b>static</b> when nothing moves, <b>exclude</b> for anything else. Choices are kept in this browser until you download them.</p>
  <button id="dl">Download labels.csv</button><span id="count"></span>
</header>
<main>{''.join(cards)}</main>
<script>
const KEY = 'tbj-scene-labels';
function load() {{ try {{ return JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{ return {{}}; }} }}
function save(v) {{ try {{ localStorage.setItem(KEY, JSON.stringify(v)); }} catch (e) {{}} }}
const saved = load();
const cards = [...document.querySelectorAll('.card')];
function refresh() {{
  const n = cards.filter(c => c.querySelector('select').value).length;
  document.getElementById('count').textContent = n + ' of ' + cards.length + ' scenes labelled';
}}
for (const c of cards) {{
  const s = c.querySelector('select'), scene = c.dataset.scene;
  if (saved[scene] && !s.value) s.value = saved[scene];
  s.addEventListener('change', () => {{ const v = load(); v[scene] = s.value; save(v); refresh(); }});
}}
refresh();
document.getElementById('dl').addEventListener('click', () => {{
  const lines = ['scenario,label'];
  for (const c of cards) {{ const v = c.querySelector('select').value; if (v) lines.push(c.dataset.scene + ',' + v); }}
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\\n') + '\\n'], {{type: 'text/csv'}}));
  a.download = 'labels.csv'; a.click();
}});
</script></body></html>"""
    path = cfg.path("review_dir") / "review.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path


def manifest_summary(rows: list[dict]) -> dict:
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["scenario"], r.get("label", ""))
    labelled = {s: l for s, l in by_scene.items() if l}
    counts: dict[str, int] = {}
    for l in labelled.values():
        counts[l] = counts.get(l, 0) + 1
    return {"scenes": len(by_scene), "labelled": len(labelled), "by_label": counts,
            "selected_videos": len(selected_rows(rows))}


def dump_json(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
