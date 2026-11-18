"""End to end on 4 synthetic clips with the fake judge: build, sync run, batch, analyze."""

import copy
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tbj import physiq, synthetic
from tbj.build import build_all, read_conditions, write_check_page
from tbj.cli import main
from tbj.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]


def test_full_pipeline(tmp_path):
    data = copy.deepcopy(dict(load_config(REPO / "config.yaml")))
    data["preprocess"]["width"] = synthetic.W
    data["plans"]["pilot"]["clips"] = 2
    data["plans"]["consistency"]["clips"] = 2
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(data))
    cfg = Config(data, tmp_path)

    rows = synthetic.write_dataset(cfg.path("raw_dir"))
    keep = {"0001", "0004", "0013", "0016"}  # two scenes, two clips each
    physiq.write_manifest(cfg, [r for r in rows if r["video_id"] in keep])
    build_all(cfg, str(cfg_path), workers=1)
    cond = read_conditions(cfg)
    assert len(cond) == 4 * 13
    assert cond["sha256"].is_unique and cond["has_motion"].all()
    # Clean played backward is the strong reversal, frame for frame (they share one upload).
    by = cond.set_index("condition_id")
    for vid in keep:
        assert by.loc[f"{vid}__clean", "rev_sha256"] == by.loc[f"{vid}__reversal_strong", "sha256"]
    assert write_check_page(cfg).exists()

    run = lambda *argv: main(["--config", str(cfg_path), *argv])
    run("run", "pilot", "--fake")
    run("batch", "submit", "main", "--fake")
    run("batch", "submit", "setups", "--fake")
    run("batch", "collect", "--fake")
    run("run", "consistency", "--fake")
    run("run", "checklist_dense", "--fake")
    run("run", "main", "--fake")  # everything cached: must run nothing new
    run("analyze")

    calls = pd.read_csv(tmp_path / "results" / "answers.csv")
    per_setup = calls[calls["repeat"] == 0].groupby("setup").size().to_dict()
    assert per_setup == {"default": 52, "dense": 36, "checklist": 36, "paired": 36, "checklist_dense": 4}
    summary = (tmp_path / "results" / "summary.md").read_text()
    assert "Flag rate by condition" in summary and "Consistency" in summary
    assert "## Checklist answers" in summary and "Checklist, 4 frames/s" in summary
    assert "## H5:" in summary and "## H6:" in summary and "hand-labelled" not in summary  # no visibility labels yet
    run("spots")
    spots = pd.read_csv(tmp_path / "data" / "motion_at_spot.csv")
    assert len(spots) == 4 * 12 and spots["motion_at_spot"].notna().sum() == 4 * 8  # 3 freezes, 3 drops, 2 partial reversals
    run("spots")  # cached: measures nothing again

    # Visibility sample: a page to label in; a downloaded CSV in labels/ shows up in the summary
    # and its labels are shown again when the page is rebuilt.
    run("check", "--visibility")
    page = (tmp_path / "data" / "conditions" / "visibility.html").read_text()
    ids = re.findall(r'data-id="([^"]+)"', page)
    assert len(ids) == 8 * 4 and page.count('type="radio"') == 8 * 4 * 3 and 'id="dl"' in page
    assert not (tmp_path / "labels" / "visibility.csv").exists()
    labels = ["visible", "subtle", "invisible", "visible"] * 8
    (tmp_path / "labels").mkdir(exist_ok=True)
    pd.DataFrame({"condition_id": ids, "condition": [i.split("__")[1] for i in ids], "scenario": "x",
                  "perspective": "left", "take": 1, "label": labels, "note": ""}).to_csv(
        tmp_path / "labels" / "visibility.csv", index=False)
    run("check", "--visibility")
    assert f'"{ids[2]}": {{"label": "invisible"' in (tmp_path / "data" / "conditions" / "visibility.html").read_text()
    run("analyze")
    assert "hand-labelled" in (tmp_path / "results" / "summary.md").read_text()
    assert (tmp_path / "results" / "detection.png").stat().st_size > 10_000

    # Changing a prompt-affecting setting invalidates only that setup's answers.
    data["setups"]["dense"]["fps"] = 6
    cfg_path.write_text(yaml.safe_dump(data))
    from tbj.analyze import load_results
    from tbj.store import Store
    df = load_results(load_config(cfg_path), Store(cfg.path("store")).calls(), read_conditions(cfg))
    assert "dense" not in set(df["setup"]) and "default" in set(df["setup"])

    # Excluding a clip drops it from conditions.csv without re-rendering anything.
    data["exclude_videos"] = ["0004"]
    cfg_path.write_text(yaml.safe_dump(data))
    build_all(load_config(cfg_path), str(cfg_path), workers=1)
    cond = read_conditions(cfg)
    assert "0004" not in set(cond["video_id"]) and len(cond) == 3 * 13

    # An unquoted id (YAML would read 0103 as a number) is refused with a clear message.
    data["exclude_videos"] = [103]
    cfg_path.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit):
        build_all(load_config(cfg_path), str(cfg_path), workers=1)
