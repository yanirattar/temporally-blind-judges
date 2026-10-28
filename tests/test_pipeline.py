"""End to end on 4 synthetic clips with the fake judge: build, sync run, batch, analyze."""

import copy
from pathlib import Path

import pandas as pd
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
    run("run", "main", "--fake")  # everything cached: must run nothing new
    run("analyze")

    calls = pd.read_csv(tmp_path / "results" / "answers.csv")
    per_setup = calls[calls["repeat"] == 0].groupby("setup").size().to_dict()
    assert per_setup == {"default": 52, "dense": 36, "checklist": 36, "paired": 36}
    summary = (tmp_path / "results" / "summary.md").read_text()
    assert "Flag rate by condition" in summary and "Consistency" in summary
    assert (tmp_path / "results" / "detection.png").stat().st_size > 10_000

    # Changing a prompt-affecting setting invalidates only that setup's answers.
    data["setups"]["dense"]["fps"] = 6
    cfg_path.write_text(yaml.safe_dump(data))
    from tbj.analyze import load_results
    from tbj.store import Store
    df = load_results(load_config(cfg_path), Store(cfg.path("store")).calls(), read_conditions(cfg))
    assert "dense" not in set(df["setup"]) and "default" in set(df["setup"])
