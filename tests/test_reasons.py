import json

import numpy as np
import pandas as pd

from tbj.analyze import reason_categories, reason_tags, sampling_fix


def test_reason_tags_on_pilot_style_reasons():
    assert reason_tags("The video is played in reverse: shards reassemble and fly upward against gravity.") == {"backward"}
    assert reason_tags("The basketball teleports directly into the crate between frames.") == {"jump"}
    assert reason_tags("The ball remains completely frozen in mid-air.") == {"pause"}
    assert reason_tags("The liquid level in the glass decreases instead of filling up.") == {"backward"}
    # Not time problems, or not backward in time:
    assert reason_tags("The rolling ball suddenly reverses its trajectory on the flat table.") == set()
    assert reason_tags("The water level inside the dispenser does not decrease at all.") == set()
    assert reason_tags("The orange ball phases directly through the crate.") == set()
    # Added after the sampling pilot (2026-11-18):
    assert reason_tags("The ball appears on the lower ramp before suddenly jumping to the top of the track.") == {"jump"}
    assert reason_tags("A large black balloon suddenly appears out of thin air on the table.") == {"jump"}
    assert reason_tags("It abruptly transforms into a duck with no continuous physical transition.") == {"jump"}
    assert reason_tags("The liquid level resets and drops between 00:05 and 00:06.") == {"jump"}
    assert reason_tags("The pieces reassemble against gravity to hang intact from the clip.") == {"backward"}
    assert reason_tags("The ball hangs in mid-air.") == {"pause"}


def test_reason_categories_counts_only_flagged_single_clip_answers():
    def row(setup, kind, cond, flag, reason):
        corruption, strength = ("clean", "") if cond == "clean" else cond.split("_")
        return {"setup": setup, "kind": kind, "repeat": 0, "corruption": corruption, "strength": strength,
                "flag": flag, "answer": json.dumps({"reason": reason})}
    df = pd.DataFrame([
        row("default", "single", "clean", 1, "The duck teleports into the box between frames."),
        row("default", "single", "clean", 1, "The dispenser level never drops."),
        row("default", "single", "clean", 0, "Natural fall."),
        row("default", "single", "reversal_strong", 1, "Played in reverse: the mug reassembles."),
        row("paired", "paired", "reversal_strong", 1, "Video B runs backward."),
    ])
    rc = reason_categories(df).set_index(["setup", "corruption", "strength"])
    clean = rc.loc[("default", "clean", "all")]
    assert clean["flagged"] == 2 and clean["jump"] == 0.5 and clean["none"] == 0.5
    assert rc.loc[("default", "reversal", "strong"), "backward"] == 1.0
    assert "paired" not in rc.index.get_level_values(0)


def test_sampling_fix_finds_fewer_false_alarms():
    rng = np.random.default_rng(3)
    rows = []
    for setup, fa, det in (("default", 0.45, 0.8), ("sampling_aware", 0.15, 0.75)):
        for sc in range(20):
            for v in range(5):
                for cond in ("clean", "freeze_medium", "shuffle_strong"):
                    corruption, strength = ("clean", "") if cond == "clean" else cond.split("_")
                    p = fa if cond == "clean" else det
                    rows.append({"setup": setup, "repeat": 0, "scenario": f"s{sc}", "condition_id": f"{sc}-{v}__{cond}",
                                 "corruption": corruption, "strength": strength, "flag": int(rng.random() < p)})
    sf = sampling_fix(pd.DataFrame(rows), b=500).set_index("measure")
    fa = sf.loc["False alarms (clean)"]
    assert fa["diff"] < -0.2 and fa["diff_hi"] < 0
    assert sf.loc["Balanced accuracy", "diff"] > 0
