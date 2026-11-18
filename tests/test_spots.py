import numpy as np
import pandas as pd

from tbj.analyze import reversal_jumps, short_action_gain, visibility
from tbj.motion import relative_motion
from tbj.spots import spot_motion


def test_relative_motion_is_zero_at_baseline_and_one_at_peak():
    profile = np.full(100, 0.01, dtype=np.float32)
    profile[40:50] = 0.2
    rel = relative_motion(profile)
    assert abs(rel.max() - 1) < 1e-6 and rel[:30].max() == 0 and rel[80:].max() == 0
    assert np.all(relative_motion(np.zeros(50)) == 0)


def test_spot_motion_reads_the_affected_frames():
    rel = np.zeros(240)
    rel[100:110] = 1.0
    assert spot_motion(rel, "drop", 100, 8) == 1.0             # skipped frames all moving
    assert spot_motion(rel, "drop", 200, 8) == 0.0             # skipped frames still
    assert spot_motion(rel, "freeze", 104, 30) == 1.0          # stops mid-motion; its length doesn't matter
    assert spot_motion(rel, "freeze", 150, 30) == 0.0          # stops where nothing moves
    assert spot_motion(rel, "reversal", 95, 20) == 1.0         # partial reversal: strongest motion in the stretch
    assert spot_motion(rel, "drop", 96, 30) == 1.0             # a longer drop over the same event is not "less visible"
    assert np.isnan(spot_motion(rel, "reversal", 0, 240))      # full reversal: whole clip
    assert np.isnan(spot_motion(rel, "shuffle", 0, 240))


def _answers(p_flag, n_scenes=20, per_scene=6, seed=0):
    """Fake default and dense answers. p_flag(setup, corruption, motion_s, spot) -> flag probability."""
    rng = np.random.default_rng(seed)
    rows = []
    for setup in ("default", "dense"):
        for s in range(n_scenes):
            for v in range(per_scene):
                motion_s = 0.5 if v < per_scene // 2 else 3.0
                for cond in ("clean", "freeze_medium", "drop_strong", "shuffle_strong"):
                    corruption, strength = ("clean", "") if cond == "clean" else cond.split("_")
                    spot = np.nan if corruption in ("clean", "shuffle") else (0.05 if (s + v) % 2 else 0.8)
                    rows.append({"setup": setup, "repeat": 0, "condition_id": f"{s}-{v}__{cond}",
                                 "video_id": f"{s}-{v}", "scenario": f"scene{s}", "corruption": corruption,
                                 "strength": strength, "motion_s": motion_s, "motion_at_spot": spot,
                                 "flag": int(rng.random() < p_flag(setup, corruption, motion_s, spot))})
    return pd.DataFrame(rows)


def test_short_action_gain_finds_a_larger_gain_on_short_actions():
    def p(setup, corruption, motion_s, spot):
        if setup == "dense" and motion_s < 1:
            return 0.8
        return 0.3
    h5 = short_action_gain(_answers(p), b=500)
    row = h5[h5["corruption"] == "all"].iloc[0]
    assert row["short_gain"] > 0.35 and abs(row["long_gain"]) < 0.1
    assert row["diff_lo"] > 0


def test_visibility_shares_and_interval():
    labels = pd.DataFrame({
        "condition": ["freeze_mild"] * 10 + ["drop_strong"] * 10 + ["reversal_mild"] * 4,
        "label": ["visible"] * 8 + ["subtle", "invisible"] + ["Visible "] * 5 + ["invisible"] * 5 + ["", "x", "subtle", "visible"],
    })
    v = visibility(labels).set_index("corruption")
    assert v.loc["freeze", "n"] == 10 and abs(v.loc["freeze", "invisible"] - 0.1) < 1e-9
    assert v.loc["freeze", "invisible_lo"] < 0.1 < v.loc["freeze", "invisible_hi"] < 0.5
    assert v.loc["drop", "visible"] == 0.5 and v.loc["drop", "invisible"] == 0.5   # labels are trimmed, any case
    assert v.loc["partial reversal", "n"] == 2                                      # blank and unknown labels ignored
    assert visibility(pd.DataFrame()).empty


def test_reversal_jumps_finds_partial_reversals_caught_more():
    rng = np.random.default_rng(1)
    rows = []
    for setup in ("default", "dense"):
        for sc in range(20):
            for v in range(6):
                for strength, p in (("mild", 0.6), ("medium", 0.7), ("strong", 0.2)):
                    rows.append({"setup": setup, "repeat": 0, "scenario": f"scene{sc}", "video_id": f"{sc}-{v}",
                                 "corruption": "reversal", "strength": strength, "flag": int(rng.random() < p)})
    rj = reversal_jumps(pd.DataFrame(rows), b=500).set_index("setup")
    assert rj.loc["default", "medium_diff"] > 0.3 and rj.loc["default", "medium_lo"] > 0
    assert rj.loc["dense", "mild_n"] == 120
