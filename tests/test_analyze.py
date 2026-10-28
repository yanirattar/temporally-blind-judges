import numpy as np
import pandas as pd

from tbj.analyze import comparisons, consistency, metrics


def _df(rates: dict, n_scenes=20, per_scene=6, seed=0):
    """Fake answers with known flag rates per (setup, condition)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (setup, cond), p in rates.items():
        corruption, strength = ("clean", "") if cond == "clean" else cond.split("_")
        for s in range(n_scenes):
            for v in range(per_scene):
                flag = int(rng.random() < p)
                rows.append({"setup": setup, "repeat": 0, "condition_id": f"{s}-{v}__{cond}",
                             "scenario": f"scene{s}", "corruption": corruption, "strength": strength,
                             "flag": flag, "score": flag * 0.8 + rng.random() * 0.2, "any_check": None,
                             "kind": "single", "answer": "{}"})
    return pd.DataFrame(rows)


def test_metrics_recover_known_rates_and_cis_cover_them():
    df = _df({("default", "clean"): 0.1, ("default", "reversal_strong"): 0.6,
              ("paired", "clean"): 0.1, ("paired", "reversal_strong"): 0.9})
    m = metrics(df, b=500)
    rev = m[(m.setup == "default") & (m.corruption == "reversal") & (m.strength == "strong")].iloc[0]
    assert abs(rev.rate - 0.6) < 0.08 and rev.rate_lo < rev.rate < rev.rate_hi
    assert rev.rate_lo < 0.6 < rev.rate_hi
    fa = m[(m.setup == "default") & (m.corruption == "clean")].iloc[0]
    assert abs(fa.rate - 0.1) < 0.06
    assert 0.7 < rev.balanced_accuracy < 0.8
    assert rev.auroc > 0.7


def test_comparison_finds_the_better_setup():
    df = _df({("default", "clean"): 0.1, ("default", "reversal_strong"): 0.4,
              ("paired", "clean"): 0.1, ("paired", "reversal_strong"): 0.9})
    c = comparisons(df, b=500)
    row = c[(c.setup == "paired") & (c.measure == "detection (all corrupted)")].iloc[0]
    assert row["diff"] > 0.35 and row.diff_lo > 0


def test_consistency_kappa():
    df = _df({("default", "clean"): 0.5})
    again = df.copy()
    again["repeat"] = 1
    c = consistency(pd.concat([df, again]))
    assert c.iloc[0].agreement == 1.0 and c.iloc[0].kappa == 1.0
