"""Metrics, confidence intervals, comparisons, figure and summary.

Definitions (per setup, first run only, i.e. repeat 0):
  detection rate    share of corrupted clips the judge flags as implausible
  false-alarm rate  share of clean clips the judge flags
  balanced accuracy (detection + 1 - false alarm) / 2, against that setup's clean clips
  AUROC             how well the judge's implausibility score separates corrupted from clean

95% confidence intervals come from a cluster bootstrap over scenes: clips of the
same scene are correlated, so whole scenes are resampled, not single clips.
Only answers that match the current config (prompt text, frame rate, model, clip
bytes) are used, so results from old prompt versions never mix in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .corrupt import CORRUPTIONS, STRENGTHS
from .judge import CHECKLIST_ITEMS, load_prompt

SETUP_ORDER = ["default", "dense", "checklist", "checklist_any", "checklist_dense", "paired"]
SETUP_LABELS = {"default": "Default", "dense": "Dense sampling", "checklist": "Time checklist",
                "checklist_any": "Checklist (any item)", "checklist_dense": "Checklist, 4 frames/s",
                "paired": "Paired with reverse"}
# Categorical slots 1-4 of the reference palette (light mode), fixed per setup.
SETUP_COLORS = {"default": "#2a78d6", "dense": "#eb6834", "checklist": "#1baf7a", "paired": "#eda100"}


# ------------------------------------------------------------------ loading

def load_results(cfg: Config, calls: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    """Successful calls that match the current config, joined to their conditions."""
    if calls.empty:
        return calls
    current = {}
    for name, s in cfg["setups"].items():
        current[name] = (load_prompt(s["prompt"]).hash, float(s["fps"]))
    calls = calls[calls["status"] == "ok"].copy()
    keep = calls.apply(lambda r: r["setup"] in current
                       and (r["prompt_hash"], float(r["fps"])) == current[r["setup"]]
                       and r["model"] == cfg["judge"]["model"], axis=1)
    calls = calls[keep]
    cols = ["condition_id", "video_id", "scenario", "category", "perspective", "take", "condition",
            "corruption", "strength", "sha256", "human_label", "motion_overlap"]
    df = calls.merge(cond[cols], on="condition_id", how="inner")
    df = df[df["clip_sha"] == df["sha256"]].copy()
    df["flag"] = df["flag"].astype(int)
    return df


def _with_any_check(df: pd.DataFrame) -> pd.DataFrame:
    """Adds pseudo-setup 'checklist_any': flagged if any checklist question is answered yes."""
    chk = df[df["setup"] == "checklist"].copy()
    if chk.empty:
        return df
    chk["setup"] = "checklist_any"
    chk["flag"] = (chk["flag"].astype(bool) | chk["any_check"].fillna(0).astype(bool)).astype(int)
    return pd.concat([df, chk], ignore_index=True)


# ------------------------------------------------------------------ bootstrap

def _weights(scenes: list[str], b: int, seed: int) -> np.ndarray:
    n = len(scenes)
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=b).astype(float)


def _ci(samples: np.ndarray) -> tuple[float, float]:
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return (np.nan, np.nan)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _rate(sub: pd.DataFrame, scenes: list[str], w: np.ndarray, col: str = "flag"):
    """Point estimate and bootstrap samples of the mean of `col` over rows of `sub`."""
    g = sub.groupby("scenario")[col].agg(["sum", "count"]).reindex(scenes, fill_value=0)
    flags, totals = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    point = flags.sum() / totals.sum() if totals.sum() else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (w @ flags) / (w @ totals)
    return point, boot


def _auroc(pos: pd.DataFrame, neg: pd.DataFrame, scenes: list[str], w: np.ndarray):
    """AUROC of score (corrupted vs clean), with scene-weighted bootstrap samples."""
    if pos.empty or neg.empty:
        return np.nan, np.full(len(w), np.nan)
    idx = {s: i for i, s in enumerate(scenes)}
    sp, sn = pos["score"].to_numpy(float), neg["score"].to_numpy(float)
    m = (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])
    point = float(m.mean())
    wp = w[:, [idx[s] for s in pos["scenario"]]]
    wn = w[:, [idx[s] for s in neg["scenario"]]]
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.einsum("bi,ij,bj->b", wp, m, wn) / (wp.sum(1) * wn.sum(1))
    return point, boot


# ------------------------------------------------------------------ metrics

def metrics(df: pd.DataFrame, b: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per setup x condition group (clean, each corruption x strength, all corrupted)."""
    df = _with_any_check(df[(df["repeat"] == 0) & (df["corruption"] != "external")])
    if df.empty:
        return pd.DataFrame()
    scenes = sorted(df["scenario"].unique())
    w = _weights(scenes, b, seed)
    rows = []
    for setup in [s for s in SETUP_ORDER if s in set(df["setup"])]:
        d = df[df["setup"] == setup]
        clean = d[d["corruption"] == "clean"]
        fa, fa_boot = _rate(clean, scenes, w)
        rows.append({"setup": setup, "corruption": "clean", "strength": "", "n": len(clean),
                     "rate": fa, "rate_lo": _ci(fa_boot)[0], "rate_hi": _ci(fa_boot)[1],
                     "balanced_accuracy": np.nan, "ba_lo": np.nan, "ba_hi": np.nan,
                     "auroc": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan})
        groups = [(c, s) for c in CORRUPTIONS for s in STRENGTHS] + [("all", "")]
        for c, s in groups:
            sub = d[d["corruption"] != "clean"] if c == "all" else d[(d["corruption"] == c) & (d["strength"] == s)]
            if sub.empty:
                continue
            det, det_boot = _rate(sub, scenes, w)
            ba_boot = (det_boot + 1 - fa_boot) / 2
            auc, auc_boot = _auroc(sub, clean, scenes, w)
            rows.append({"setup": setup, "corruption": c, "strength": s, "n": len(sub),
                         "rate": det, "rate_lo": _ci(det_boot)[0], "rate_hi": _ci(det_boot)[1],
                         "balanced_accuracy": (det + 1 - fa) / 2 if np.isfinite(fa) else np.nan,
                         "ba_lo": _ci(ba_boot)[0], "ba_hi": _ci(ba_boot)[1],
                         "auroc": auc, "auroc_lo": _ci(auc_boot)[0], "auroc_hi": _ci(auc_boot)[1]})
    return pd.DataFrame(rows)


def comparisons(df: pd.DataFrame, baseline: str = "default", b: int = 2000, seed: int = 1) -> pd.DataFrame:
    """Each setup minus the baseline, on the clips both were run on."""
    df = _with_any_check(df[(df["repeat"] == 0) & (df["corruption"] != "external")])
    if baseline not in set(df["setup"]):
        return pd.DataFrame()
    scenes = sorted(df["scenario"].unique())
    w = _weights(scenes, b, seed)
    base = df[df["setup"] == baseline]
    rows = []
    for setup in [s for s in SETUP_ORDER if s in set(df["setup"]) and s != baseline]:
        other = df[df["setup"] == setup]
        common = set(base["condition_id"]) & set(other["condition_id"])
        for label, select in [("false alarm (clean)", lambda x: x["corruption"] == "clean"),
                              ("detection (all corrupted)", lambda x: x["corruption"] != "clean"),
                              *[(f"detection ({c})", (lambda c: lambda x: x["corruption"] == c)(c)) for c in CORRUPTIONS]]:
            a = base[base["condition_id"].isin(common) & select(base)]
            o = other[other["condition_id"].isin(common) & select(other)]
            if a.empty:
                continue
            pa, ba = _rate(a, scenes, w)
            po, bo = _rate(o, scenes, w)
            lo, hi = _ci(bo - ba)
            rows.append({"setup": setup, "measure": label, "n": len(o), "baseline": pa, "setup_rate": po,
                         "diff": po - pa, "diff_lo": lo, "diff_hi": hi})
    return pd.DataFrame(rows)


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement and Cohen's kappa between repeat 0 and repeat 1 of the same question."""
    rows = []
    for setup, d in df[df["corruption"] != "external"].groupby("setup"):
        r0 = d[d["repeat"] == 0].set_index("condition_id")["flag"]
        r1 = d[d["repeat"] == 1].set_index("condition_id")["flag"]
        common = r0.index.intersection(r1.index)
        if len(common) == 0:
            continue
        a, c = r0.loc[common].to_numpy(), r1.loc[common].to_numpy()
        agree = float((a == c).mean())
        pe = a.mean() * c.mean() + (1 - a.mean()) * (1 - c.mean())
        kappa = (agree - pe) / (1 - pe) if pe < 1 else np.nan
        rows.append({"setup": setup, "n": len(common), "agreement": agree, "kappa": kappa})
    return pd.DataFrame(rows)


def external_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Judge vs human labels on outside videos (e.g. generated videos)."""
    d = df[(df["corruption"] == "external") & (df["repeat"] == 0)]
    rows = []
    for setup, g in _with_any_check(d).groupby("setup"):
        human = (g["human_label"] == "implausible").astype(int).to_numpy()
        judge = g["flag"].to_numpy()
        acc = float((human == judge).mean())
        pe = human.mean() * judge.mean() + (1 - human.mean()) * (1 - judge.mean())
        rows.append({"setup": setup, "n": len(g), "accuracy": acc,
                     "kappa": (acc - pe) / (1 - pe) if pe < 1 else np.nan,
                     "judge_flag_rate": float(judge.mean()), "human_implausible_rate": float(human.mean())})
    return pd.DataFrame(rows)


def costs(calls: pd.DataFrame) -> pd.DataFrame:
    if calls.empty:
        return pd.DataFrame()
    g = calls.groupby(["setup", "mode"]).agg(
        calls=("key", "count"), ok=("status", lambda s: int((s == "ok").sum())),
        input_tokens=("input_tokens", "mean"), output_tokens=("output_tokens", "mean"),
        thought_tokens=("thought_tokens", "mean"), cost_usd=("cost_usd", "sum")).reset_index()
    g["cost_per_call"] = g["cost_usd"] / g["calls"]
    return g


def stated_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """For flagged corrupted clips (single-clip setups): how often the judge names a time problem."""
    d = df[(df["repeat"] == 0) & (df["kind"] == "single") & (df["flag"] == 1) & (~df["corruption"].isin(["clean", "external"]))]
    if d.empty:
        return pd.DataFrame()
    d = d.assign(problem=d["answer"].map(lambda a: json.loads(a).get("problem_type", "")))
    return (d.groupby(["setup", "corruption"])["problem"]
            .apply(lambda s: float((s == "time_direction_or_continuity").mean())).rename("share_time_problem")
            .reset_index())


# ------------------------------------------------------------------ outputs

def _pct(x: float) -> str:
    return "–" if not np.isfinite(x) else f"{100 * x:.0f}%"


def _cell(rate, lo, hi) -> str:
    if not np.isfinite(rate):
        return "–"
    return f"{_pct(rate)} [{_pct(lo)}–{_pct(hi)}]"


def figure(m: pd.DataFrame, path: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setups = [s for s in ["default", "dense", "checklist", "paired"] if s in set(m["setup"])]
    if m.empty or not setups:
        return None
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": grid,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.6), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.8]}, facecolor=surface)
    x = np.arange(len(STRENGTHS))
    offsets = np.linspace(-0.12, 0.12, len(setups))
    for ax, c in zip(axes[:4], CORRUPTIONS):
        for off, s in zip(offsets, setups):
            sub = m[(m["setup"] == s) & (m["corruption"] == c)].set_index("strength").reindex(STRENGTHS)
            y = sub["rate"].to_numpy(float)
            err = np.vstack([y - sub["rate_lo"].to_numpy(float), sub["rate_hi"].to_numpy(float) - y])
            ax.errorbar(x + off, y, yerr=err, color=SETUP_COLORS[s], lw=2, marker="o", ms=6,
                        mec=surface, mew=1.5, elinewidth=1, capsize=0, label=SETUP_LABELS[s])
        ax.set_title(c.capitalize(), color=ink, fontsize=11, loc="left")
        ax.set_xticks(x, [s.capitalize() for s in STRENGTHS])
    ax = axes[4]
    for i, s in enumerate(setups):
        row = m[(m["setup"] == s) & (m["corruption"] == "clean")]
        if row.empty:
            continue
        r = row.iloc[0]
        ax.errorbar([i], [r["rate"]], yerr=[[r["rate"] - r["rate_lo"]], [r["rate_hi"] - r["rate"]]],
                    color=SETUP_COLORS[s], marker="o", ms=6, mec=surface, mew=1.5, elinewidth=1, capsize=0)
    short = {"default": "Default", "dense": "Dense", "checklist": "Checklist", "paired": "Paired"}
    ax.set_xticks(range(len(setups)), [short[s] for s in setups], rotation=30, ha="right")
    ax.set_title("Clean: false alarms", color=ink, fontsize=11, loc="left")
    for ax in axes:
        ax.set_facecolor(surface)
        ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Clips flagged as implausible")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(setups), frameon=False, bbox_to_anchor=(0.5, 1.0),
               labelcolor=ink)
    fig.suptitle("How often the judge flags time-corrupted clips (95% CI, scene bootstrap)",
                 x=0.01, y=1.08, ha="left", color=ink, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
    return path


def summary_markdown(cfg: Config, df: pd.DataFrame, m: pd.DataFrame, comp: pd.DataFrame, cons: pd.DataFrame,
                     ext: pd.DataFrame, cost: pd.DataFrame, reasons: pd.DataFrame, total_spent: float) -> str:
    lines = ["# Results", ""]
    n_scenes = df["scenario"].nunique() if not df.empty else 0
    lines += [f"Judge `{cfg['judge']['model']}` · {len(df)} answers used · {df['video_id'].nunique() if not df.empty else 0} "
              f"clips from {n_scenes} scenes · total API spend ${total_spent:.2f}", ""]
    if not m.empty:
        setups = [s for s in SETUP_ORDER if s in set(m["setup"])]
        lines += ["## Flag rate by condition", "",
                  "Share of clips judged implausible, with 95% CI. For corrupted clips higher is better; "
                  "for clean clips (false alarms) lower is better.", "",
                  "| Condition | " + " | ".join(SETUP_LABELS[s] for s in setups) + " |",
                  "|---|" + "---|" * len(setups)]
        order = [("clean", "")] + [(c, s) for c in CORRUPTIONS for s in STRENGTHS] + [("all", "")]
        for c, s in order:
            cells = []
            for st in setups:
                r = m[(m["setup"] == st) & (m["corruption"] == c) & (m["strength"] == s)]
                cells.append(_cell(*r.iloc[0][["rate", "rate_lo", "rate_hi"]]) if not r.empty else "–")
            name = "Clean (false alarm)" if c == "clean" else ("All corrupted" if c == "all" else f"{c.capitalize()}, {s}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "| Overall | " + " | ".join(SETUP_LABELS[s] for s in setups) + " |", "|---|" + "---|" * len(setups)]
        for label, col in [("Balanced accuracy", "balanced_accuracy"), ("AUROC", "auroc")]:
            cells = []
            for st in setups:
                r = m[(m["setup"] == st) & (m["corruption"] == "all")]
                lo, hi = ("ba_lo", "ba_hi") if col == "balanced_accuracy" else ("auroc_lo", "auroc_hi")
                cells.append("–" if r.empty or not np.isfinite(r.iloc[0][col]) else
                             f"{r.iloc[0][col]:.2f} [{r.iloc[0][lo]:.2f}–{r.iloc[0][hi]:.2f}]")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    if not comp.empty:
        lines += ["## Each setup vs default", "", "Difference in flag rate on the same clips (percentage points, 95% CI).", "",
                  "| Setup | Measure | Default | Setup | Difference |", "|---|---|---|---|---|"]
        for _, r in comp.iterrows():
            lines.append(f"| {SETUP_LABELS.get(r['setup'], r['setup'])} | {r['measure']} | {_pct(r['baseline'])} | "
                         f"{_pct(r['setup_rate'])} | {100 * r['diff']:+.0f} pts [{100 * r['diff_lo']:+.0f}, {100 * r['diff_hi']:+.0f}] |")
        lines.append("")
    if not cons.empty:
        lines += ["## Consistency (same question asked twice)", "", "| Setup | Clips | Agreement | Cohen's kappa |", "|---|---|---|---|"]
        for _, r in cons.iterrows():
            lines.append(f"| {SETUP_LABELS.get(r['setup'], r['setup'])} | {r['n']} | {_pct(r['agreement'])} | {r['kappa']:.2f} |")
        lines.append("")
    if not ext.empty:
        lines += ["## Agreement with human labels (outside videos)", "", "| Setup | Videos | Accuracy | Cohen's kappa |", "|---|---|---|---|"]
        for _, r in ext.iterrows():
            lines.append(f"| {SETUP_LABELS.get(r['setup'], r['setup'])} | {r['n']} | {_pct(r['accuracy'])} | {r['kappa']:.2f} |")
        lines.append("")
    if not reasons.empty:
        lines += ["## Does the judge name a time problem when it flags a clip?", "",
                  "| Setup | Corruption | Share naming a time problem |", "|---|---|---|"]
        for _, r in reasons.iterrows():
            lines.append(f"| {SETUP_LABELS.get(r['setup'], r['setup'])} | {r['corruption']} | {_pct(r['share_time_problem'])} |")
        lines.append("")
    if not cost.empty:
        lines += ["## Cost", "", "| Setup | Mode | Calls | Avg input tokens | Avg output + thinking | Cost |", "|---|---|---|---|---|---|"]
        for _, r in cost.iterrows():
            lines.append(f"| {SETUP_LABELS.get(r['setup'], r['setup'])} | {r['mode']} | {r['calls']} | {r['input_tokens']:.0f} | "
                         f"{r['output_tokens'] + r['thought_tokens']:.0f} | ${r['cost_usd']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def checklist_answers(df: pd.DataFrame) -> pd.DataFrame:
    """For checklist setups: how often each time question is answered yes, on clean and corrupted clips."""
    d = df[(df["repeat"] == 0) & (df["kind"] == "checklist") & (df["corruption"] != "external")]
    if d.empty:
        return pd.DataFrame()
    d = d.assign(clips=np.where(d["corruption"] == "clean", "clean", "corrupted"))
    answers = d["answer"].map(json.loads)
    for item in CHECKLIST_ITEMS:
        d[item] = answers.map(lambda a, item=item: bool(a.get(item)))
    rows = []
    for (setup, clips), g in d.groupby(["setup", "clips"]):
        flagged = g[g["flag"] == 1]
        rows.append({"setup": setup, "clips": clips, "n": len(g), "flagged": g["flag"].mean(),
                     **{item: g[item].mean() for item in CHECKLIST_ITEMS},
                     "jump_among_flagged": flagged["jump_or_teleport"].mean() if len(flagged) else np.nan})
    return pd.DataFrame(rows)


def checklist_markdown(chk: pd.DataFrame) -> str:
    if chk.empty:
        return ""
    lines = ["## Checklist answers", "",
             "Share of clips where each time question was answered yes. On clean clips, Flagged is the "
             "false-alarm rate; Jump among flagged is the share of flagged clips whose answers included a jump.", "",
             "| Setup | Clips | n | Flagged | Backward | Pause | Jump | Order | Jump among flagged |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in chk.iterrows():
        lines.append(f"| {SETUP_LABELS.get(r['setup'], r['setup'])} | {r['clips']} | {r['n']} | {_pct(r['flagged'])} | "
                     f"{_pct(r['backward_motion'])} | {_pct(r['unexplained_pause'])} | {_pct(r['jump_or_teleport'])} | "
                     f"{_pct(r['impossible_order'])} | {_pct(r['jump_among_flagged'])} |")
    return "\n".join(lines) + "\n"


def run_analysis(cfg: Config, calls: pd.DataFrame, cond: pd.DataFrame, total_spent: float) -> Path:
    out = cfg.path("results_dir")
    out.mkdir(parents=True, exist_ok=True)
    df = load_results(cfg, calls, cond)
    if df.empty:
        raise SystemExit("No successful answers match the current config yet. Run a plan first.")
    m = metrics(df)
    comp = comparisons(df)
    cons = consistency(df)
    ext = external_agreement(df)
    cost = costs(calls)
    reasons = stated_reasons(df)
    chk = checklist_answers(df)
    for name, table in [("metrics", m), ("comparisons", comp), ("consistency", cons),
                        ("external", ext), ("costs", cost), ("reasons", reasons), ("checklist", chk)]:
        if not table.empty:
            table.to_csv(out / f"{name}.csv", index=False)
    df.drop(columns=["raw_text"], errors="ignore").to_csv(out / "answers.csv", index=False)
    figure(m, out / "detection.png")
    (out / "summary.md").write_text(summary_markdown(cfg, df, m, comp, cons, ext, cost, reasons, total_spent)
                                    + "\n" + checklist_markdown(chk))
    return out