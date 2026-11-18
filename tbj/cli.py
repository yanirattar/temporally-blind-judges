"""Command line: `tbj <command>` (or `python -m tbj <command>`). Run from the repo folder.

Workflow
  tbj init                         manifest of the 396 Physics-IQ videos
  tbj download --set review        one video per scene (66 files)
  tbj review                       open data/review/review.html, label scenes, download labels.csv
  tbj apply-labels labels.csv
  tbj download --set selected      all videos of the scenes you labelled irreversible
  tbj build                        render the 13 conditions per clip
  tbj check                        open data/conditions/check.html and look at the corruptions
  tbj check --visibility           label a sample of corruptions visible / subtle / invisible
  tbj spots                        screen for clips whose motion detection caught the wrong thing
  tbj plan pilot                   how many calls a plan needs and what it should cost
  tbj run pilot                    run a plan now (sync)
  tbj batch submit main            run a plan through the Batch API (half price)
  tbj batch status | collect
  tbj analyze                      results/summary.md, results/detection.png and CSVs
  tbj demo                         the whole pipeline on synthetic clips with a fake judge (free)
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import yaml

from . import physiq
from .config import load_config


def _confirm(prompt: str, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{prompt} [skipped: pass --yes to confirm when not running interactively]")
        return False
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def _runner(cfg, fake: bool):
    from .build import read_conditions
    from .runner import Runner, make_client
    from .store import Store
    return Runner(cfg, make_client(cfg, fake=fake), Store(cfg.path("store")), read_conditions(cfg))


def _conditions(cfg):
    from .build import read_conditions
    return read_conditions(cfg)


# ------------------------------------------------------------------ commands

def cmd_init(cfg, args):
    path = physiq.init_manifest(cfg, force=args.force)
    rows = physiq.read_manifest(cfg)
    hints = {}
    for r in physiq.review_rows(rows):
        hints[r["suggested"]] = hints.get(r["suggested"], 0) + 1
    print(f"Wrote {path} ({len(rows)} videos, {len(physiq.scenes(rows))} scenes).")
    print(f"Suggestions from the descriptions (confirm by watching): {hints}")


def cmd_download(cfg, args):
    rows = physiq.read_manifest(cfg)
    chosen = physiq.review_rows(rows) if args.set == "review" else physiq.selected_rows(rows)
    if args.set == "selected" and not chosen:
        raise SystemExit("No scenes labelled 'irreversible' yet. Run `tbj review` and `tbj apply-labels` first.")
    print(f"Downloading {len(chosen)} videos ({cfg['physiq']['fps']} FPS) to {cfg.path('raw_dir')}")
    new, skipped = physiq.download(cfg, chosen, workers=args.workers)
    print(f"Done: {new} downloaded, {skipped} already present.")


def cmd_review(cfg, args):
    rows = physiq.read_manifest(cfg)
    items = physiq.build_review_previews(cfg, physiq.review_rows(rows))
    path = physiq.write_review_page(cfg, items)
    print(f"Open {path} in a browser, label each scene, then click 'Download labels.csv' and run "
          f"`tbj apply-labels <path to labels.csv>`.")


def cmd_apply_labels(cfg, args):
    counts = physiq.apply_labels(cfg, args.csv)
    s = physiq.manifest_summary(physiq.read_manifest(cfg))
    print(f"Applied {sum(counts.values())} scene labels {counts}. "
          f"{s['labelled']}/{s['scenes']} scenes labelled; {s['selected_videos']} videos selected.")


def cmd_build(cfg, args):
    from .build import build_all
    path = build_all(cfg, args.config, workers=args.workers, limit=args.limit)
    df = _conditions(cfg)
    print(f"Wrote {path}: {len(df)} conditions for {df['video_id'].nunique()} clips.")


def cmd_spots(cfg, args):
    from .spots import describe, spot_table
    cond = _conditions(cfg)
    print(describe(spot_table(cfg, cond), cond))


def cmd_check(cfg, args):
    from .build import write_check_page, write_visibility_page
    if args.visibility:
        page = write_visibility_page(cfg)
        print(f"Open {page}, label each pair in the page, then press 'Download visibility.csv' and move it:\n"
              f"  mv ~/Downloads/visibility.csv {cfg.root / 'labels' / 'visibility.csv'}")
        return
    print(f"Open {write_check_page(cfg)} in a browser and confirm each corruption looks wrong.")


def cmd_add_external(cfg, args):
    from .build import add_external
    print(f"Added {add_external(cfg, args.csv)} outside videos. Run them with `tbj run external` "
          f"or `tbj batch submit external`.")


def cmd_plan(cfg, args):
    from .runner import Runner
    from .store import Store
    from .plan import expand
    runner = Runner(cfg, None, Store(cfg.path("store")), _conditions(cfg))
    items = runner.prepare(expand(cfg, args.plan, _conditions(cfg)))
    pending = runner.pending(items)
    by_setup: dict[str, int] = {}
    for it in pending:
        by_setup[it.job.setup] = by_setup.get(it.job.setup, 0) + 1
    _, sync_cost = runner.estimate(pending, batch=False)
    _, batch_cost = runner.estimate(pending, batch=True)
    print(f"Plan '{args.plan}': {len(items)} calls, {len(pending)} still to run {by_setup}.")
    print(f"Estimated cost: ${sync_cost:.2f} sync, ${batch_cost:.2f} batch "
          f"(rough: assumes {cfg['judge']['tokens_per_frame']} tokens per frame and "
          f"{cfg['judge']['output_tokens_guess']} output+thinking tokens per call).")
    print(f"Spent so far: ${runner.store.total_cost():.2f} of the ${cfg['budget_usd']:.0f} budget.")


def cmd_run(cfg, args):
    from .plan import expand
    runner = _runner(cfg, args.fake)
    items = runner.pending(runner.prepare(expand(cfg, args.plan, _conditions(cfg))))
    if args.limit:
        items = items[:args.limit]
    if not items:
        print("Nothing to run: every call in this plan already has an answer.")
        return
    _, est = runner.estimate(items, batch=False)
    budget = args.budget or cfg["budget_usd"]
    print(f"{len(items)} calls to run now, estimated ${est:.2f}; spent so far ${runner.store.total_cost():.2f} "
          f"of the ${budget:.0f} budget.")
    if not args.fake and not _confirm("Run now?", args.yes):
        return
    s = cfg["sync"]
    counts = runner.run_sync(items, budget=budget, concurrency=s["concurrency"], max_retries=s["max_retries"])
    print(f"Finished: {counts['ok']} ok, {counts['failed']} failed"
          + (" (stopped at the budget)" if counts["stopped"] else "") + f". Total spent ${runner.store.total_cost():.2f}.")
    if counts["failed"]:
        print("Failed calls are retried the next time you run the plan.")


def cmd_batch(cfg, args):
    runner = _runner(cfg, args.fake)
    if args.action == "submit":
        if not args.plan:
            raise SystemExit("Usage: tbj batch submit <plan>")
        from .plan import expand
        items = runner.pending(runner.prepare(expand(cfg, args.plan, _conditions(cfg))))
        if args.limit:
            items = items[:args.limit]
        if not items:
            print("Nothing to submit: every call in this plan is answered or already in a batch.")
            return
        _, est = runner.estimate(items, batch=True)
        print(f"{len(items)} calls to submit, estimated ${est:.2f} at batch prices; "
              f"spent so far ${runner.store.total_cost():.2f}.")
        if runner.store.total_cost() + est > cfg["budget_usd"]:
            raise SystemExit(f"This would exceed the ${cfg['budget_usd']} budget in config.yaml. Raise it or use --limit.")
        if not args.fake and not _confirm("Submit?", args.yes):
            return
        b = cfg["batch"]
        names = runner.submit_batches(items, args.plan, b["chunk_size"], b["min_upload_life_hours"])
        print(f"Submitted {len(names)} batch jobs. Check with `tbj batch status`; results usually arrive within 24 h.")
    elif args.action == "status":
        rows = runner.batch_status()
        if not rows:
            print("No open batch jobs.")
        for r in rows:
            print(f"  {r['display_name']}: {r['state']} ({r['n_requests']} requests)")
    else:
        c = runner.collect()
        print(f"Collected {c['batches_done']} finished jobs: {c['ok']} answers, {c['failed']} failed. "
              f"{c['batches_running']} still running. Total spent ${runner.store.total_cost():.2f}.")


def cmd_analyze(cfg, args):
    from .analyze import run_analysis
    from .store import Store
    store = Store(cfg.path("store"))
    out = run_analysis(cfg, store.calls(), _conditions(cfg), store.total_cost())
    print(f"Wrote {out / 'summary.md'}, {out / 'detection.png'} and CSV tables to {out}.")


def cmd_status(cfg, args):
    s = physiq.manifest_summary(physiq.read_manifest(cfg))
    print(f"Scenes labelled: {s['labelled']}/{s['scenes']} {s['by_label']}; videos selected: {s['selected_videos']}")
    if cfg.path("conditions_csv").exists():
        df = _conditions(cfg)
        print(f"Conditions built: {len(df)} for {df['video_id'].nunique()} clips")
    if cfg.path("store").exists():
        from .store import Store
        store = Store(cfg.path("store"))
        calls = store.calls()
        if not calls.empty:
            print("Answers stored:")
            print(calls.groupby(["setup", "repeat", "status"]).size().to_string())
        print(f"Spent: ${store.total_cost():.2f} of ${cfg['budget_usd']:.0f}; open batch jobs: {len(store.open_batches())}")


def cmd_demo(cfg, args):
    """The full pipeline on synthetic clips with the fake judge. Costs nothing."""
    from . import synthetic
    from .config import Config

    root = Path(args.dir).resolve()
    if root.exists() and args.fresh:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    data = copy.deepcopy(dict(cfg))
    data["preprocess"]["width"] = synthetic.W
    data["plans"]["pilot"]["clips"] = 8
    data["plans"]["consistency"]["clips"] = 8
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))
    demo = Config(data, root)
    print(f"Demo project in {root}")

    print("1/7 Writing synthetic clips and a labelled manifest")
    physiq.write_manifest(demo, synthetic.write_dataset(demo.path("raw_dir")))
    print("2/7 Building conditions")
    from .build import build_all, write_check_page
    build_all(demo, str(config_path), workers=1)
    write_check_page(demo)

    def run(argv):
        print(f"$ tbj {' '.join(argv)}")
        main(["--config", str(config_path), *argv])

    print("3/7 Pilot (sync, fake judge with simulated 503 errors)")
    run(["run", "pilot", "--fake"])
    print("4/7 Main run and setup comparison through the (fake) Batch API")
    run(["batch", "submit", "main", "--fake"])
    run(["batch", "submit", "setups", "--fake"])
    run(["batch", "collect", "--fake"])
    print("5/7 Consistency rerun")
    run(["run", "consistency", "--fake"])
    print("6/7 Status")
    run(["status"])
    print("7/7 Analysis")
    run(["analyze"])
    print(f"\nDemo done. Open {root / 'results' / 'summary.md'} and {root / 'results' / 'detection.png'}.\n"
          "The numbers come from the fake judge and mean nothing; they only show the pipeline works.")


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tbj", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml", help="path to config.yaml (default: ./config.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="create the clip manifest")
    s.add_argument("--force", action="store_true", help="rebuild it, keeping existing labels")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("download", help="download Physics-IQ videos")
    s.add_argument("--set", choices=["review", "selected"], required=True)
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(func=cmd_download)

    sub.add_parser("review", help="build the scene review page").set_defaults(func=cmd_review)

    s = sub.add_parser("apply-labels", help="apply a scenario,label CSV from the review page")
    s.add_argument("csv")
    s.set_defaults(func=cmd_apply_labels)

    s = sub.add_parser("build", help="render all conditions for the selected clips")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--limit", type=int, help="only the first N clips (for a quick test)")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("check", help="build the corruption check page")
    s.add_argument("--visibility", action="store_true",
                   help="a 32-clip sample of freezes, drops and partial reversals to label by eye")
    s.set_defaults(func=cmd_check)
    sub.add_parser("spots", help="screen for clips whose motion detection caught the wrong thing").set_defaults(func=cmd_spots)

    s = sub.add_parser("add-external", help="register outside videos with human labels")
    s.add_argument("csv", help="CSV with columns path,label[,id]")
    s.set_defaults(func=cmd_add_external)

    s = sub.add_parser("plan", help="count calls and estimate the cost of a plan")
    s.add_argument("plan")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("run", help="run a plan now (sync)")
    s.add_argument("plan")
    s.add_argument("--limit", type=int)
    s.add_argument("--budget", type=float, help="override budget_usd for this run")
    s.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    s.add_argument("--fake", action="store_true", help="use the fake judge (free)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("batch", help="Batch API: submit, status, collect")
    s.add_argument("action", choices=["submit", "status", "collect"])
    s.add_argument("plan", nargs="?")
    s.add_argument("--limit", type=int)
    s.add_argument("--yes", action="store_true")
    s.add_argument("--fake", action="store_true")
    s.set_defaults(func=cmd_batch)

    sub.add_parser("analyze", help="metrics, figure and summary").set_defaults(func=cmd_analyze)
    sub.add_parser("status", help="progress and spend so far").set_defaults(func=cmd_status)

    s = sub.add_parser("demo", help="run everything on synthetic clips with a fake judge")
    s.add_argument("--dir", default="demo")
    s.add_argument("--fresh", action="store_true", help="delete the demo folder first")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
