# Temporally Blind Judges?

**Stress-testing VLM judges of physical plausibility with time-only video corruptions.**

Video-generation benchmarks increasingly ask a vision-language model, often Gemini, whether a generated video is physically plausible. Video-language models are known to be weak at telling which way time runs. If the judge can't see time, its physics scores miss exactly the errors that matter.

This project takes real videos where physics is correct by definition (Google DeepMind's [Physics-IQ](https://github.com/google-deepmind/physics-IQ-benchmark)) and corrupts them in ways that break **only time**: every frame the judge sees is a real frame, only the order changes. Because each corruption is made on purpose, the correct verdict is known without human labels.

**Research questions**

1. **Sensitivity:** how often does the judge catch each corruption, at each strength, and how often does it wrongly flag clean clips?
2. **Why it fails:** does denser frame sampling or an explicit time checklist change the result?
3. **A fix:** does showing the clip next to its reversed copy close the gap, and at what cost?
4. **Does it matter:** do the blind spots predict the judge's errors on real generated videos labelled by people?

> Results: run the pipeline and see `results/summary.md`. No numbers in this README come from Gemini.

---

## Quick start

```bash
git clone <your repo url> temporally-blind-judges && cd temporally-blind-judges
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                      # 60 tests, about a minute
tbj demo --fresh            # the whole pipeline on synthetic clips with a fake judge: free, about 3 minutes
```

The demo writes `demo/results/summary.md` and `demo/results/detection.png`. Its numbers come from a fake judge and mean nothing; they only prove the pipeline works on your machine.

For real runs you need a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey):

```bash
export GEMINI_API_KEY=...   # Windows PowerShell: $env:GEMINI_API_KEY="..."
```

Set a budget alert in Google Cloud billing on day one. `config.yaml` also has a hard `budget_usd` (default $25): sync runs stop and batch submissions are refused past it.

ffmpeg is needed to encode clips. If it isn't on your PATH, the bundled `imageio-ffmpeg` binary is used.

---

## Workflow

Run every command from the repo folder.

| Step | Command | What happens | Your time |
|---|---|---|---|
| 1 | `tbj init` | Manifest of the 396 Physics-IQ videos, with a first-pass hint per scene | 1 min |
| 2 | `tbj download --set review` | One video per scene (66 files, 4K) | 15–40 min, unattended |
| 3 | `tbj review` | Writes `data/review/review.html`: each scene forward and reversed | 10 min to build |
| 4 | Open the page, label scenes, click **Download labels.csv** | You decide which scenes are irreversible | ~1.5 h |
| 5 | `tbj apply-labels ~/Downloads/labels.csv` | Labels every video of each scene | 1 min |
| 6 | `tbj download --set selected` | All 6 videos of each irreversible scene | unattended |
| 7 | `tbj build` | 13 conditions + reversed copies per clip, downscaled to 768 px | ~1 h, unattended |
| 8 | `tbj check` | `data/conditions/check.html`: clean vs corrupted side by side | 30 min to look |
| 8b | `tbj spots` | Screen for clips whose motion detection caught a side event (e.g. a reflection); lists clips to look at | ~5 min |
| 8c | `tbj check --visibility` | 32 freezes, drops and partial reversals to label visible / subtle / invisible in the page; download the CSV to `labels/visibility.csv` | ~45 min to label |
| 9 | `tbj plan pilot` then `tbj run pilot` | 20 clips × 4 conditions × 4 setups, about 320 calls | ~1 h |
| 10 | `tbj batch submit main` and `tbj batch submit setups` | The full run at half price | 5 min, results within 24 h |
| 11 | `tbj batch collect` | Stores the answers (run `tbj batch status` to check first) | 1 min |
| 12 | `tbj run consistency` | The same questions again on 30 clips, to measure stability | ~30 min |
| 13 | `tbj analyze` | `results/summary.md`, `results/detection.png`, CSV tables | 1 min |

`tbj status` shows progress and spend at any time. Every command is safe to re-run: finished work is skipped.

**Labelling scenes (step 4).** Label a scene **irreversible** only if its reversed clip looks clearly impossible: spills un-spilling, dropped objects rising to where they were released, pieces reassembling. A rotating teapot or a ball rolling on a flat table looks fine backward, so it can't serve as ground truth. The hints from `tbj init` (30 scenes "likely irreversible") are guesses from the text descriptions; your eyes decide. Expect about 25 scenes, so about 150 clips.

**Checking corruptions (step 8).** Each corrupted clip should look wrong next to its clean version. If a freeze or drop lands where nothing moves, motion detection missed something: adjust `motion_threshold` in `config.yaml` and re-run `tbj build`.

---

## Cost

`tbj plan <name>` prints an estimate before anything runs. With `gemini-3.8-flash` ($0.75 in / $3.75 out per million tokens through Dec 31, 2026; half price in batch) and about 150 clips:

| Plan | Calls | Estimate |
|---|---|---|
| pilot (sync) | ~320 | ~$1 |
| main (batch) | ~1,950 | ~$2.5 |
| setups (batch) | ~4,050 | ~$6 |
| consistency (sync) | ~390 | ~$1 |
| external validation (batch, ~300 videos) | ~1,200 | ~$2 |

Total about $12–30. The range is the thinking tokens, which are billed as output and only known after the pilot. The `cost` table in `results/summary.md` reports actual spend per setup.

---

## How it works

### Corruptions

All corruptions are frame-index maps (`tbj/corrupt.py`): output frame *t* shows input frame `index[t]`. Every map keeps the clip length, so duration is never a cue.

| Corruption | Mild | Medium | Strong | What the judge sees |
|---|---|---|---|---|
| Reversal | middle 25% | middle 50% | whole clip | Part of the action plays backward |
| Shuffle | 2 segments | 4 segments | 8 segments | Scenes jump between moments; no segment keeps its original successor |
| Freeze | 0.25 s | 0.5 s | 1 s | Everything pauses mid-motion, then continues where it stopped |
| Drop | 0.1 s | 0.25 s | 0.5 s | Moving objects jump ahead |

Partial reversals, freezes and drops are placed inside the **motion span** (`tbj/motion.py`): the frames where the share of changed pixels is above 20% of its peak. Placement is random but reproducible (seeded by clip and condition). `conditions.csv` records where each corruption landed and how much of it overlaps the motion.

A useful property: the clean clip played backward is, frame for frame, the strong reversal. So in the paired setup, the clean and strong-reversal conditions show the judge the same two videos. A consistent judge picks the same one both times.

### Judging setups

| Setup | Kind | Prompt | Frames/s | Question |
|---|---|---|---|---|
| `default` | single | `default_v1` | 1 | Overall plausibility rubric; time is never mentioned |
| `dense` | single | `default_v1` | 4 | Same, with 4× the frames |
| `checklist` | checklist | `checklist_v1` | 1 | Explicit yes/no questions: backward motion, pauses, jumps, impossible order |
| `paired` | paired | `paired_v1` | 1 | The clip and its reversed copy, in random order: which one runs forward? |

Each setup changes one thing relative to `default`, so differences can be attributed. Answers are JSON constrained by a schema (`tbj/judge.py`), with the one-line reason first. All calls use static frame extraction (frames at exactly the setup's rate, all placed in context, rather than Gemini's opt-in agentic mode that chooses what to watch), low media resolution (about 70 tokens per frame on Gemini 3), low thinking, the model's default temperature and no audio.

**Flag** (clip judged implausible): verdict "implausible" for single and checklist setups; for paired, preferring the reversed copy or rejecting both. **Score** (for AUROC): (5 − plausibility score) / 4, or a confidence-weighted equivalent for paired. The checklist also gets a looser variant, `checklist_any`: flagged if any question is answered yes.

### Caching, batches and uploads

- Every call is stored in `runs/results.sqlite` under a key built from the clip bytes, the prompt text, the frame rate, the model settings and the repeat index. Re-running never pays twice; editing a prompt re-runs only the setups that use it.
- To change a prompt, add a new file (for example `tbj/prompts/default_v2.txt`) and point the setup at it. Old answers stay in the store, and `tbj analyze` only uses answers that match the current config.
- Clips are uploaded once per content hash through the Gemini Files API (`runs/uploads.json`). Uploads expire after about 48 hours; batch submissions re-upload anything with under 26 hours left.
- Sync runs retry rate limits and server errors with exponential backoff. Batch jobs that fail or expire return their requests to the queue.

### Statistics

`tbj analyze` reports, per setup and condition: detection rate, false-alarm rate on clean clips, balanced accuracy and AUROC, each with a 95% confidence interval from a **cluster bootstrap over scenes**. Clips of one scene are correlated, so whole scenes are resampled. It also reports each setup minus `default` on the same clips, agreement and Cohen's kappa between repeated runs, how often a flagged clip's stated reason is a time problem, and cost per setup.

Two planned analyses are written into the code before any judge call:
- **H5:** the gain from dense sampling on clips whose detected motion lasts under 1 s vs 1 s or more (medium and strong corruptions), and the difference between the two gains.
- **H6:** detection of the ¼ and ½ reversals against the full reversal, on the same clips, at default and dense settings. Partial reversals jump at both edges; the full reversal has no jumps, only direction.
- **Visibility (control):** a corruption on a near-still moment can be physically impossible yet invisible, and counts as a miss. `tbj check --visibility` shows a random sample of 32 freezes, drops and partial reversals to label by eye; `tbj analyze` reports the share labelled invisible per corruption with a 95% interval. Invisible corruptions can lower a detection rate by at most their share, so a gap larger than that bound cannot be explained by them. (An automatic pixel-based measure was tried first and dropped: it failed checks against the eye in both directions; `tbj spots` keeps it only as a screen for clips whose motion detection caught a side event.)

### Validation on generated videos (question 4)

Put labelled videos (for example from ImplausiBench or FinePhyEval, if their licences allow) in a CSV with columns `path,label[,id]`, where `label` is `plausible` or `implausible`. Then:

```bash
tbj add-external validation.csv
tbj batch submit external        # later: tbj batch collect && tbj analyze
```

The summary gains a table of accuracy and Cohen's kappa against the human labels for each setup.

---

## Repo layout

```
config.yaml          every setting: paths, corruption strengths, model, setups, plans, prices, budget
tbj/
  physiq.py          manifest, downloads, scene review page
  motion.py          motion span detection
  corrupt.py         the corruptions, as frame-index maps
  build.py           renders conditions; check page; outside videos
  judge.py           prompts, JSON schemas, cache keys, request building, parsing, cost
  prompts/           versioned prompt files
  uploads.py         Gemini Files API uploads with expiry tracking
  store.py           SQLite store of every call and batch job
  plan.py            plans -> jobs
  runner.py          sync runner and Batch API runner
  spots.py           screen for clips whose motion detection caught a side event
  analyze.py         metrics, bootstrap CIs, figure, summary
  fake.py            fake Gemini client for the demo and tests
  synthetic.py       synthetic clips for the demo and tests
tests/               unit tests and an end-to-end test
```

Generated files (`data/`, `runs/`, `demo/`) are git-ignored. Commit `results/summary.md`, the CSVs and `results/detection.png` when you publish.

## Notes

- Pin one model version for the whole project and report it. Changing `judge.model` starts a fresh set of answers.
- Physics-IQ videos are not redistributed here; `tbj download` fetches them from the benchmark's public bucket. If an HTTPS download is refused, the error message prints the equivalent `gcloud storage cp` command. Cite Physics-IQ as its repository asks.
- `tbj/resources/physiq_descriptions.csv` is copied from the Physics-IQ repository (Apache 2.0); see `tbj/resources/NOTICE.md`.
- Add a licence file before publishing (Apache 2.0 matches Physics-IQ).
