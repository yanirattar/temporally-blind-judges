# Results

Judge `gemini-3.8-flash` · 10990 answers used · 212 clips from 37 scenes · total API spend $14.21

## Flag rate by condition

Share of clips judged implausible, with 95% CI. For corrupted clips higher is better; for clean clips (false alarms) lower is better.

| Condition | Default | Dense sampling | Told about sampling | Time checklist | Checklist (any item) | Checklist, 4 frames/s | Paired with reverse |
|---|---|---|---|---|---|---|---|
| Clean (false alarm) | 52% [42%–62%] | 35% [25%–46%] | 30% [20%–40%] | 41% [31%–52%] | 41% [31%–52%] | 20% [11%–31%] | 7% [2%–12%] |
| Reversal, mild | 80% [72%–87%] | – | – | – | – | – | – |
| Reversal, medium | 89% [81%–95%] | 97% [94%–99%] | 94% [90%–98%] | 94% [89%–98%] | 94% [90%–98%] | – | 43% [35%–52%] |
| Reversal, strong | 94% [87%–100%] | 93% [87%–98%] | 91% [83%–97%] | 94% [87%–99%] | 94% [87%–99%] | – | 93% [87%–98%] |
| Shuffle, mild | 94% [89%–98%] | – | – | – | – | – | – |
| Shuffle, medium | 95% [92%–98%] | 91% [86%–95%] | 97% [94%–100%] | 96% [92%–99%] | 96% [92%–99%] | – | 55% [48%–63%] |
| Shuffle, strong | 95% [91%–98%] | 96% [92%–99%] | 96% [92%–99%] | 94% [90%–98%] | 95% [91%–98%] | – | 57% [50%–63%] |
| Freeze, mild | 46% [35%–57%] | – | – | – | – | – | – |
| Freeze, medium | 42% [31%–54%] | 41% [29%–54%] | 29% [20%–38%] | 29% [21%–39%] | 29% [21%–39%] | – | 8% [3%–15%] |
| Freeze, strong | 47% [38%–56%] | 52% [42%–63%] | 27% [18%–36%] | 33% [24%–42%] | 33% [24%–42%] | – | 4% [1%–8%] |
| Drop, mild | 49% [38%–60%] | – | – | – | – | – | – |
| Drop, medium | 54% [43%–64%] | 46% [36%–56%] | 31% [21%–41%] | 44% [34%–55%] | 44% [34%–55%] | – | 7% [3%–13%] |
| Drop, strong | 58% [47%–68%] | 46% [35%–58%] | 32% [22%–42%] | 41% [29%–53%] | 41% [29%–53%] | – | 8% [2%–16%] |
| All corrupted | 70% [65%–75%] | 70% [65%–76%] | 62% [58%–67%] | 66% [61%–70%] | 66% [61%–70%] | – | 35% [32%–37%] |

| Overall | Default | Dense sampling | Told about sampling | Time checklist | Checklist (any item) | Checklist, 4 frames/s | Paired with reverse |
|---|---|---|---|---|---|---|---|
| Balanced accuracy | 0.59 [0.56–0.62] | 0.67 [0.64–0.71] | 0.66 [0.63–0.70] | 0.62 [0.58–0.66] | 0.62 [0.58–0.66] | – | 0.64 [0.62–0.66] |
| AUROC | 0.60 [0.56–0.63] | 0.68 [0.64–0.72] | 0.66 [0.63–0.70] | 0.63 [0.59–0.67] | 0.63 [0.59–0.67] | – | 0.64 [0.63–0.66] |

## Each setup vs default

Difference in flag rate on the same clips (percentage points, 95% CI).

| Setup | Measure | Default | Setup | Difference |
|---|---|---|---|---|
| Dense sampling | false alarm (clean) | 52% | 35% | -17 pts [-27, -7] |
| Dense sampling | detection (all corrupted) | 72% | 70% | -1 pts [-6, +3] |
| Dense sampling | detection (reversal) | 92% | 95% | +3 pts [-0, +8] |
| Dense sampling | detection (shuffle) | 95% | 93% | -2 pts [-5, +2] |
| Dense sampling | detection (freeze) | 44% | 47% | +2 pts [-6, +10] |
| Dense sampling | detection (drop) | 56% | 46% | -9 pts [-17, -3] |
| Told about sampling | false alarm (clean) | 52% | 30% | -22 pts [-30, -15] |
| Told about sampling | detection (all corrupted) | 72% | 62% | -10 pts [-13, -7] |
| Told about sampling | detection (reversal) | 92% | 93% | +1 pts [-3, +5] |
| Told about sampling | detection (shuffle) | 95% | 96% | +1 pts [-1, +4] |
| Told about sampling | detection (freeze) | 44% | 28% | -17 pts [-23, -10] |
| Told about sampling | detection (drop) | 56% | 31% | -25 pts [-33, -16] |
| Time checklist | false alarm (clean) | 52% | 41% | -11 pts [-20, -2] |
| Time checklist | detection (all corrupted) | 72% | 66% | -6 pts [-10, -2] |
| Time checklist | detection (reversal) | 92% | 94% | +2 pts [-1, +6] |
| Time checklist | detection (shuffle) | 95% | 95% | +0 pts [-3, +3] |
| Time checklist | detection (freeze) | 44% | 31% | -13 pts [-20, -7] |
| Time checklist | detection (drop) | 56% | 43% | -13 pts [-21, -5] |
| Checklist (any item) | false alarm (clean) | 52% | 41% | -11 pts [-20, -2] |
| Checklist (any item) | detection (all corrupted) | 72% | 66% | -6 pts [-10, -2] |
| Checklist (any item) | detection (reversal) | 92% | 94% | +3 pts [-1, +6] |
| Checklist (any item) | detection (shuffle) | 95% | 96% | +0 pts [-3, +3] |
| Checklist (any item) | detection (freeze) | 44% | 31% | -13 pts [-20, -7] |
| Checklist (any item) | detection (drop) | 56% | 43% | -13 pts [-21, -5] |
| Checklist, 4 frames/s | false alarm (clean) | 52% | 20% | -32 pts [-44, -19] |
| Paired with reverse | false alarm (clean) | 52% | 7% | -45 pts [-56, -35] |
| Paired with reverse | detection (all corrupted) | 72% | 35% | -37 pts [-42, -32] |
| Paired with reverse | detection (reversal) | 92% | 68% | -23 pts [-29, -17] |
| Paired with reverse | detection (shuffle) | 95% | 56% | -39 pts [-46, -33] |
| Paired with reverse | detection (freeze) | 44% | 6% | -38 pts [-47, -29] |
| Paired with reverse | detection (drop) | 56% | 8% | -48 pts [-58, -38] |

## Consistency (same question asked twice)

| Setup | Clips | Agreement | Cohen's kappa |
|---|---|---|---|
| Default | 390 | 91% | 0.79 |

## Does the judge name a time problem when it flags a clip?

| Setup | Corruption | Share naming a time problem |
|---|---|---|
| Default | drop | 25% |
| Default | freeze | 16% |
| Default | reversal | 59% |
| Default | shuffle | 67% |
| Dense sampling | drop | 27% |
| Dense sampling | freeze | 16% |
| Dense sampling | reversal | 52% |
| Dense sampling | shuffle | 74% |
| Told about sampling | drop | 8% |
| Told about sampling | freeze | 3% |
| Told about sampling | reversal | 53% |
| Told about sampling | shuffle | 66% |

## Cost

| Setup | Mode | Calls | Avg input tokens | Avg output + thinking | Cost |
|---|---|---|---|---|---|
| Time checklist | batch | 1860 | 821 | 329 | $1.72 |
| Time checklist | sync | 80 | 821 | 321 | $0.15 |
| Checklist, 4 frames/s | batch | 212 | 2405 | 304 | $0.31 |
| Default | batch | 3112 | 730 | 299 | $2.60 |
| Default | sync | 80 | 730 | 274 | $0.13 |
| Dense sampling | batch | 1860 | 2314 | 344 | $2.81 |
| Dense sampling | sync | 80 | 2314 | 326 | $0.24 |
| Paired with reverse | batch | 1860 | 1189 | 933 | $4.08 |
| Paired with reverse | sync | 80 | 1189 | 651 | $0.27 |
| Told about sampling | batch | 1860 | 792 | 349 | $1.77 |
| Told about sampling | sync | 80 | 792 | 314 | $0.14 |

## Checklist answers

Share of clips where each time question was answered yes. On clean clips, Flagged is the false-alarm rate; Jump among flagged is the share of flagged clips whose answers included a jump.

| Setup | Clips | n | Flagged | Backward | Pause | Jump | Order | Jump among flagged |
|---|---|---|---|---|---|---|---|---|
| Time checklist | clean | 212 | 41% | 1% | 3% | 31% | 5% | 76% |
| Time checklist | corrupted | 1696 | 66% | 13% | 3% | 48% | 18% | 73% |
| Checklist, 4 frames/s | clean | 212 | 20% | 0% | 3% | 14% | 2% | 70% |

## H5: does dense sampling help more when the action is short?

Clips split by how long the detected motion lasts in the clean clip: under 1 s (68 clips) vs 1 s or more (144 clips). Medium and strong corruptions run in both setups. Default → dense detection, and the gain in percentage points (95% CI). H5 predicts a positive difference.

| Corruptions | Short action (n) | Short: default → dense | Long action (n) | Long: default → dense | Difference in gain (short − long) |
|---|---|---|---|---|---|
| All | 544 | 75% → 76%: +1 pts [-6, +8] | 1152 | 70% → 68%: -3 pts [-7, +2] | +4 pts [-4, +12] |
| Reversal | 136 | 91% → 100%: +9 pts [+1, +19] | 288 | 92% → 92%: +1 pts [-3, +5] | +8 pts [-1, +19] |
| Shuffle | 136 | 95% → 96%: +1 pts [-5, +6] | 288 | 95% → 92%: -3 pts [-7, +2] | +4 pts [-4, +11] |
| Freeze | 136 | 43% → 45%: +2 pts [-13, +16] | 288 | 45% → 48%: +2 pts [-7, +11] | -0 pts [-18, +16] |
| Drop | 136 | 70% → 64%: -6 pts [-18, +5] | 288 | 49% → 38%: -11 pts [-21, -2] | +5 pts [-11, +20] |

## H6: are partial reversals caught more often than the full reversal?

A partial reversal (¼ or ½ of the clip) plays a stretch backward and jumps at both edges; the full reversal has no jumps, only direction. A judge that understood direction would catch the full reversal most; H6 predicts the partial ones are caught more (positive differences). Same clips in each comparison.

| Setup | ¼ reversal vs full | ½ reversal vs full |
|---|---|---|
| Default | 80% vs 94%: -15 pts [-24, -6] (n=212) | 89% vs 94%: -6 pts [-14, +2] (n=212) |
| Dense sampling | – | 97% vs 93%: +4 pts [-1, +10] (n=212) |

## Exploratory: telling the judge it sees one frame per second

Added after the pilot, where most false alarms on clean clips described objects teleporting during fast events that fell between two sampled frames. The sampling-aware prompt is the default prompt plus one paragraph saying the video is sampled once per second. Both at 1 frame/s, same clips (clean, medium and strong corruptions); differences in percentage points with 95% CI.

| Measure | Default | Told about sampling | Difference |
|---|---|---|---|
| False alarms (clean) | 52% | 30% | -22 pts [-30, -14] |
| Detection (medium + strong) | 72% | 62% | -10 pts [-13, -7] |
| Detection: reversal | 92% | 93% | +1 pts [-3, +5] |
| Detection: shuffle | 95% | 96% | +1 pts [-1, +4] |
| Detection: freeze | 44% | 28% | -17 pts [-23, -11] |
| Detection: drop | 56% | 31% | -25 pts [-33, -16] |
| Balanced accuracy | 0.60 | 0.66 | +0.06 [+0.03, +0.10] |

## Why does the judge flag a clip? (stated reasons, exploratory)

Flagged answers in single-clip setups, sorted by keywords in the stated reason (lists fixed on 2026-11-18, before the main runs; tags can overlap, so rows need not add to 100%). For reversals the right reason is backward motion, for shuffles and drops a jump, for freezes a pause. On clean clips every flag is a false alarm, and the tag shows what the judge thought it saw. Per-strength rows are in reason_categories.csv.

| Setup | Clips | Flagged | Backward motion | Jump or disappearance | Pause or freeze | None of these |
|---|---|---|---|---|---|---|
| Default | Clean | 110 | 2% | 51% | 12% | 40% |
| Default | Reversal | 557 | 30% | 60% | 5% | 15% |
| Default | Shuffle | 603 | 15% | 80% | 3% | 11% |
| Default | Freeze | 285 | 4% | 45% | 10% | 44% |
| Default | Drop | 340 | 2% | 54% | 10% | 38% |
| Dense sampling | Clean | 75 | 4% | 35% | 17% | 49% |
| Dense sampling | Reversal | 402 | 53% | 40% | 9% | 15% |
| Dense sampling | Shuffle | 396 | 8% | 87% | 7% | 9% |
| Dense sampling | Freeze | 198 | 6% | 35% | 25% | 42% |
| Dense sampling | Drop | 196 | 3% | 42% | 15% | 42% |
| Told about sampling | Clean | 63 | 0% | 38% | 24% | 48% |
| Told about sampling | Reversal | 393 | 39% | 61% | 5% | 11% |
| Told about sampling | Shuffle | 409 | 14% | 81% | 5% | 12% |
| Told about sampling | Freeze | 118 | 0% | 31% | 16% | 57% |
| Told about sampling | Drop | 132 | 3% | 36% | 21% | 49% |
| Time checklist | Clean | 87 | 6% | 72% | 9% | 21% |
| Time checklist | Reversal | 398 | 40% | 57% | 5% | 7% |
| Time checklist | Shuffle | 404 | 19% | 85% | 4% | 4% |
| Time checklist | Freeze | 131 | 5% | 66% | 11% | 22% |
| Time checklist | Drop | 181 | 3% | 77% | 8% | 20% |
| Checklist, 4 frames/s | Clean | 43 | 7% | 63% | 14% | 23% |

## How many corruptions can a person see? (hand-labelled sample)

A random sample of freezes, drops and partial reversals, each labelled by eye (`tbj check --visibility`): visible at normal speed, subtle (only at the stated time) or invisible (not even frame by frame). Invisible corruptions can lower a detection rate by at most their share, so a gap between corruptions larger than the upper end of the interval cannot be explained by them.

| Corruption | Labelled | Visible | Subtle | Invisible (95% CI) |
|---|---|---|---|---|
| Freeze | 12 | 92% | 8% | 0% [0%–24%] |
| Drop | 12 | 75% | 0% | 25% [9%–53%] |
| Partial reversal | 8 | 100% | 0% | 0% [0%–32%] |
