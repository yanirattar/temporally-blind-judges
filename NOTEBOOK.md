# Lab notebook

Dated notes on decisions, observations and runs. Newest entries at the bottom of each section.

## Scene labelling rule

**Irreversible** = a fresh viewer, at normal speed, without knowing the original, would call
the reversed clip impossible: things un-breaking or un-spilling, clearly visible motion growing
by itself, objects starting to move from rest with nothing touching them.
Hidden causes count only if a viewer would naturally assume them (an off-screen push, a visible
string or device). Magnets, invisible wires and other exotic forces don't count.

**Subtle** (labelled reversible) = wrong only through fine details (a single hop, a missing
bounce, a too-perfect landing) or noticeable only by knowing the original.

When unsure: reversible.

## Scene labels

Pass 1: 2026-11-10. Pass 2: 2026-11-10.

## My scene predictions (written before any results)

Irreversible scenes where I expect Gemini to **miss** the full reversal:
1. ball-in-sand 
2. single-cradle
3. match
4. paper-smoke

Irreversible scenes where I expect Gemini to **catch** it:
1. ball-in-basket
2. block-domino
3. paint-on-glass
4. juice-in-water
5. cut-paper

## Decisions

- 2026-11-10: Hypotheses committed in HYPOTHESES.md. Added checklist at 4 frames/s on clean clips to test H3.
- 2026-11-10: Static frame processing, low resolution, low thinking, model gemini-3.8-flash.
- 2026-11-17: Excluded 0103, then marble-run-x 0104, 0105, 0301, 0302, by one rule: exclude a clip if its
  corruptions miss the event that makes the scene irreversible. 217 clips, 2,821 conditions.
  (See Corruption check and Screen for broken clips.)
- 2026-11-18: Visibility is controlled by a hand-labelled sample and a bound, replacing the pixel-based
  control planned on 11-17 (see Visibility sample).
- 2026-11-18: Added H6 (partial reversals caught more than the full reversal, because of their edge jumps),
  from labelling. Its analysis, like H5's, was committed before any Gemini call.
- 2026-11-18: Kept the end-hold of drops; measured how far it can reach and planned a check instead of
  rebuilding (see Limitations).

## Observations

- 2026-11-10: Showed a friend only the reversed ball-under-table clip. Unprompted he saw nothing wrong;
  asked to look for something odd, he said the ball "comes from nowhere". A human version of default vs checklist.
- 2026-11-18, from labelling the visibility sample:
  - Freezes hide wherever the speed is already near zero: at turning points (top of a bounce, end of a roll)
    and at the moment motion begins, where a freeze just looks like a delayed start (likely 0102's 1 s freeze
    at 0.03 s and 0072's at frame 0). 0304 (marble-run-y): only a brief hitch compared with the marble's
    motion on other levels.
  - Drops show only where physics fixes the speed (falling, bouncing). At the start of a pull or push,
    a skip just looks like a faster, harder action (0045 cut-paper).
  - A small jump in slow motion looks more like a dropped-frame glitch than impossible physics. If Gemini
    misses mild drops, check whether its reasons treat jumps as glitches.
  - Partial reversals carry two cues, the jumps at their edges and the backward motion (0382: the cup jumps
    to full, empties up into the container, jumps back). Led to H6.

## Subtle scenes (labelled reversible)

- ball-hits-nothing: lands too perfectly from below.
- ball-ramp: enters scene a bit off.
- ball-rolls-on-glass: bounces before edge of contact between glass and table.
- double-cradle: slight increase in velocity towards the end.

## Corruption check (2026-11-17)

### How the corruptions are made

- Every corruption only reorders frames. Output frame t shows input frame index[t], so every frame
  the judge sees is a real frame, and every clip keeps its length (duration is never a cue).
  Drops keep the length by holding the last frame (see End-hold under Limitations).
- **Motion detection** (`tbj/motion.py`): frames are shrunk to 160 px wide, made grey and lightly blurred.
  The motion signal is the share of pixels whose brightness changed by more than 6 levels between
  frames 3 apart (0.1 s). The median is the baseline (noise). The **motion window** runs from the first
  to the last frame above 20% of the peak. A clip whose peak is under 0.05% of pixels counts as static.
- **Placement:** freezes and drops start inside the motion window; partial reversals are centred in it;
  shuffles and the full reversal cover the whole clip. The exact position is random but reproducible:
  it is seeded by the base seed, the clip id and the condition name.
- **Consequence:** the 20% threshold is relative to the clip's strongest movement, so one sudden event
  (a reflection change, a pop, an impact) can shrink the window to that event and pull the corruptions onto it.
- Placement is deliberately random within the motion, not at the most meaningful moment: choosing
  the dramatic moment by hand would make corruptions easier to catch and bias the test.

### How check.html relates to all the clips

`check.html` shows 2 randomly chosen clips for each of the 12 corrupted conditions: 24 pairs, each
next to its clean version, with the corruption's timing above it. It was drawn from the 221 clips left
after excluding 0103 (2,652 corrupted clips, so about 1%); none of its pairs comes from the 4 clips
excluded later. It shows how corruptions *look*, not whether they were placed well in every clip.
The commands below check placement in all clips from the numbers in `conditions.csv`, and pick which
clips to open by hand.

### Commands used

1. Placement check, all clips:
```bash
python -c "
import pandas as pd
c = pd.read_csv('data/conditions.csv', dtype={'video_id': str})
clean = c[c.condition == 'clean'].copy()
clean['motion_s'] = (clean.motion_end - clean.motion_start) / clean.fps
clean['clip_s'] = clean.frames / clean.fps
print('Clips with no motion detected:', (~clean.has_motion.astype(bool)).sum())
print(clean.sort_values('motion_s')[['video_id', 'scenario', 'motion_s', 'clip_s']].to_string(index=False))
bad = c[c.corruption.isin(['freeze', 'drop']) & (c.motion_overlap == 0)]
print('Freezes/drops placed outside motion:', len(bad))
"
```

2. Where the corruptions land in chosen clips:
```bash
python -c "
import pandas as pd
c = pd.read_csv('data/conditions.csv', dtype={'video_id': str})
sel = c[c.video_id.isin(['0103', '0132', '0130', '0113']) & c.condition.isin(['freeze_medium', 'drop_strong'])].copy()
sel['starts_s'] = (sel.start_frame / sel.fps).round(2)
sel['lasts_s'] = (sel.length_frames / sel.fps).round(2)
sel['motion_s'] = (sel.motion_start / sel.fps).round(2).astype(str) + '-' + (sel.motion_end / sel.fps).round(2).astype(str)
print(sel[['video_id', 'scenario', 'condition', 'starts_s', 'lasts_s', 'motion_s']].to_string(index=False))
"
```

3. Clips with a short motion window at the very start (the pattern of 0103):
```bash
python -c "
import pandas as pd
c = pd.read_csv('data/conditions.csv', dtype={'video_id': str})
k = c[c.condition == 'clean'].copy()
k['from_s'] = (k.motion_start / k.fps).round(2)
k['to_s'] = (k.motion_end / k.fps).round(2)
edge = k[(k.from_s < 0.3) & (k.to_s - k.from_s < 1.0)]
print(len(edge), 'clips with a short motion window at the very start:')
print(edge[['video_id', 'scenario', 'from_s', 'to_s']].to_string(index=False))
"
```

### Results

- **Placement check:** 0 clips without motion, 0 freezes or drops outside motion. Motion windows match
  physics: falls take about 0.3 s, pours most of the clip. 73 of 222 clips have under 1 s of motion, which led to H5
  (72 of 217 after the exclusions).
- **Manual check** (`freeze_medium` and `drop_strong` of 4 clips chosen from the extremes):
  - 0103 marble-run-x: the window (0.0–0.2 s) caught a reflection of a door closing, not the marble.
    The corruptions hit the reflection and miss the marble. **Excluded.**
  - 0132 paint-on-glass: the window (0.43–0.67 s) is the brush starting to move. Freeze and drop visible on the brush. OK.
  - 0130 paint-on-glass: drop at 1.13 s (brush jumps before painting) and freeze at 6.47 s (brush moving
    after painting) are both visible. OK.
  - 0113 match-blows-balloon: freeze visible mid-pop. The drop is subtle: the pop is already almost instant,
    so skipping 0.5 s of it looks like a fast pop. Drops are hardest to see in events that happen in a flash. Fits H2.
- **Start-of-clip check:** 7 clips: 0103 and all 6 juice-in-water clips. The juice-in-water pattern repeats
  across every angle and take, so it is likely the juice entering the water at the start, not an artifact.
  Checked 0070: in clean, the juice pours from the first frame; freeze_medium visibly pauses the pour, briefly. The motion is real, so all 6 juice-in-water clips stay.
- **Decision:** 0103 excluded via `exclude_videos` in config.yaml; 221 clips remain (2,873 conditions).
Why 0103 is excluded rather than kept: the scene is labelled irreversible because of the marble, but its motion window caught a door reflection, so the freezes and drops pause or speed up the reflection. A door pausing is possible, so these clips are probably not impossible, and a judge answering “plausible” would be wrongly scored as a miss. Those misses would fall on freezes and drops, where H1 and H2 predict misses, so keeping it would tilt the results toward my hypotheses. Rule, applied to all 7 flagged clips: exclude a clip if its corruptions miss the event that made the scene irreversible (6 kept, 1 excluded). Decided before any Gemini call.
Alternatives considered: placing 0103's corruptions on the marble by hand (rejected: hand-picked placement for one clip, at the most visible moment, breaks the same-procedure-for-every-clip rule) and trimming its first 0.3 s so motion detection finds the marble (rejected: it can't be a general rule, since juice-in-water pours start at frame 0, so it would be a one-clip special case).
Idea for later: does the judge notice timing errors in small background events? This would need several clips where the background event itself can’t pause, with corruptions placed on it on purpose.


### The 24 pairs on check.html

| Condition | Clip | OK? | Note |
|---|---|---|---|
|reversal_mild|paper-fall-water · right take 2|yes|Lands on the aftermath, not the drop: water motion runs backward (grows by itself), with jumps at both edges of the reversed stretch. Likely miss at 1 fps.|
|reversal_mild|dominos-with-space|yes|Dominos come back up in an ipossible way (ignoring magnets) then go back down.|
|reversal_medium|napkin-soak · left take 1|yes|Very clear that napkin gets pulled up and unwet in an impossible way.|
|reversal_medium|match · left take 1|yes|Match lights again after coming out of the water: impossible. The states last long enough to be sampled at 1 fps, so a miss here points to the judge ignoring order, not to sampling.|
|reversal_strong|ball-hits-duck · left take 1|yes|Ball starts mmotion suddnely.|
|reversal_strong|balls-collide · center take 1|yes|Very clearly impossible as the balls momvments are gnereated without exterior force.|
|shuffle_mild|mug-breaks · right take 2|yes|Starts broken, jumps back to the intact mug, then it falls and breaks: impossible. The jump survives any frame rate, but the real fall-and-break still plays forward, so a judge summing up “a mug breaks” could miss the broken start.|
|shuffle_mild|duck-and-dominos · left take 2|yes|The dominos starts fallen and then jump back into place to then cleanly be knocked over.|
|shuffle_medium|ball-and-block-fall · left take 2|yes|The objects fall, then suddnely jump back up and then suddnely back down. If not juding the relation between frames, nothing seems off.|
|shuffle_medium|cut-paper · left take 2|yes|Order 3-2-1-0: segments play in reverse order, each forward. Starts fully cut, then jumps back three times to less-cut states, ending with the whole sheet just starting to be cut. Each frame looks normal and the jumps survive any sampling, so a miss would mean the judge ignored order.|
|shuffle_strong|liquid-overfill · right take 2|yes|States constanly change between spilled and unspilled|
|shuffle_strong|domino-in-juice · right take 2|yes|Domino jumps back up.|
|freeze_mild|unstable-block-stack · left take 2|yes|Only for a short moment (can be missed at 1fps and needs dense sampling) the cubes freeze right before they hit the table just a bit higher than the surface and then keep falling cleenly.|
|freeze_mild|ball-train · center take 2|yes|Ball stops mid-movment to then resume at full speed, contradicting inertia principles, for a short moment.|
|freeze_medium|duck-falls-in-box · right take 2|yes|The duck stops falling mid-air to then resume.|
|freeze_medium|napkin-soak · left take 2|yes|The napking stops falling mid-air to then resume.|
|freeze_strong|rolling-reflection · left take 1|yes|The ball stops for 1 s just before the object, then restarts, hits it and stops: impossible (it restarts from rest with nothing pushing it). The restart, hit and stop take under 1 s, so at 1 fps they can fall between two sampled frames, and the judge would see only “near the object” followed by “at rest after the hit”. Needs dense.|
|freeze_strong|paper-in-water · center take 2|yes|Paper floats without itself or water below it moving.|
|drop_mild|ball-in-basket · center take 1|yes, subtle|The 0.1 s cut removes the moment the ball touches the desk, so it bounces before reaching the desk: impossible. The bounce falls within one sampled gap even at 4 fps, so likely missed at both rates.|
|drop_mild|marble-run-y · right take 2|yes, subtle|Lands on the bounce: the marble skips part of its arc and is back on the track too soon, which is impossible for that height. But it looks like a plausible smaller bounce. Near the limit of what anyone can see.|
|drop_medium|duck-falls-in-box · right take 1|yes|The object is seen being let go but disappears instead of us seeing it falling inside the box.|
|drop_medium|paper-fall-water · center take 2 (0332)|yes, invisible|Skip lands on the settling, about 1.7 s after the paper hits the water. Not visible by eye, even frame by frame. Measurable but small: frames 139→140 change 1.26% of pixels vs 0.42–0.84% for neighbouring steps (the landing: 3.48%), on the water and the paper’s top edge.|
|drop_strong|paper-in-water · center take 2|yes|The paper ball on the water suddnely change its orientation without movemnt being seen and also stabilizes much faster on the water.|
|drop_strong|ball-hits-duck · center take 1|yes|Both the ball's direction change without showing its impact with the duck, and the duck is being pushed back without showing something causing it to move.|

0332 is marked invisible by the visibility rules of 2026-11-18 (it can't be seen even frame by frame).

## Screen for broken clips (`tbj spots`, 2026-11-17)

**What it does.** For each freeze, drop and partial reversal: the clean clip's strongest pixel change among
the affected frames, divided by the clip's peak. A clip with 3 or more of its 6 freezes and drops on still frames
(under 0.05) is flagged. Its scores decide nothing and are not used in the analysis: I look at each flagged clip,
the 0103 rule decides, and an exclusion goes into `exclude_videos` with its reason here. It is a screen, not a
measure of visibility (see Visibility sample for why).

**marble-run-x (excluded).** A reflection on the whiteboard drove motion detection in 4 of the 5 clips left
after 0103. Take 1 (0104, 0105): the window starts at 0.0 s on the reflection, as in 0103; the marble moves only
from about 2.4 s. Take 2 (0301, 0302): the marble moves from 2.8 s, but the strongest motion is a reflection at
6.2–7.1 s, so the window stretches to it. Only about 4 of their 24 freezes and drops land on the marble.
0303 (take 2, right) doesn't see the reflection: its window (2.77–4.13 s) is the marble, so it stays.

**Flagged after that, all kept (2026-11-18).** 0130 and 0131 (paint-on-glass): the strongest motion is the brush
starting while the camera's auto-exposure brightens the picture, but the brush moves all clip long, so the
corruptions land on it. 0139 (paper-smoke), 0263 (fill-glass-red-drink) and 0382 (water-in-juice): the strongest
motion is the event itself (burning paper, the pour). Their zeros come from steady motion, which the median
treats as noise.

**Start of clips (2026-11-18).** 46 freezes and drops start in the first 0.5 s, in 23 clips; 27 of them in juice-in-water,
whose window is the pour (checked on 0070). The other 19, in 8 scenes, all land on the scene’s own action, already under 
way: the claw or needle moving before the release (domino-in-juice, dominos-with-space), pours (fill-glass-red-drink,
water-in-juice), burning (match, paper-smoke), the rig turning the wrench holder or spinning the brush (magnet-wrench,
paint-on-glass). In 0132 the brightening and the brush start together. All kept.

## Visibility sample (2026-11-18)

**Why.** A corruption that lands on a near-still moment is impossible but invisible (0332 above). Such clips count
as misses, and only freezes, drops and partial reversals can be hit (shuffles and full reversals change the whole
clip). So they would tilt H1 in its own favour, and dilute H2 for drops.

**Tried and dropped: a pixel measure** (`tbj spots` as a control, planned 11-17, before any Gemini call). Checked
against my eyes it failed both ways: 0332's invisible drop scored 0.43, because water glints change many pixels
without a visible event; 0130's clearly visible brush freeze scored 0, because when something moves all clip long,
the median used as noise is itself motion. (A first version also averaged over the affected frames, which scored
longer corruptions as less visible.) A control that fails a check against the eye can't be reported, so it is
not used in the analysis.

**Method.** `tbj check --visibility`: a random sample of 4 clips per freeze, drop and partial-reversal condition
(32 in all), each labelled from the corrupted clip alone: **visible** (noticed at normal speed), **subtle** (only
at the stated time, frame by frame if needed), **invisible** (not even then, or only by comparing with the clean
clip). When unsure, the less visible label. Labels and notes: `labels/visibility.csv`.

**Bound.** Invisible corruptions can lower a detection rate by at most their own share: measured = (1 − u)·d + u·f,
so the loss is u·(d − f) ≤ u, where u is the invisible share, d the detection rate on visible ones and f the false-
alarm rate. Using the upper end of u's 95% interval: a gap larger than that can't come from invisible clips; a gap
close to it is inconclusive (not a reason to label more).

**Result.**

| Corruption | Labelled | Visible | Subtle | Invisible (95% CI) |
|---|---|---|---|---|
| Freeze | 12 | 11 | 1 | 0 (0–24%) |
| Drop | 12 | 9 | 0 | 3 (9–53%) |
| Partial reversal | 8 | 8 | 0 | 0 (0–32%) |

- **H1:** no invisible freezes or partial reversals. Invisible clips can explain at most about 24 points of a
  freeze–shuffle gap, and at most 32 for partial reversals (less for reversals pooled, since full reversals can't be invisible).
- **H2:** drops are where invisibility is real (about 1 in 4). Invisible drops are misses in both setups, so they
  can only shrink the gain from dense sampling on drops, never create it.
- **H6:** invisibility can only lower partial-reversal detection, the opposite of H6's prediction, so it can't produce it.

## Limitations (for the write-up)

- **Detection rates are a lower bound** on the judge's sensitivity: some corrupted clips are invisible even to a
  careful viewer (about 1 in 4 drops). The only human baseline is the 32-clip visibility sample, labelled by one person.
- **Motion detection follows the clip's strongest motion.** When that is a side event (a reflection, an exposure
  change), the window can follow it. Screened by hand (above), but a side event that leaves no still frames would not be flagged.
- **End-hold (drops).** Drops keep the clip's length by holding the last frame for as long as the skip. Where
  motion continues to the end, that's a second cue besides the jump. Given the judge's sampling it can only show up
  for strong drops at 4 fps (two identical frames at 7.5 and 7.75 s); at 1 fps the last sampled frame comes before
  the hold. Exposure: 16 of 217 clips have detected motion in their last 0.6 s (single-cradle 6, marble-run-y 4,
  liquid-on-duck 2, paint-on-glass 2, balls-collide 1, water-in-juice 1), slightly undercounted since slow motion
  under the threshold isn't detected. So the cue can raise strong-drop detection at 4 fps by at most about 7 points,
  and dense's gain on medium and strong drops (H2) by at most about 4. Plan: in the failure review, count how often
  the judge's reasons for flagged strong drops at 4 fps mention a stop or freeze at the end. Considered and rejected:
  slowing the end (a stutter or blended frames, which is a second corruption) and trimming all clips to 7.5 s so
  drops fill the end with real footage (a full rebuild and redoing the checks, for a cue the judge mostly can't see).

## Runs

