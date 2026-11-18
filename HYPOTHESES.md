hypotheses for the experiment's results:

• H1: at default settings, reversals and freezes are caught far less often than shuffles, because the judge relies on what single frames show. However, at 1 frame per second most freezes fall between sampled frames, so bad performance here is more attributed to sampling than the judge.
  Hypothesis is wrong if reversals are caught about as often as shuffles.

• H2: denser sampling helps short corruptions (freezes, drops) more than reversals. I also think we will see better relative improvement on freeze than drops since drops is more subtle and asks the model to compare several frames where as freeze is more of a change between subsequent frames. 
  Hypothesis is wrong if reversals gain as much or more or denser sampling didn't help.

• H3: The checklist raises detection, but false alarms on clean clips rise too, because the
  model becomes stricter. At 1 frame/s, normal motion between sparse frames may look like a
  jump to a primed judge, so many false alarms will come with a "jump or teleport" answer.
  Hypothesis is wrong if false alarms rise by less than 5 points.
  Test: the checklist at 4 frames/s on clean clips. If false alarms drop there, sampling was the cause.


• H4: The paired setup sharply improves reversal detection, but not freeze or drop detection.
  Both clips in the pair contain those errors, and the paired question asks only about
  direction, so the judge mostly picks the clip that plays forward rather than "neither".
  Hypothesis is wrong if reversal detection rises by less than 20 points, or if "neither" answers make
  freeze and drop detection rise.

• H5 (exploratory, added after the placement check, before any Gemini call): dense sampling helps more on clips whose main action is short. In 72 of 217 clips the detected motion lasts under 1 second, so at 1 frame/s the judge sees at most one frame of the action and can only compare the before and after states. At 4 frames/s it sees the action itself.
  Test: compare the gain in detection (dense minus default, medium and strong corruptions) for clips with under 1 s of motion versus clips with more.
  Hypothesis is wrong if the gain is about the same for both groups, or larger for long-action clips.

• H6 added on 2026-11-18, before any Gemini call, from labeling the visibility sample): partial reversals (¼ and ½ of the clip) are caught more often than the full reversal, even though the full reversal runs backward for longest. A partial reversal jumps at both edges of its backward stretch, and a jump can be caught without knowing which way time runs; the full reversal has no jumps, only direction. If the judge understood direction, the full reversal would be caught most.
  Test: detection of the ¼ and ½ reversals minus the full reversal, on the same clips, at default settings (and ½ vs full at 4 frames/s).
  Wrong if the full reversal is caught as often as the partial ones, or more.
