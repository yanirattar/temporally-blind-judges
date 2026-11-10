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
