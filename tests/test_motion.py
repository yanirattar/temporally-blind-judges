import numpy as np

from tbj import synthetic
from tbj.motion import motion_profile, motion_span


def _clip(move_from=None, move_to=None, n=120, size=(90, 160)):
    rng = np.random.default_rng(0)
    frames = np.full((n, *size, 3), 200, np.uint8)
    frames = np.clip(frames.astype(int) + rng.integers(-3, 4, frames.shape), 0, 255).astype(np.uint8)
    if move_from is not None:
        for t in range(n):
            x = 10 + int(np.clip(t - move_from, 0, move_to - move_from) * 1.5)
            frames[t, 40:50, x:x + 10] = 30
    return frames


def test_static_noisy_clip_has_no_motion():
    span = motion_span(motion_profile(_clip()))
    assert not span.has_motion


def test_small_mover_is_found():
    span = motion_span(motion_profile(_clip(40, 80)))
    assert span.has_motion
    assert abs(span.start - 40) <= 5 and abs(span.end - 80) <= 5


def test_synthetic_scenes_motion_matches_physics():
    # the ball is released at 1.0 s (take 1) and settles before 3 s
    span = motion_span(motion_profile(synthetic.make_clip("synth-ball-drop", "center", 1)))
    assert span.has_motion and 0.8 <= span.start / 30 <= 1.3 and span.end / 30 <= 3.2
    # the glass fills for 4 s from 1.0 s: slow, but must be detected
    span = motion_span(motion_profile(synthetic.make_clip("synth-fill-glass", "center", 1)))
    assert span.has_motion and span.end / 30 >= 4.5
