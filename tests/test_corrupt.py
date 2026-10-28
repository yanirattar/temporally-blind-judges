import numpy as np
import pytest

from tbj.corrupt import (CORRUPTIONS, STRENGTHS, condition_names, make_corruption, rng_for, shuffle,
                         split_condition)
from tbj.motion import MotionSpan

N, FPS = 240, 30.0
SPAN = MotionSpan(60, 150, True, 0.02)
STRENGTHS_CFG = {
    "reversal": {"mild": 0.25, "medium": 0.5, "strong": 1.0},
    "shuffle": {"mild": 2, "medium": 4, "strong": 8},
    "freeze": {"mild": 0.25, "medium": 0.5, "strong": 1.0},
    "drop": {"mild": 0.1, "medium": 0.25, "strong": 0.5},
}


def make(name, vid="0001", span=SPAN):
    return make_corruption(name, N, FPS, span, STRENGTHS_CFG, seed=1234, video_id=vid)


def test_condition_names():
    names = condition_names()
    assert names[0] == "clean" and len(names) == 13 and len(set(names)) == 13
    assert split_condition("freeze_medium") == ("freeze", "medium")
    with pytest.raises(ValueError):
        split_condition("melt_strong")


@pytest.mark.parametrize("name", condition_names())
def test_every_condition_keeps_length_and_real_frames(name):
    c = make(name)
    assert len(c.index) == N
    assert c.index.min() >= 0 and c.index.max() < N
    assert np.array_equal(c.reversed_index, c.index[::-1])


def test_clean_is_identity():
    assert np.array_equal(make("clean").index, np.arange(N))


def test_strong_reversal_is_whole_clip_backward():
    assert np.array_equal(make("reversal_strong").index, np.arange(N)[::-1])


@pytest.mark.parametrize("strength,frac", [("mild", 0.25), ("medium", 0.5)])
def test_partial_reversal_reverses_one_window_in_motion(strength, frac):
    c = make(f"reversal_{strength}")
    length = round(frac * N)
    assert c.length == length
    window = c.index[c.start:c.start + length]
    assert np.array_equal(window, np.arange(c.start, c.start + length)[::-1])
    outside = np.r_[c.index[:c.start], c.index[c.start + length:]]
    assert np.array_equal(outside, np.r_[np.arange(c.start), np.arange(c.start + length, N)])
    centre = c.start + length // 2
    assert SPAN.start <= centre <= SPAN.end or c.start in (0, N - length)


@pytest.mark.parametrize("strength,k", [("mild", 2), ("medium", 4), ("strong", 8)])
def test_shuffle_uses_all_segments_with_no_kept_neighbours(strength, k):
    c = make(f"shuffle_{strength}")
    order = list(map(int, c.detail.split("-")))
    assert sorted(order) == list(range(k))
    assert not any(b - a == 1 for a, b in zip(order, order[1:]))
    assert sorted(c.index) == list(range(N))  # a permutation of all frames


def test_shuffle_never_collapses_to_a_coarser_shuffle():
    for seed in range(300):
        _, detail = shuffle(N, 4, rng_for(seed, "x"))
        assert detail not in ("2-3-0-1", "0-1-2-3")


@pytest.mark.parametrize("strength,sec", [("mild", 0.25), ("medium", 0.5), ("strong", 1.0)])
def test_freeze_pauses_then_continues_without_jump(strength, sec):
    c = make(f"freeze_{strength}")
    m = round(sec * FPS)
    s = c.start
    assert SPAN.start <= s <= SPAN.end
    assert np.all(c.index[s:s + m + 1] == s)        # frame s held m extra times
    assert c.index[s + m + 1] == s + 1               # resumes where it stopped
    assert np.all(np.diff(c.index[s + m + 1:]) == 1)


@pytest.mark.parametrize("strength,sec", [("mild", 0.1), ("medium", 0.25), ("strong", 0.5)])
def test_drop_skips_frames_then_holds_last(strength, sec):
    c = make(f"drop_{strength}")
    m = round(sec * FPS)
    s = c.start
    assert c.index[s] == s + m                       # jump over m frames
    assert np.all(c.index[-m:] == N - 1)             # last frame held to keep the length
    assert c.motion_overlap > 0


def test_placement_is_reproducible_and_varies_by_clip():
    a1, a2, b = make("freeze_medium", "0001"), make("freeze_medium", "0001"), make("freeze_medium", "0002")
    assert np.array_equal(a1.index, a2.index)
    starts = {make("freeze_medium", f"{i:04d}").start for i in range(20)}
    assert len(starts) > 5


def test_all_conditions_differ_for_a_clip():
    seen = set()
    for name in condition_names():
        key = make(name).index.tobytes()
        assert key not in seen, name
        seen.add(key)


def test_static_clip_still_gets_valid_corruptions():
    static = MotionSpan(0, N - 1, False, 0.0)
    for c in CORRUPTIONS:
        for s in STRENGTHS:
            corr = make(f"{c}_{s}", span=static)
            assert len(corr.index) == N and corr.motion_overlap == 0.0
