"""Time-only corruptions, expressed as frame-index maps.

A corruption never edits pixels. It returns `index`, an array where output frame t
shows input frame index[t]. Every frame the judge sees is therefore a real frame of
the original clip; only the order in time changes. All maps keep the clip length,
so duration is never a cue.

  reversal  a window of the clip plays backward (strong = the whole clip)
  shuffle   the clip is cut into k equal segments played in a shuffled order
  freeze    the scene pauses on one frame for m frames, then continues where it
            stopped (the last m frames of the clip are cut to keep the length)
  drop      m frames are removed, so moving objects jump ahead; the last frame is
            held at the end to keep the length (Physics-IQ clips end at rest)

Strengths come from config.yaml. Placement is random but reproducible: the seed is
derived from the base seed, the video id and the condition name.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .motion import MotionSpan

CORRUPTIONS = ("reversal", "shuffle", "freeze", "drop")
STRENGTHS = ("mild", "medium", "strong")


def condition_names() -> list[str]:
    """'clean' plus every corruption x strength, e.g. 'reversal_strong'."""
    return ["clean"] + [f"{c}_{s}" for c in CORRUPTIONS for s in STRENGTHS]


def split_condition(name: str) -> tuple[str, str]:
    """'freeze_medium' -> ('freeze', 'medium'); 'clean' -> ('clean', '')."""
    if name == "clean":
        return "clean", ""
    corruption, strength = name.rsplit("_", 1)
    if corruption not in CORRUPTIONS or strength not in STRENGTHS:
        raise ValueError(f"Unknown condition: {name}")
    return corruption, strength


@dataclass
class Corruption:
    name: str               # 'clean' or e.g. 'freeze_medium'
    corruption: str         # 'clean' or one of CORRUPTIONS
    strength: str           # '' for clean
    index: np.ndarray       # output frame t shows input frame index[t]
    start: int              # first affected input frame
    length: int             # number of affected frames (window, pause or gap)
    motion_overlap: float   # share of affected frames inside the motion span
    detail: str = ""        # e.g. the segment order of a shuffle

    @property
    def reversed_index(self) -> np.ndarray:
        """The same clip played backward: used for the paired judging setup."""
        return self.index[::-1].copy()


def rng_for(seed: int, *parts: object) -> np.random.Generator:
    digest = hashlib.sha256("|".join([str(seed), *map(str, parts)]).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _overlap(start: int, length: int, span: MotionSpan) -> float:
    if length <= 0 or not span.has_motion:
        return 0.0
    lo, hi = max(start, span.start), min(start + length - 1, span.end)
    return max(0, hi - lo + 1) / length


def _start_in_span(rng: np.random.Generator, span: MotionSpan, lo: int, hi: int) -> int:
    """A start frame inside the motion span, clamped to the legal range [lo, hi]."""
    a, b = max(lo, span.start), min(hi, span.end)
    if a > b:  # the motion span does not intersect the legal range
        a, b = lo, hi
    return int(rng.integers(a, b + 1))


def reversal(n: int, fraction: float, span: MotionSpan, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    index = np.arange(n)
    if fraction >= 1.0:
        return index[::-1].copy(), 0, n
    length = int(np.clip(round(fraction * n), 2, n))
    centre = _start_in_span(rng, span, 0, n - 1)
    start = int(np.clip(centre - length // 2, 0, n - length))
    index[start:start + length] = index[start:start + length][::-1]
    return index, start, length


def _keeps_neighbours(order: np.ndarray) -> bool:
    """True if some segment is still followed by its original successor. Such an order
    is really a coarser shuffle: 2-3-0-1 of 4 segments is just the halves swapped."""
    return bool(np.any(np.diff(order) == 1))


def shuffle(n: int, segments: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Cut into `segments` equal parts and play them in an order where no part is
    followed by its original successor, so k segments always means k - 1 cuts."""
    segments = int(np.clip(segments, 2, n))
    bounds = np.linspace(0, n, segments + 1).astype(int)
    pieces = [np.arange(bounds[i], bounds[i + 1]) for i in range(segments)]
    order = rng.permutation(segments)
    while _keeps_neighbours(order):
        order = rng.permutation(segments)
    return np.concatenate([pieces[i] for i in order]), "-".join(map(str, order))


def freeze(n: int, frames: int, span: MotionSpan, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    frames = int(np.clip(frames, 1, n - 2))
    start = _start_in_span(rng, span, 0, n - frames - 2)
    index = np.concatenate([
        np.arange(0, start + 1),
        np.full(frames, start),
        np.arange(start + 1, n - frames),
    ])
    return index, start, frames


def drop(n: int, frames: int, span: MotionSpan, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    frames = int(np.clip(frames, 1, n - 2))
    start = _start_in_span(rng, span, 1, n - frames - 1)
    index = np.concatenate([
        np.arange(0, start),
        np.arange(start + frames, n),
        np.full(frames, n - 1),
    ])
    return index, start, frames


def make_corruption(name: str, n: int, fps: float, span: MotionSpan, strengths: dict,
                    seed: int, video_id: object) -> Corruption:
    """Build one condition for a clip of n frames.

    `strengths` is the `corruptions:` block of config.yaml, e.g.
    {'reversal': {'mild': 0.25, ...}, 'freeze': {'mild': 0.25, ...}, ...}.
    Reversal values are fractions of the clip, shuffle values segment counts,
    freeze and drop values seconds.
    """
    corruption, strength = split_condition(name)
    if corruption == "clean":
        return Corruption(name, "clean", "", np.arange(n), 0, 0, 0.0)
    value = strengths[corruption][strength]
    rng = rng_for(seed, video_id, name)
    detail = ""
    if corruption == "reversal":
        index, start, length = reversal(n, float(value), span, rng)
        overlap = _overlap(start, length, span)
    elif corruption == "shuffle":
        index, detail = shuffle(n, int(value), rng)
        start, length = 0, n
        overlap = 1.0 if span.has_motion else 0.0
    elif corruption == "freeze":
        index, start, length = freeze(n, round(float(value) * fps), span, rng)
        overlap = 1.0 if span.has_motion and span.start <= start <= span.end else 0.0
    else:  # drop
        index, start, length = drop(n, round(float(value) * fps), span, rng)
        overlap = _overlap(start, length, span)
    assert len(index) == n and index.min() >= 0 and index.max() < n
    return Corruption(name, corruption, strength, index, start, length, overlap, detail)
