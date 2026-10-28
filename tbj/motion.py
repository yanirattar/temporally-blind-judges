"""Where in a clip things move.

Physics-IQ is shot with a static camera, so frame differences locate the action.
The signal is the share of pixels that changed by more than a few grey levels
between frames a short lag apart (after a light blur to suppress sensor noise).
Counting changed pixels, rather than averaging the difference over the whole
frame, keeps small movers such as a tennis ball or a rising liquid level visible.

Freezes, frame drops and partial reversals are placed inside the motion span;
outside it they would change nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionSpan:
    start: int        # first frame of the motion span
    end: int          # last frame of the motion span (inclusive)
    has_motion: bool  # False when the clip is essentially static
    peak: float       # peak share of changed pixels, above baseline

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def motion_profile(frames: np.ndarray, small_width: int = 160, lag: int = 3,
                   pixel_threshold: float = 6.0) -> np.ndarray:
    """Share of pixels whose grey level changed by more than `pixel_threshold`
    since `lag` frames earlier. Frame t gets the change between t-lag and t."""
    n, h, w, _ = frames.shape
    small_h = max(2, int(round(h * small_width / w)))
    grey = np.stack([
        cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (small_width, small_h), interpolation=cv2.INTER_AREA),
                                      cv2.COLOR_BGR2GRAY), (5, 5), 0)
        for f in frames
    ]).astype(np.float32)
    lag = max(1, min(lag, n - 1))
    changed = np.zeros(n, dtype=np.float32)
    changed[lag:] = (np.abs(grey[lag:] - grey[:-lag]) > pixel_threshold).mean(axis=(1, 2))
    # Centre the signal: the change between t-lag and t belongs to the frames in between.
    return np.roll(changed, -(lag // 2))


def motion_span(profile: np.ndarray, threshold: float = 0.2, smooth: int = 5,
                min_peak: float = 0.0005) -> MotionSpan:
    """Frames from the first to the last time the smoothed signal exceeds
    `threshold` x its peak above the baseline (the median: noise and flicker).

    `min_peak` is a share of pixels; below it the clip counts as static.
    """
    n = len(profile)
    if n < 3:
        return MotionSpan(0, max(n - 1, 0), False, 0.0)
    kernel = np.ones(smooth, dtype=np.float32) / smooth
    smoothed = np.convolve(profile, kernel, mode="same")
    above = smoothed - np.median(smoothed)
    peak = float(above.max())
    if peak < min_peak:
        return MotionSpan(0, n - 1, False, peak)
    active = np.flatnonzero(above >= threshold * peak)
    return MotionSpan(int(active[0]), int(active[-1]), True, peak)
