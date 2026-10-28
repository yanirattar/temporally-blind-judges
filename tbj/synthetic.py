"""Synthetic stand-ins for Physics-IQ clips, used by `tbj demo` and the tests.

Four simple irreversible scenes (a ball dropped and settling, a block sliding to a
stop, a glass filling, a bar toppling), each "filmed" from 3 perspectives in 2 takes,
named like Physics-IQ files so the whole pipeline runs on them unchanged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .video import write_video

SCENES = ["synth-ball-drop", "synth-slide-stop", "synth-fill-glass", "synth-topple"]
PERSPECTIVES = ["left", "center", "right"]
W, H, FPS, SECONDS = 320, 180, 30, 8


def _background(shift: int) -> np.ndarray:
    img = np.full((H, W, 3), (236, 232, 226), np.uint8)
    cv2.rectangle(img, (0, 140), (W, H), (120, 150, 185), -1)  # table
    cv2.line(img, (0, 140), (W, 140), (90, 110, 140), 2)
    cv2.rectangle(img, (20 + shift, 20), (60 + shift, 50), (200, 190, 175), -1)  # a picture on the wall
    return img


def _frame(scene: str, t: float, shift: int, t0: float) -> np.ndarray:
    img = _background(shift)
    cx = W // 2 + shift
    tt = max(0.0, t - t0)
    if scene == "synth-ball-drop":
        # drop from y=30 to the table (y=130), bounce with restitution 0.55
        g, y0, floor, e = 600.0, 30.0, 130.0, 0.55
        y, v, rem = y0, 0.0, tt
        dt = 1 / 600
        while rem > 0:
            v += g * dt
            y += v * dt
            if y >= floor:
                y, v = floor, -v * e if abs(v) > 20 else 0.0
            rem -= dt
        cv2.line(img, (cx - 20, 18), (cx + 20, 18), (60, 60, 60), 3)  # the release bar
        cv2.circle(img, (cx, int(y)), 10, (40, 120, 230), -1)
    elif scene == "synth-slide-stop":
        v0, mu = 160.0, 60.0  # px/s, deceleration px/s^2
        t_stop = v0 / mu
        tc = min(tt, t_stop)
        x = 30 + shift + v0 * tc - 0.5 * mu * tc ** 2
        cv2.rectangle(img, (int(x), 118), (int(x) + 28, 140), (60, 160, 90), -1)
    elif scene == "synth-fill-glass":
        level = min(1.0, tt / 4.0)
        top, bottom, left, right = 70, 138, cx - 22, cx + 22
        fill_top = int(bottom - level * (bottom - top - 8))
        if level > 0:
            cv2.rectangle(img, (left + 2, fill_top), (right - 2, bottom - 2), (60, 60, 210), -1)
        if 0 < tt < 4.0:
            cv2.line(img, (cx + 4, 20), (cx + 4, fill_top), (60, 60, 210), 3)  # the pouring stream
        cv2.rectangle(img, (left, top), (right, bottom), (90, 90, 90), 2)
        cv2.rectangle(img, (cx - 10, 8), (cx + 30, 20), (120, 120, 120), -1)  # the dispenser
    else:  # synth-topple
        alpha = 3.0  # rad/s^2, roughly constant for a short bar
        angle = min(np.pi / 2, 0.02 + 0.5 * alpha * tt ** 2) if tt > 0 else 0.02
        base = (cx - 30, 140)
        tip = (int(base[0] + 80 * np.sin(angle)), int(base[1] - 80 * np.cos(angle)))
        cv2.line(img, base, tip, (40, 80, 160), 9)
    return img


def make_clip(scene: str, perspective: str, take: int) -> np.ndarray:
    shift = {"left": -40, "center": 0, "right": 40}[perspective]
    t0 = 1.0 + 0.25 * (take - 1)
    rng = np.random.default_rng(int(hashlib.sha256(f"{scene}|{perspective}|{take}".encode()).hexdigest()[:8], 16))
    frames = []
    for i in range(FPS * SECONDS):
        f = _frame(scene, i / FPS, shift, t0).astype(np.int16)
        f += rng.integers(-2, 3, size=f.shape, dtype=np.int16)  # sensor noise
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(frames)


def write_dataset(raw_dir: Path) -> list[dict]:
    """Writes 24 clips named like Physics-IQ files; returns manifest rows (labelled irreversible)."""
    rows = []
    vid = 0
    for take in (1, 2):
        for scene in SCENES:
            for p in PERSPECTIVES:
                vid += 1
                name = f"{vid:04d}_perspective-{p}_take-{take}_trimmed-{scene}.mp4"
                path = raw_dir / f"take-{take}" / name
                if not path.exists():
                    write_video(path, make_clip(scene, p, take), FPS, crf=20)
                rows.append({"video_id": f"{vid:04d}", "scenario": scene, "perspective": p, "take": take,
                             "category": "Synthetic", "filename": name, "suggested": "likely_irreversible",
                             "label": "irreversible", "description": f"Synthetic scene {scene}."})
    return rows
