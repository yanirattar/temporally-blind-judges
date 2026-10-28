"""Reading, writing and hashing video files.

Frames are decoded with OpenCV and encoded with ffmpeg (H.264, no audio), so every
clip the judge sees has the same codec, size and frame rate whatever its corruption.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: $TBJ_FFMPEG, then PATH, then the imageio-ffmpeg bundle."""
    exe = os.environ.get("TBJ_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "ffmpeg not found. Install it (e.g. `brew install ffmpeg` or `apt install ffmpeg`) "
            "or `pip install imageio-ffmpeg`."
        ) from e


def read_frames(path: str | os.PathLike, width: int | None = None) -> tuple[np.ndarray, float]:
    """Decode a video into a (n, h, w, 3) uint8 BGR array, optionally resized to `width`.

    The height keeps the aspect ratio and is rounded to an even number, which H.264 needs.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if width and frame.shape[1] != width:
            height = int(round(frame.shape[0] * width / frame.shape[1] / 2.0)) * 2
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from {path}")
    return np.stack(frames), float(fps)


def write_video(
    path: str | os.PathLike,
    frames: np.ndarray,
    fps: float,
    crf: int = 23,
    preset: str = "veryfast",
) -> None:
    """Encode frames (n, h, w, 3 BGR) as an H.264 MP4 without audio. Writes atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, _ = frames.shape
    if h % 2 or w % 2:
        raise ValueError(f"Frame size must be even for yuv420p, got {w}x{h}")
    tmp = path.with_name(path.stem + ".partial.mp4")
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp),
    ]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {path}:\n{proc.stderr.decode(errors='replace')[-2000:]}")
    tmp.replace(path)


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: str | os.PathLike) -> dict:
    """Frame count, fps and size, read with OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info
