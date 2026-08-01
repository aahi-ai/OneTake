"""
Find an ffmpeg binary.

Order: an explicit FFMPEG_BINARY override, then whatever is on PATH, then the
static build that ships with imageio-ffmpeg. That last fallback is why this
module exists — it means `pip install -r requirements.txt` is genuinely the
only setup step, with no system package manager involved. Homebrew being
broken on one machine shouldn't stop the whole thing from running.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache


@lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    override = os.environ.get("FFMPEG_BINARY")
    if override and os.path.exists(override):
        return override

    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    raise RuntimeError(
        "No ffmpeg available. Either:\n"
        "  pip install imageio-ffmpeg      (no admin rights needed)\n"
        "  brew install ffmpeg             (macOS)\n"
        "  sudo apt install ffmpeg         (Ubuntu)"
    )