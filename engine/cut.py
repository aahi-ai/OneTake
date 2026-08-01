"""
Turn a keep-list into a finished video file.

Two paths:

    precise — one ffmpeg pass, trim/atrim + concat filter. Frame-accurate, and
              re-encodes. This is the default because a seam in the demo video
              is worse than waiting twenty seconds.

    fast    — extract each segment with -c copy and concat the results. Seconds
              instead of minutes, but cuts snap to the nearest keyframe, so a
              segment boundary can drift up to a GOP (often ~2s). Fine for
              previewing, wrong for the final render.

The detail that matters either way: an 8ms fade at both ends of every audio
segment. Splicing raw PCM at an arbitrary sample almost always lands mid-
waveform, and the discontinuity is an audible click on every single cut. Nobody
mentions this in tutorials and it's the first thing a viewer notices.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

FADE = 0.008  # seconds — long enough to kill the click, short enough to be inaudible


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="ignore").strip().splitlines()[-6:]
        raise RuntimeError("ffmpeg failed:\n" + "\n".join(tail))


def render_precise(src: str | Path, keeps: list[tuple[float, float]],
                   dst: str | Path, crf: int = 20) -> Path:
    """Single-pass filter_complex concat. Frame-accurate."""
    if not keeps:
        raise ValueError("Nothing left to keep — every segment was cut.")

    parts, labels = [], []
    for i, (s, e) in enumerate(keeps):
        d = e - s
        parts.append(
            f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={FADE},"
            f"afade=t=out:st={max(0.0, d - FADE):.4f}:d={FADE}[a{i}]"
        )
        labels.append(f"[v{i}][a{i}]")

    graph = ";".join(parts) + ";" + "".join(labels) + \
            f"concat=n={len(keeps)}:v=1:a=1[vout][aout]"

    # A long keep-list produces a filter graph past the shell's argument limit,
    # so it always goes to a file rather than the command line.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(graph)
        graph_path = f.name

    _run([
        "ffmpeg", "-y", "-nostdin",
        "-i", str(src),
        "-filter_complex_script", graph_path,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(dst),
    ])
    Path(graph_path).unlink(missing_ok=True)
    return Path(dst)


def render_fast(src: str | Path, keeps: list[tuple[float, float]],
                dst: str | Path) -> Path:
    """Stream-copy each segment, then concat. Fast, keyframe-quantised."""
    if not keeps:
        raise ValueError("Nothing left to keep — every segment was cut.")

    tmp = Path(tempfile.mkdtemp(prefix="onetake_"))
    listing = tmp / "segments.txt"
    with listing.open("w") as lf:
        for i, (s, e) in enumerate(keeps):
            piece = tmp / f"{i:04d}.mp4"
            _run([
                "ffmpeg", "-y", "-nostdin",
                "-ss", f"{s:.4f}", "-to", f"{e:.4f}",
                "-i", str(src), "-c", "copy", "-avoid_negative_ts", "1",
                str(piece),
            ])
            lf.write(f"file '{piece}'\n")

    _run([
        "ffmpeg", "-y", "-nostdin", "-f", "concat", "-safe", "0",
        "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(dst),
    ])
    return Path(dst)


def render(src, keeps, dst, mode: str = "precise") -> Path:
    return render_fast(src, keeps, dst) if mode == "fast" else render_precise(src, keeps, dst)


def write_edl(keeps: list[tuple[float, float]], dst: str | Path,
              fps: float = 30.0) -> Path:
    """Export a CMX3600 edit decision list.

    Cheap to produce and it means the output isn't a dead end — the cut can be
    opened in Premiere or Resolve and adjusted by hand.
    """
    def tc(t: float) -> str:
        f = int(round(t * fps))
        h, rem = divmod(f, int(3600 * fps))
        m, rem = divmod(rem, int(60 * fps))
        s, fr = divmod(rem, int(fps))
        return f"{h:02d}:{m:02d}:{s:02d}:{fr:02d}"

    lines = ["TITLE: ONETAKE", "FCM: NON-DROP FRAME", ""]
    rec = 0.0
    for i, (s, e) in enumerate(keeps, 1):
        d = e - s
        lines.append(f"{i:03d}  AX       AA/V  C        "
                     f"{tc(s)} {tc(e)} {tc(rec)} {tc(rec + d)}")
        rec += d
    Path(dst).write_text("\n".join(lines) + "\n")
    return Path(dst)
