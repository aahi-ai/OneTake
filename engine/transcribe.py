"""
Audio in, word-level timestamps out.

Everything downstream keys off precise word boundaries, so this module exists
to produce two things and nothing else:

    words  — [Word(text, start, end, prob)] with real per-word timings
    rms    — a 10ms energy envelope, used to find the true edge of speech

Whisper's word timestamps are good but they round outward: the reported end of
a word usually lands a little after the speaker actually stopped. Cutting on
those raw numbers leaves audible stubs of the next syllable. So we keep the
energy track around and snap cut points to the nearest quiet frame.
"""

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 10
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0

    @property
    def clean(self) -> str:
        """Lowercased, stripped of punctuation. What the detectors match on."""
        return self.text.strip().strip(".,!?;:\"'—-").lower()


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it first:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg"
        )


def load_audio(video_path: str | Path) -> np.ndarray:
    """Decode any container to mono float32 PCM at 16 kHz via a pipe.

    Piping avoids writing a temp wav, which matters when the input is a 4K
    screen recording and disk is the slow part of the pipeline.
    """
    _require_ffmpeg()
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-threads", "0",
            "-i", str(video_path),
            "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE), "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="ignore").strip().splitlines()[-3:]
        raise RuntimeError("ffmpeg could not decode that file:\n" + "\n".join(tail))

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        raise RuntimeError("That file has no audio track, so there is nothing to cut.")
    return pcm


def rms_envelope(pcm: np.ndarray) -> np.ndarray:
    """Per-frame RMS energy. One value every 10ms."""
    n = len(pcm) // FRAME_LEN
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    frames = pcm[: n * FRAME_LEN].reshape(n, FRAME_LEN)
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12).astype(np.float32)


def noise_floor(rms: np.ndarray) -> float:
    """Estimate room tone from the quietest fifth of the recording.

    A fixed silence threshold fails badly across setups — a treated room and a
    laptop mic in a kitchen differ by 20 dB. Taking a low percentile of the
    energy distribution adapts to whatever room this was recorded in.
    """
    return float(np.percentile(rms, 20))


def snap_to_silence(t: float, rms: np.ndarray, floor: float,
                    direction: int, max_shift: float = 0.12) -> float:
    """Nudge a cut point toward the nearest quiet frame.

    direction = +1 searches forward (use for the start of a keep segment),
    -1 searches backward (use for the end). Bounded by max_shift so a cut
    never wanders into a neighbouring word.
    """
    gate = floor * 2.2
    idx = int(t * 1000 / FRAME_MS)
    limit = int(max_shift * 1000 / FRAME_MS)
    for step in range(limit):
        j = idx + direction * step
        if 0 <= j < len(rms) and rms[j] <= gate:
            return j * FRAME_MS / 1000.0
    return t


def transcribe(video_path: str | Path, model_size: str = "base.en",
               device: str = "auto", progress=None) -> tuple[list[Word], np.ndarray, float]:
    """Run ASR and return (words, rms envelope, duration seconds).

    model_size: tiny.en is ~2x faster and noticeably sloppier on word edges.
    base.en is the right default. small.en if the speaker has an accent the
    smaller models fumble.
    """
    from faster_whisper import WhisperModel

    pcm = load_audio(video_path)
    duration = len(pcm) / SAMPLE_RATE
    rms = rms_envelope(pcm)

    if progress:
        progress("loading model", 0.05)

    compute = "int8" if device in ("auto", "cpu") else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute)

    segments, _ = model.transcribe(
        pcm,
        word_timestamps=True,
        vad_filter=False,          # we do our own pause analysis; VAD would hide it
        condition_on_previous_text=False,  # stops repeated-phrase hallucination loops
        beam_size=5,
    )

    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            if not w.word.strip():
                continue
            words.append(Word(w.word.strip(), float(w.start), float(w.end),
                              float(getattr(w, "probability", 1.0))))
        if progress and duration:
            progress("transcribing", 0.05 + 0.55 * min(seg.end / duration, 1.0))

    if not words:
        raise RuntimeError("No speech was detected in that file.")
    return words, rms, duration
