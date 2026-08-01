"""
Decide what to remove.

Four detectors, in the order they run:

    1. dead air   — pauses longer than a beat, trimmed but never closed flat
    2. filler     — um, uh, standalone discourse words
    3. stutter    — "I- I- I think", "the the", "wh- what"
    4. retake     — you flubbed a line, said it again, and only the last one counts

Filler removal is the commoditized part; every editor ships it. Retake
detection is the part that actually saves an hour, because the expensive thing
in a recording session isn't the ums, it's hunting through footage for which
attempt was clean.

Design rule that everything else bends around: NEVER close a pause to zero.
Speech with all the air removed sounds inhuman — a shotgun of syllables with no
breath. Every cut here leaves BREATH seconds behind, and every keep segment is
padded by PAD so consonant onsets don't get clipped.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import numpy as np

from .transcribe import Word, snap_to_silence

# ── tuning ──────────────────────────────────────────────────────────────────
BREATH = 0.20          # seconds of pause always left between sentences
PAD = 0.06             # padding kept around every retained span
MIN_PAUSE = 0.55       # a gap longer than this is dead air worth trimming
SENTENCE_GAP = 0.60    # a gap longer than this starts a new "attempt"
RETAKE_SIM = 0.72      # difflib ratio above which two attempts are the same line
MIN_KEEP = 0.25        # keeps shorter than this are slivers between adjacent cuts,
                       # not speech. Left in, they play as an audible fragment.
JOIN_GAP = 0.15        # cuts closer together than this get merged into one

FILLERS = {
    "um", "uh", "umm", "uhh", "erm", "er", "ah", "ahh", "hmm", "mm", "mhm",
    "eh", "uhm", "hm",
}
# Words that are only filler when they stand alone between pauses. "like" in
# "I like it" must survive; "like," floating on its own must not.
SOFT_FILLERS = {"like", "so", "basically", "actually", "literally", "right", "okay"}


@dataclass
class Params:
    """Tunables. Bundled so feedback can adjust them without re-transcribing."""
    breath: float = BREATH
    min_pause: float = MIN_PAUSE
    retake_sim: float = RETAKE_SIM
    filler: bool = True
    stutter: bool = True
    dead_air: bool = True
    retake: bool = True
    soft_filler: bool = False    # also cut like/so/basically even mid-sentence


@dataclass
class Cut:
    start: float
    end: float
    kind: str          # dead_air | filler | stutter | retake
    text: str = ""

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Attempt:
    """A run of words uninterrupted by a long pause. Roughly one sentence."""
    words: list[Word]

    @property
    def start(self) -> float: return self.words[0].start

    @property
    def end(self) -> float: return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.clean for w in self.words if w.clean)


# ── 1. dead air ─────────────────────────────────────────────────────────────
def find_dead_air(words: list[Word], duration: float,
                  min_pause: float = MIN_PAUSE, breath: float = BREATH) -> list[Cut]:
    """Trim long gaps down to BREATH. Also trims lead-in and trail-off."""
    cuts: list[Cut] = []

    if words[0].start > min_pause:
        cuts.append(Cut(0.0, max(0.0, words[0].start - breath), "dead_air", "lead-in"))

    for a, b in zip(words, words[1:]):
        gap = b.start - a.end
        if gap > min_pause:
            # keep BREATH split either side of the cut so the edit breathes
            cuts.append(Cut(a.end + breath / 2, b.start - breath / 2, "dead_air",
                            f"{gap:.1f}s pause"))

    if duration - words[-1].end > min_pause:
        cuts.append(Cut(words[-1].end + breath, duration, "dead_air", "trail-off"))

    return cuts


# ── 2. filler ───────────────────────────────────────────────────────────────
def find_fillers(words: list[Word], soft: bool = False) -> list[Cut]:
    cuts: list[Cut] = []
    for i, w in enumerate(words):
        c = w.clean
        if not c:
            continue
        if c in FILLERS:
            cuts.append(Cut(w.start, w.end, "filler", w.text))
            continue
        # Soft fillers normally only count when isolated by pauses on both
        # sides, so "I like it" survives. Aggressive mode drops that guard —
        # useful when someone says "like" forty times, risky otherwise.
        if c in SOFT_FILLERS:
            if soft:
                cuts.append(Cut(w.start, w.end, "filler", w.text))
                continue
            before = w.start - words[i - 1].end if i > 0 else 9.0
            after = words[i + 1].start - w.end if i < len(words) - 1 else 9.0
            if before > 0.25 and after > 0.25:
                cuts.append(Cut(w.start, w.end, "filler", w.text))
    return cuts


# ── 3. stutter ──────────────────────────────────────────────────────────────
def find_stutters(words: list[Word]) -> list[Cut]:
    """Drop all but the last of a repeated-token run, and drop truncated
    false starts like 'wh-' immediately before 'what'."""
    cuts: list[Cut] = []
    i = 0
    while i < len(words):
        c = words[i].clean
        if not c:
            i += 1
            continue

        # exact repeats: "I I I think" -> keep the final "I"
        j = i
        while j + 1 < len(words) and words[j + 1].clean == c:
            j += 1
        if j > i:
            for w in words[i:j]:
                cuts.append(Cut(w.start, w.end, "stutter", w.text))
            i = j + 1
            continue

        # truncated false start: short fragment that prefixes the next word
        if i + 1 < len(words):
            nxt = words[i + 1].clean
            gap = words[i + 1].start - words[i].end
            if (1 <= len(c) <= 3 and nxt.startswith(c) and len(nxt) > len(c)
                    and gap < 0.45):
                cuts.append(Cut(words[i].start, words[i].end, "stutter", words[i].text))
        i += 1
    return cuts


# ── 4. retake ───────────────────────────────────────────────────────────────
def split_attempts(words: list[Word]) -> list[Attempt]:
    groups, cur = [], [words[0]]
    for prev, w in zip(words, words[1:]):
        if w.start - prev.end > SENTENCE_GAP:
            groups.append(Attempt(cur))
            cur = [w]
        else:
            cur.append(w)
    groups.append(Attempt(cur))
    return groups


def find_retakes(words: list[Word], sim_threshold: float = RETAKE_SIM) -> list[Cut]:
    """Drop earlier attempts at a line that was immediately re-recorded.

    Two patterns, both compared only against nearby attempts because a real
    repeat happens seconds later, not minutes:

      near-duplicate — you said the line, disliked it, said it again
      false start    — a short fragment that the next attempt subsumes
    """
    attempts = split_attempts(words)
    cuts: list[Cut] = []
    dropped: set[int] = set()

    for i, a in enumerate(attempts):
        if i in dropped or not a.text:
            continue
        # look ahead a couple of attempts: "the line, an aside, the line again"
        for k in range(i + 1, min(i + 3, len(attempts))):
            b = attempts[k]
            if k in dropped or not b.text:
                continue

            sim = difflib.SequenceMatcher(None, a.text, b.text).ratio()
            subsumed = (len(a.text) < len(b.text) * 0.6
                        and b.text.startswith(a.text[: max(6, len(a.text) // 2)]))

            if sim >= sim_threshold or subsumed:
                kind = "false start" if subsumed else f"retake ({sim:.0%} match)"
                cuts.append(Cut(max(0.0, a.start - PAD), a.end + PAD, "retake",
                                f"{kind}: {a.text[:60]}"))
                dropped.add(i)
                break
    return cuts


# ── merge into a keep list ──────────────────────────────────────────────────
def _speech_between(words: list[Word], a: float, b: float) -> bool:
    """Is there any actual speech in the window (a, b)?"""
    return any(w.end > a and w.start < b for w in words)


def merge(cuts: list[Cut], words: list[Word] | None = None) -> list[Cut]:
    """Union overlapping cuts, then bridge any two cuts with nothing but
    silence between them.

    Time-thresholding alone leaves slivers: two cuts 0.3s apart with no words
    in the gap produce a 0.3s fragment of room tone that plays as a glitch.
    Checking for speech instead of checking the clock removes them properly,
    and is happy to bridge a long gap when that gap is genuinely empty.

    Retake wins the label when two kinds collide, since it's the more
    informative thing to report.
    """
    if not cuts:
        return []
    rank = {"retake": 3, "stutter": 2, "filler": 1, "dead_air": 0}
    ordered = sorted(cuts, key=lambda c: c.start)
    out = [ordered[0]]

    for c in ordered[1:]:
        last = out[-1]
        overlapping = c.start <= last.end + JOIN_GAP
        bridgeable = (words is not None
                      and c.start > last.end
                      and not _speech_between(words, last.end, c.start))
        if overlapping or bridgeable:
            if rank[c.kind] > rank[last.kind]:
                last.kind, last.text = c.kind, c.text
            last.end = max(last.end, c.end)
        else:
            out.append(c)
    return out


def to_keeps(cuts: list[Cut], duration: float, rms: np.ndarray,
             floor: float) -> list[tuple[float, float]]:
    """Invert the cut list into spans to keep, padded and snapped to silence."""
    keeps, t = [], 0.0
    for c in cuts:
        if c.start > t:
            keeps.append((t, c.start))
        t = max(t, c.end)
    if t < duration:
        keeps.append((t, duration))

    out = []
    for s, e in keeps:
        s = snap_to_silence(max(0.0, s - PAD), rms, floor, direction=+1)
        e = snap_to_silence(min(duration, e + PAD), rms, floor, direction=-1)
        if e - s >= MIN_KEEP:
            out.append((s, e))
    return out


def analyze(words: list[Word], duration: float, rms: np.ndarray, floor: float,
            enable: dict[str, bool] | None = None, params: Params | None = None):
    """Run every enabled detector and return (cuts, keeps).

    Re-running this with different params costs milliseconds, because the
    expensive step — transcription — already happened. That's what makes
    "actually, keep the ums" feel instant instead of a second full pass.
    """
    p = params or Params()
    if enable:
        p.filler = enable.get("filler", p.filler)
        p.stutter = enable.get("stutter", p.stutter)
        p.dead_air = enable.get("dead_air", p.dead_air)
        p.retake = enable.get("retake", p.retake)

    cuts: list[Cut] = []
    if p.retake:
        cuts += find_retakes(words, p.retake_sim)
    if p.stutter:
        cuts += find_stutters(words)
    if p.filler:
        cuts += find_fillers(words, p.soft_filler)
    if p.dead_air:
        cuts += find_dead_air(words, duration, p.min_pause, p.breath)

    cuts = merge(cuts, words)
    return cuts, to_keeps(cuts, duration, rms, floor)