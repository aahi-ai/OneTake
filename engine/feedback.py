"""
Turn a sentence of feedback into edits.

Two kinds of instruction arrive here and they work completely differently:

  tuning   "keep the ums", "it feels rushed", "cut it tighter"
           These adjust detector thresholds. Every knob that matters is already
           a number in Params, so intent maps straight onto those numbers.

  surgery  "remove the last 2 seconds", "cut from 0:12 to 0:19"
           These name a span of time directly. No detector is involved — the
           range just gets added to the cut list.

ViniClip sends feedback to an LLM and re-edits from the response. More
flexible, but it needs an API key, adds seconds of latency, and can return
something unusable mid-demo. Matching intent directly is instant, offline, and
cannot fail in a way that leaves you with no video.

Re-detection costs milliseconds because the transcript is already cached, so
either kind of change feels like an undo rather than a second render.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .detect import Params

# ── tuning rules ────────────────────────────────────────────────────────────
# (pattern, human explanation, mutation)
RULES: list[tuple[str, str, dict]] = [
    # turn categories OFF
    (r"\b(keep|leave|don'?t (cut|remove)|stop cutting|restore)\b.{0,24}\b(um+s?|uh+s?|filler)",
     "keeping filler words", {"filler": False}),
    (r"\b(keep|leave|don'?t (cut|remove)|stop cutting|restore)\b.{0,24}\b(pause|silence|gap|dead ?air|breath)",
     "keeping the pauses", {"dead_air": False}),
    (r"\b(keep|leave|don'?t (cut|remove)|stop cutting|restore)\b.{0,24}\b(retake|repeat|second take|both takes)",
     "keeping every take", {"retake": False}),
    (r"\b(keep|leave|don'?t (cut|remove)|stop cutting|restore)\b.{0,24}\b(stutter|stammer|repeat word)",
     "keeping stutters", {"stutter": False}),

    # turn categories ON, harder. "remove all the ums" means the default pass
    # missed some, so switch on aggressive mode rather than just re-enabling.
    (r"\b(remove|cut|kill|get rid of|delete|take out)\b.{0,24}\b(um+s?|uh+s?|filler)",
     "cutting every filler, including the borderline ones",
     {"filler": True, "soft_filler": True}),
    (r"\b(remove|cut|kill|get rid of|delete|take out)\b.{0,24}\b(pause|silence|gap|dead ?air)",
     "cutting the pauses harder", {"dead_air": True, "min_pause": 0.38}),
    (r"\b(remove|cut|kill|get rid of|delete|take out)\b.{0,24}\b(retake|repeat|extra take)",
     "cutting retakes more eagerly", {"retake": True, "retake_sim": 0.62}),
    (r"\b(remove|cut|kill|get rid of|delete|take out)\b.{0,24}\b(stutter|stammer)",
     "cutting stutters", {"stutter": True}),

    # pacing
    (r"\b(too (fast|rushed|choppy|abrupt|tight)|rushed|choppy|no room|can'?t breathe|breathing room|slow(er)? (it )?down|less aggressive|too much)\b",
     "leaving more room between lines", {"breath": 0.42, "min_pause": 1.0}),
    (r"\b(too (slow|loose|long)|tighter|snappier|cut more|more aggressive|shorter|speed (it )?up|drags?)\b",
     "cutting tighter", {"breath": 0.12, "min_pause": 0.38}),

    # retake sensitivity
    (r"\b(missed a retake|didn'?t catch|catch more retakes|still repeated|said (it|that) twice)\b",
     "catching retakes more eagerly", {"retake_sim": 0.60}),
    (r"\b(cut the wrong take|removed (a )?good|too many retakes|deleted something i wanted|over ?cut)\b",
     "being stricter about what counts as a retake", {"retake_sim": 0.86}),

    # blunt instruments
    (r"\b(cut everything(?!\s+(?:before|after))|maximum|be ruthless|as short as possible)\b",
     "cutting as hard as it can",
     {"breath": 0.10, "min_pause": 0.30, "retake_sim": 0.58, "soft_filler": True}),
    (r"\b(barely|minimal|light touch|only the ums|just fillers?)\b",
     "only removing filler words", {"dead_air": False, "retake": False, "stutter": False}),
]


# ── time parsing ────────────────────────────────────────────────────────────
CLIP = r"(?:remove|cut|trim|drop|delete|take out|lose|chop)"
UNIT = r"(seconds?|secs?|s|minutes?|mins?|m)"
NUM = r"(\d+(?:\.\d+)?)"
STAMP = r"(\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)"


def _secs(token: str, unit: str = "s") -> float:
    """Accept 1:05, 65, or 2 with a unit word. Returns seconds."""
    token = token.strip()
    if ":" in token:
        m, s = token.split(":", 1)
        return int(m) * 60 + float(s)
    v = float(token)
    return v * 60 if unit.startswith("m") and unit != "ms" else v


def find_ranges(text: str, duration: float) -> list[tuple[float, float, str]]:
    """Pull explicit time spans out of the feedback.

    Bounded to the clip's real duration so "remove the last 30 seconds" on a
    12-second take trims to the start rather than producing a negative span.
    """
    out: list[tuple[float, float, str]] = []
    t = text.lower()

    # "remove the last 2 seconds" / "cut the final 30s"
    for m in re.finditer(rf"{CLIP}\s+(?:the\s+)?(?:last|final|ending?)\s+{NUM}\s*{UNIT}?", t):
        n = _secs(m.group(1), m.group(2) or "s")
        out.append((max(0.0, duration - n), duration, f"last {n:g}s"))

    # "remove the first 3 seconds" / "trim the opening 5s"
    for m in re.finditer(rf"{CLIP}\s+(?:the\s+)?(?:first|opening|beginning|start(?:ing)?)\s+{NUM}\s*{UNIT}?", t):
        n = _secs(m.group(1), m.group(2) or "s")
        out.append((0.0, min(n, duration), f"first {n:g}s"))

    # "cut from 0:12 to 0:19"
    for m in re.finditer(rf"{CLIP}?\s*(?:from\s+)?{STAMP}\s*(?:to|-|–|until|through)\s*{STAMP}", t):
        a, b = _secs(m.group(1)), _secs(m.group(2))
        if b > a:
            out.append((max(0.0, a), min(b, duration), f"{a:g}s–{b:g}s"))

    # "remove everything after 0:30" / "cut everything before 0:05"
    for m in re.finditer(rf"{CLIP}\s+(?:everything\s+)?after\s+{STAMP}", t):
        a = _secs(m.group(1))
        out.append((min(a, duration), duration, f"after {a:g}s"))
    for m in re.finditer(rf"{CLIP}\s+(?:everything\s+)?before\s+{STAMP}", t):
        b = _secs(m.group(1))
        out.append((0.0, min(b, duration), f"before {b:g}s"))

    # de-duplicate spans the patterns above may both have matched
    seen, uniq = set(), []
    for s, e, label in out:
        key = (round(s, 2), round(e, 2))
        if key not in seen and e - s > 0.05:
            seen.add(key)
            uniq.append((s, e, label))
    return uniq


def parse(text: str, base: Params | None = None, duration: float = 0.0):
    """Return (params, notes, ranges).

    notes get shown back to the user — silently altering the edit and saying
    "done" leaves them no way to tell whether they were understood.
    """
    p = replace(base or Params())
    notes: list[str] = []
    low = " " + text.lower().strip() + " "

    ranges = find_ranges(low, duration) if duration else []
    for _, _, label in ranges:
        notes.append(f"removing {label}")

    for pattern, note, change in RULES:
        if re.search(pattern, low):
            # If the user named a time span, a bare "remove"/"cut" verb in the
            # same sentence belongs to that span — it isn't also a request to
            # retune a whole detector category.
            if ranges and note.startswith("cutting"):
                continue
            for k, v in change.items():
                setattr(p, k, v)
            notes.append(note)

    return p, notes, ranges