"""
Turn a sentence of feedback into detector settings.

ViniClip sends feedback to an LLM and re-edits from the response. That's more
flexible, but it needs an API key, adds seconds of latency, and can return
something unusable mid-demo. Since every knob that matters is already a number
in Params, matching intent against those knobs directly is instant, offline,
and cannot fail in a way that leaves you with no video.

Re-detection costs milliseconds because the transcript is already cached — the
expensive part never runs twice. That's what makes "actually, keep the ums"
feel like an undo rather than a second render.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .detect import Params

# (pattern, human explanation, mutation)
RULES: list[tuple[str, str, dict]] = [
    # keep specific categories
    (r"\b(keep|leave|don'?t (cut|remove)|stop cutting|restore)\b.{0,24}\b(um+s?|uh+s?|filler)",
     "keeping filler words", {"filler": False}),
    (r"\b(keep|leave|don'?t (cut|remove)|stop cutting|restore)\b.{0,24}\b(pause|silence|gap|dead ?air|breath)",
     "keeping the pauses", {"dead_air": False}),
    (r"\b(keep|leave|don'?t (cut|remove)|stop cutting|restore)\b.{0,24}\b(retake|repeat|second take|both takes)",
     "keeping every take", {"retake": False}),
    (r"\b(keep|leave|don'?t (cut|remove)|stop cutting|restore)\b.{0,24}\b(stutter|stammer|repeat word)",
     "keeping stutters", {"stutter": False}),

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
    (r"\b(cut everything|maximum|be ruthless|as short as possible)\b",
     "cutting as hard as it can", {"breath": 0.10, "min_pause": 0.30, "retake_sim": 0.58}),
    (r"\b(barely|minimal|light touch|only the ums|just fillers?)\b",
     "only removing filler words", {"dead_air": False, "retake": False, "stutter": False}),
]


def parse(text: str, base: Params | None = None) -> tuple[Params, list[str]]:
    """Return updated params and a plain list of what changed.

    The explanations get shown back to the user. Silently altering the edit and
    saying "done" gives them no way to tell whether they were understood.
    """
    p = replace(base or Params())
    notes: list[str] = []
    low = " " + text.lower().strip() + " "

    for pattern, note, change in RULES:
        if re.search(pattern, low):
            for k, v in change.items():
                setattr(p, k, v)
            notes.append(note)

    return p, notes
