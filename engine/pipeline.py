"""
Orchestration. One call in, a finished cut and a report out.

Split deliberately into analyze() and render() rather than one function,
because the editor surface needs to re-render after you toggle cuts on and off
and transcription is by far the slowest step. Analyze once, render many times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .transcribe import transcribe, noise_floor, Word
from .detect import analyze as detect_analyze, Cut, to_keeps, Params
from .cut import render, write_edl


@dataclass
class Analysis:
    duration: float
    words: list[Word]
    cuts: list[Cut]
    keeps: list[tuple[float, float]]
    rms: object = field(repr=False, default=None)
    floor: float = 0.0

    @property
    def kept(self) -> float:
        return sum(e - s for s, e in self.keeps)

    @property
    def removed(self) -> float:
        return self.duration - self.kept

    def stats(self) -> dict:
        by_kind: dict[str, dict] = {}
        for c in self.cuts:
            k = by_kind.setdefault(c.kind, {"count": 0, "seconds": 0.0})
            k["count"] += 1
            k["seconds"] = round(k["seconds"] + c.dur, 2)
        return {
            "original_seconds": round(self.duration, 2),
            "final_seconds": round(self.kept, 2),
            "removed_seconds": round(self.removed, 2),
            "percent_removed": round(100 * self.removed / self.duration, 1) if self.duration else 0,
            "cut_count": len(self.cuts),
            "by_kind": by_kind,
        }

    def to_json(self) -> dict:
        return {
            "stats": self.stats(),
            "duration": round(self.duration, 3),
            "keeps": [[round(s, 3), round(e, 3)] for s, e in self.keeps],
            "cuts": [
                {"id": i, "start": round(c.start, 3), "end": round(c.end, 3),
                 "kind": c.kind, "text": c.text, "dur": round(c.dur, 3)}
                for i, c in enumerate(self.cuts)
            ],
            "transcript": [
                {"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in self.words
            ],
        }


def analyze(video: str | Path, model_size: str = "base.en",
            enable: dict[str, bool] | None = None, progress=None,
            params: Params | None = None) -> Analysis:
    words, rms, duration = transcribe(video, model_size=model_size, progress=progress)
    floor = noise_floor(rms)
    if progress:
        progress("finding cuts", 0.70)
    cuts, keeps = detect_analyze(words, duration, rms, floor, enable=enable, params=params)
    return Analysis(duration=duration, words=words, cuts=cuts, keeps=keeps,
                    rms=rms, floor=floor)


def redetect(a: Analysis, params: Params) -> Analysis:
    """Re-run the detectors with new settings. No transcription, so this is
    effectively instant — the whole point of caching words on the Analysis."""
    cuts, keeps = detect_analyze(a.words, a.duration, a.rms, a.floor, params=params)
    a.cuts, a.keeps = cuts, keeps
    return a

def apply_ranges(a: Analysis, ranges: list[tuple[float, float, str]]) -> Analysis:
    """Add hand-specified spans to the cut list.

    These bypass the detectors entirely — the user named a span of time, so it
    goes. Merging afterwards means a manual cut adjacent to a detected one
    fuses into a single clean removal instead of leaving a sliver between them.
    """
    from .detect import Cut, merge
    for s, e, label in ranges:
        a.cuts.append(Cut(s, e, "manual", label))
    a.cuts = merge(a.cuts, a.words)
    a.keeps = to_keeps(a.cuts, a.duration, a.rms, a.floor)
    return a


def rekeep(a: Analysis, disabled: set[int]) -> list[tuple[float, float]]:
    """Recompute keeps with some cuts switched off, for the editor surface."""
    live = [c for i, c in enumerate(a.cuts) if i not in disabled]
    return to_keeps(live, a.duration, a.rms, a.floor)


def process(video: str | Path, out: str | Path, model_size: str = "base.en",
            mode: str = "precise", enable: dict[str, bool] | None = None,
            progress=None) -> tuple[Path, Analysis]:
    a = analyze(video, model_size=model_size, enable=enable, progress=progress)
    if progress:
        progress("rendering", 0.80)
    path = render(video, a.keeps, out, mode=mode)
    if progress:
        progress("done", 1.0)
    return path, a


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="onetake", description="Cut the bad takes out of a recording.")
    p.add_argument("video")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("-m", "--model", default="base.en",
                   help="tiny.en | base.en | small.en")
    p.add_argument("--fast", action="store_true", help="stream-copy instead of re-encoding")
    p.add_argument("--keep-fillers", action="store_true")
    p.add_argument("--keep-retakes", action="store_true")
    p.add_argument("--edl", action="store_true", help="also write a CMX3600 edit list")
    p.add_argument("--report", action="store_true", help="print the full cut list as JSON")
    args = p.parse_args()

    src = Path(args.video)
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_onetake.mp4")

    def show(stage: str, frac: float) -> None:
        bar = "█" * int(frac * 28)
        print(f"\r  {bar:<28} {frac*100:3.0f}%  {stage:<14}", end="", flush=True)

    try:
        path, a = process(
            src, dst, model_size=args.model,
            mode="fast" if args.fast else "precise",
            enable={"filler": not args.keep_fillers, "retake": not args.keep_retakes},
            progress=show,
        )
    except RuntimeError as e:
        print(f"\n  {e}", file=sys.stderr)
        sys.exit(1)

    s = a.stats()
    print(f"\n\n  {src.name}  →  {path.name}")
    print(f"  {s['original_seconds']:.0f}s → {s['final_seconds']:.0f}s "
          f"({s['percent_removed']:.0f}% removed, {s['cut_count']} cuts)\n")
    for kind, v in sorted(s["by_kind"].items(), key=lambda kv: -kv[1]["seconds"]):
        print(f"    {kind:<10} {v['count']:>3} cuts   {v['seconds']:>6.1f}s")

    if args.edl:
        edl = write_edl(a.keeps, dst.with_suffix(".edl"))
        print(f"\n  edit list: {edl.name}")
    if args.report:
        print(json.dumps(a.to_json(), indent=2))
