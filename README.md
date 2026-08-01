# OneTake

Your first take is the final take.

Point a camera at yourself, talk through it however it comes out, and get back
the version you meant to record. OneTake finds the filler words, the stutters,
the dead pauses, and — the part editors actually lose hours to — the lines you
flubbed and immediately said again.

Everything runs locally. No API keys, no upload to anyone's server.

## Run it

```bash
pip install -r requirements.txt
uvicorn server.main:app --port 8000
```

Open <http://localhost:8000>. ffmpeg must be on your PATH.

Command line, if you'd rather skip the UI:

```bash
python -m engine.pipeline take.mp4                  # writes take_onetake.mp4
python -m engine.pipeline take.mp4 --edl --report   # + edit list + full cut log
python -m engine.pipeline take.mp4 --keep-retakes   # fillers only
```

## Three ways in

**Import** — drop a file, get a cut back.
**Record** — record in the browser; stopping starts the cut automatically.
**Edit** — every proposed cut is shown on a reel and struck through in the
transcript. Click anything to put it back, then re-render.

Analyze and render are separate endpoints, so restoring a cut re-renders in
seconds. Transcription only ever happens once per take.

## What it removes

| | |
|---|---|
| **Retake** | Adjacent attempts compared with `difflib`. Similarity ≥ 0.72, or a short fragment the next attempt subsumes, drops the earlier one. |
| **Stutter** | Repeated token runs (`I I I think`) keep only the last. Truncated false starts (`wh-` before `what`) go. |
| **Filler** | `um`, `uh`, `erm`. Soft fillers like `like` and `so` only count when isolated by pauses on both sides, so `I like it` survives. |
| **Dead air** | Gaps over 0.55s, trimmed to a fixed breath. Plus lead-in and trail-off. |

## Three decisions that matter

**Pauses are never closed to zero.** Every cut leaves 200ms of breath behind.
Strip all the air out of speech and it stops sounding like a person — it
becomes a shotgun of syllables with no room to breathe. This single constant is
most of the difference between output that sounds edited and output that sounds
broken.

**Every audio splice gets an 8ms fade.** Cutting raw PCM lands mid-waveform
almost every time, and the discontinuity is an audible click on every single
edit. No tutorial mentions this and it is the first thing a listener notices.

**The silence threshold adapts to the room.** A fixed dB gate fails across
setups by a wide margin — a treated room and a laptop mic in a kitchen are
nothing alike. The noise floor is estimated from the 20th percentile of the
recording's own energy distribution, and cut points snap to the nearest frame
below it so words never clip.

## Layout

```
onetake/
├── engine/
│   ├── transcribe.py   word timings (faster-whisper) + RMS envelope + silence snapping
│   ├── detect.py       the four detectors, plus cut merging and keep inversion
│   ├── cut.py          ffmpeg render — precise (re-encode) and fast (stream copy) + EDL
│   └── pipeline.py     orchestration, stats, CLI
├── server/main.py      FastAPI — analyze once, render many
└── web/
    ├── index.html
    ├── style.css
    └── app.js
```

## Known limits

- One speaker. Two people talking over each other will confuse retake detection.
- `base.en` is English-only; swap to `base` for other languages.
- The fast render path snaps to keyframes, so boundaries can drift up to a GOP.
  Precise is the default for that reason.
- Retake detection compares each attempt only against the next two. A line
  re-recorded much later in the take won't be matched.
