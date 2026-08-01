"""
OneTake API.

    POST /api/analyze          upload a file, start a job
    GET  /api/jobs/{id}        poll progress, then read the cut list
    POST /api/render/{id}      render with some cuts toggled off
    GET  /api/source/{id}      original file, for the editor's timeline preview
    GET  /api/result/{id}      the finished cut
    GET  /api/edl/{id}         CMX3600 edit list

Analyze and render are separate endpoints on purpose. Transcription dominates
the runtime, so the editor toggles cuts and re-renders without paying for ASR
again — a re-render is a few seconds instead of a minute.

    uvicorn server.main:app --reload --port 8000
"""

from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

import subprocess

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import analyze as run_analyze, rekeep, redetect, apply_ranges, parse_feedback
from engine.cut import render, write_edl
from engine.ff import ffmpeg_bin

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)

app = FastAPI(title="OneTake")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _set(job_id: str, **kw) -> None:
    with LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def _job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No such job. It may have expired on a server restart.")
    return job


def normalise(src: Path) -> Path:
    """Rewrite a browser recording into a seekable container.

    MediaRecorder writes a live stream: no duration in the header, no seek
    index, and on some browsers a broken final cluster if the tab closed
    abruptly. ffmpeg can decode it linearly but trim/seek against it is
    unreliable, which shows up as cuts landing in the wrong place. One remux
    pass fixes all of that and costs a couple of seconds.
    """
    if src.suffix.lower() not in (".webm", ".mkv", ".ogg"):
        return src
    out = src.with_name("normalised.mp4")
    proc = subprocess.run(
        [ffmpeg_bin(), "-y", "-nostdin", "-fflags", "+genpts", "-i", str(src),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", str(out)],
        capture_output=True,
    )
    if proc.returncode != 0 or not out.exists():
        return src          # fall back to the original rather than failing outright
    return out


def _work(job_id: str, src: Path, model: str, enable: dict) -> None:
    def progress(stage: str, frac: float) -> None:
        _set(job_id, stage=stage, progress=round(frac, 3))

    try:
        progress("preparing", 0.02)
        src = normalise(src)
        _set(job_id, src=src)
        a = run_analyze(src, model_size=model, enable=enable, progress=progress)
        _set(job_id, analysis=a, result=a.to_json(), stage="analyzed", progress=0.75)

        out = src.with_name("cut.mp4")
        progress("rendering", 0.80)
        render(src, a.keeps, out, mode="precise")
        write_edl(a.keeps, src.with_name("cut.edl"))
        _set(job_id, output=out, stage="done", progress=1.0)
    except Exception as e:                              # surfaced to the UI verbatim
        _set(job_id, stage="error", error=str(e), progress=1.0)


@app.post("/api/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    model: str = Form("base.en"),
    fillers: bool = Form(True),
    stutters: bool = Form(True),
    dead_air: bool = Form(True),
    retakes: bool = Form(True),
):
    job_id = uuid.uuid4().hex[:12]
    d = WORK / job_id
    d.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    src = d / f"source{suffix}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    if src.stat().st_size == 0:
        raise HTTPException(400, "That upload was empty.")

    _set(job_id, id=job_id, src=src, stage="queued", progress=0.0, name=file.filename)
    enable = {"filler": fillers, "stutter": stutters,
              "dead_air": dead_air, "retake": retakes}
    threading.Thread(target=_work, args=(job_id, src, model, enable), daemon=True).start()
    return {"job_id": job_id}


@app.websocket("/ws/record/{job_id}")
async def record_socket(ws: WebSocket, job_id: str, ext: str = "webm",
                        model: str = "base.en"):
    """Stream a recording to disk while it is still being recorded.

    Borrowed from ViniClip's approach, simplified to one socket. Uploading a
    finished blob means the user watches a progress bar for a file that already
    exists on their own machine; streaming chunks as MediaRecorder emits them
    means the file is complete the instant they hit stop, and only analysis
    remains.

    A single muxed stream rather than separate video and audio sockets: two
    streams have to be re-synced on the server, and drift there is much worse
    than the small amount of latency this costs.
    """
    await ws.accept()
    d = WORK / job_id
    d.mkdir(parents=True, exist_ok=True)
    src = d / f"source.{ext}"

    _set(job_id, id=job_id, src=src, stage="recording", progress=0.0, name="live take")

    written = 0
    try:
        with src.open("wb") as f:
            while True:
                chunk = await ws.receive_bytes()
                f.write(chunk)
                written += len(chunk)
    except (WebSocketDisconnect, RuntimeError):
        pass

    if written == 0:
        _set(job_id, stage="error", error="No video data arrived from the browser.")
        return

    enable = {"filler": True, "stutter": True, "dead_air": True, "retake": True}
    threading.Thread(target=_work, args=(job_id, src, model, enable), daemon=True).start()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = _job(job_id)
    return {
        "id": job_id,
        "stage": j.get("stage"),
        "progress": j.get("progress", 0),
        "error": j.get("error"),
        "name": j.get("name"),
        "result": j.get("result") if j.get("stage") in ("analyzed", "done") else None,
        "ready": j.get("stage") == "done",
    }


@app.post("/api/render/{job_id}")
async def render_endpoint(job_id: str, disabled: str = Form(""), mode: str = Form("precise")):
    """Re-render with cuts the user switched off in the editor.

    `disabled` is a comma-separated list of cut ids to keep in the video.
    """
    j = _job(job_id)
    a = j.get("analysis")
    if a is None:
        raise HTTPException(409, "That job hasn't finished analysing yet.")

    off = {int(x) for x in disabled.split(",") if x.strip().isdigit()}
    keeps = rekeep(a, off)
    if not keeps:
        raise HTTPException(400, "Every segment is cut — nothing would be left.")

    out = j["src"].with_name("cut.mp4")
    _set(job_id, stage="rendering", progress=0.5)
    try:
        render(j["src"], keeps, out, mode=mode)
        write_edl(keeps, j["src"].with_name("cut.edl"))
    except Exception as e:
        _set(job_id, stage="error", error=str(e))
        raise HTTPException(500, str(e))

    _set(job_id, output=out, stage="done", progress=1.0)
    kept = sum(e - s for s, e in keeps)
    return {
        "ok": True,
        "final_seconds": round(kept, 2),
        "removed_seconds": round(a.duration - kept, 2),
        "percent_removed": round(100 * (a.duration - kept) / a.duration, 1),
    }


@app.post("/api/feedback/{job_id}")
async def feedback_endpoint(job_id: str, text: str = Form(...)):
    """Take plain-English notes, retune the detectors, re-render.

    No transcription happens here, so this returns in about the time the render
    takes. The notes list is echoed back so the user can see what was
    understood rather than guessing why the video changed.
    """
    j = _job(job_id)
    a = j.get("analysis")
    if a is None:
        raise HTTPException(409, "That job hasn't finished analysing yet.")

    params, notes, ranges = parse_feedback(text, j.get("params"), a.duration)
    if not notes:
        return {"ok": False, "notes": [],
                "message": "I didn't catch a change in that. Try things like "
                           "\"keep the ums\", \"it feels rushed\", or \"cut it tighter\"."}

    a = redetect(a, params)
    if ranges:
        a = apply_ranges(a, ranges)
    out = Path(j["src"]).with_name("cut.mp4")
    _set(job_id, stage="rendering", progress=0.5, params=params)
    try:
        render(j["src"], a.keeps, out, mode="precise")
        write_edl(a.keeps, Path(j["src"]).with_name("cut.edl"))
    except Exception as e:
        _set(job_id, stage="error", error=str(e))
        raise HTTPException(500, str(e))

    _set(job_id, output=out, stage="done", progress=1.0,
         analysis=a, result=a.to_json())
    return {"ok": True, "notes": notes, "result": a.to_json()}


@app.get("/api/source/{job_id}")
def source(job_id: str):
    return FileResponse(_job(job_id)["src"])


@app.get("/api/result/{job_id}")
def result(job_id: str):
    j = _job(job_id)
    out = j.get("output")
    if not out or not Path(out).exists():
        raise HTTPException(404, "Not rendered yet.")
    return FileResponse(out, filename="onetake_cut.mp4", media_type="video/mp4")


@app.get("/api/edl/{job_id}")
def edl(job_id: str):
    p = Path(_job(job_id)["src"]).with_name("cut.edl")
    if not p.exists():
        raise HTTPException(404, "No edit list for that job.")
    return FileResponse(p, filename="onetake.edl", media_type="text/plain")


@app.exception_handler(RuntimeError)
async def runtime_error(_, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"error": str(exc)})


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
