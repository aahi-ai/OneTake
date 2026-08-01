/* onetake front end
   record → cut → request changes → download.
   No drag overlay: it was covering the whole app when a dragleave got missed.
   Files come in through the picker only. */

const $ = (s) => document.querySelector(s);

const S = { job: null, data: null, rec: null, ws: null, stream: null, tick: null, sent: 0 };

const page = (n) => {
  document.querySelectorAll(".pg").forEach((p) => p.classList.remove("on"));
  $("#pg-" + n).classList.add("on");
};
const msg = (m) => ($("#msg").textContent = m);
const bail = (m) => { msg(m); console.error("[onetake]", m); };
const clock = (s) => {
  s = Math.max(0, Math.round(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

document.querySelectorAll(".nv").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".nv").forEach((x) => x.classList.toggle("on", x === b));
    page(b.dataset.go);
    if (b.dataset.go === "rec") camera();
  };
});

/* ── mime probe ────────────────────────────────────────────────
   Recording used to silently do nothing because this was hardcoded to
   video/webm — unsupported in Safari, so the MediaRecorder constructor threw,
   onstop never fired, and no request was ever made. */
const MIMES = [
  ["video/webm;codecs=vp9,opus", "webm"],
  ["video/webm;codecs=vp8,opus", "webm"],
  ["video/webm", "webm"],
  ["video/mp4;codecs=avc1,mp4a.40.2", "mp4"],
  ["video/mp4", "mp4"],
];
const pickMime = () =>
  typeof MediaRecorder === "undefined"
    ? null
    : MIMES.map(([type, ext]) => ({ type, ext })).find((m) => MediaRecorder.isTypeSupported(m.type)) || null;

/* ── 1 · record ────────────────────────────────────────────── */
async function camera() {
  if (S.stream) return;
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    $("#preview").srcObject = S.stream;
    $("#frame-idle").style.display = "none";
    msg("camera ready");
  } catch {
    bail("camera or mic blocked — allow it in browser settings");
  }
}
camera();

$("#go").onclick = async () => {
  await camera();
  if (!S.stream) return;

  const mime = pickMime();
  if (!mime) return bail("this browser can't record — try Chrome");

  S.job = Math.random().toString(16).slice(2, 14);
  S.sent = 0;
  $("#f-job").textContent = S.job;

  // Stream to disk while recording (ViniClip's trick) so the file is already
  // whole the instant you hit stop, and only analysis is left to do.
  const proto = location.protocol === "https:" ? "wss" : "ws";
  S.ws = new WebSocket(`${proto}://${location.host}/ws/record/${S.job}?ext=${mime.ext}&model=base.en`);
  S.ws.binaryType = "arraybuffer";
  const live = await new Promise((res) => {
    S.ws.onopen = () => res(true);
    S.ws.onerror = () => res(false);
    setTimeout(() => res(false), 3000);
  });

  const held = [];
  try {
    S.rec = new MediaRecorder(S.stream, { mimeType: mime.type });
  } catch (e) {
    return bail("recorder wouldn't start: " + e.message);
  }

  S.rec.ondataavailable = (e) => {
    if (!e.data.size) return;
    held.push(e.data);                 // always keep a copy — the socket can drop mid-take
    if (live && S.ws.readyState === 1) {
      e.data.arrayBuffer().then((b) => {
        S.ws.send(b);
        S.sent += b.byteLength;
        $("#sent").textContent = (S.sent / 1048576).toFixed(1) + " mb";
      });
    }
  };
  S.rec.onerror = (e) => bail("recording error: " + (e.error?.name || "unknown"));
  S.rec.onstop = () => {
    const whole = new Blob(held, { type: mime.type });
    console.log("[onetake] stopped:", held.length, "chunks,", whole.size, "bytes, ws live:", live);

    if (live && S.ws.readyState === 1) {
      setTimeout(() => S.ws.close(), 250);
      page("work");
      meter("preparing", 0.02);
      poll();
    } else if (whole.size) {
      msg("socket dropped — uploading the file instead");
      send(new File([whole], "take." + mime.ext, { type: mime.type }));
    } else {
      bail("nothing was recorded — check camera permissions and try again");
    }
  };

  S.rec.start(1000);
  const t0 = Date.now();
  S.tick = setInterval(() => ($("#rt").textContent = clock((Date.now() - t0) / 1000)), 400);
  $("#dot").classList.add("live");
  $("#go").disabled = true;
  $("#stop").disabled = false;
  msg(live ? "recording — streaming live" : "recording — buffering locally");
};

$("#stop").onclick = () => {
  if (!S.rec || S.rec.state === "inactive") return;
  S.rec.stop();
  clearInterval(S.tick);
  $("#dot").classList.remove("live");
  $("#go").disabled = false;
  $("#stop").disabled = true;
  S.stream?.getTracks().forEach((t) => t.stop());
  S.stream = null;
};

/* ── 2 · upload ────────────────────────────────────────────── */
$("#zone").onclick = () => $("#file").click();
$("#file").onchange = (e) => e.target.files[0] && send(e.target.files[0]);

async function send(file) {
  const fd = new FormData();
  fd.append("file", file, file.name || "take.webm");
  fd.append("model", $("#model").value);
  fd.append("fillers", $("#o-filler").checked);
  fd.append("stutters", $("#o-stutter").checked);
  fd.append("dead_air", $("#o-dead").checked);
  fd.append("retakes", $("#o-retake").checked);

  page("work");
  meter("uploading", 0.02);
  msg("uploading " + file.name);

  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const raw = await r.text();
    let j = {};
    try { j = JSON.parse(raw); } catch { /* server sent a stack trace, not json */ }
    if (!r.ok) throw new Error(j.detail || raw.slice(0, 180) || `server said ${r.status}`);
    S.job = j.job_id;
    $("#f-job").textContent = S.job;
    poll();
  } catch (e) {
    page("up");
    bail("upload failed: " + e.message);
    alert("onetake couldn't process that take:\n\n" + e.message);
  }
}

function meter(stage, f) {
  $("#w-st").textContent = stage;
  $("#w-fl").style.width = f * 100 + "%";
  $("#w-pc").textContent = Math.round(f * 100) + "%";
}

async function poll() {
  try {
    const j = await (await fetch(`/api/jobs/${S.job}`)).json();
    if (j.stage === "error") { page("up"); bail(j.error || "processing failed"); return alert("onetake failed:\n\n" + (j.error || "unknown")); }
    meter(j.stage || "working", j.progress || 0);
    msg(j.stage || "working");
    if (j.ready && j.result) { S.data = j.result; result(); return; }
  } catch { /* mid-render, retry */ }
  setTimeout(poll, 700);
}

/* ── 3 · result ────────────────────────────────────────────── */
function result() {
  page("out");
  msg("done");
  $("#out").src = `/api/result/${S.job}?t=${Date.now()}`;
  $("#dl").href = `/api/result/${S.job}`;
  $("#edl").href = `/api/edl/${S.job}`;
  paint();
}

function paint() {
  const s = S.data.stats;
  $("#f-in").textContent = clock(s.original_seconds);
  $("#f-out").textContent = clock(s.final_seconds);
  $("#f-gone").textContent = "−" + clock(s.removed_seconds);
  $("#f-n").textContent = s.cut_count;

  const reel = $("#reel");
  reel.innerHTML = "";
  S.data.cuts.forEach((c) => {
    const m = document.createElement("div");
    m.className = "mk " + c.kind;
    m.style.left = (c.start / S.data.duration) * 100 + "%";
    m.style.width = Math.max((c.dur / S.data.duration) * 100, 0.4) + "%";
    m.title = `${c.kind.replace("_", " ")} · ${c.dur.toFixed(2)}s\n${c.text || ""}`;
    m.onclick = () => { $("#out").currentTime = Math.max(0, c.start - 1); $("#out").play(); };
    reel.appendChild(m);
  });

  const box = $("#script");
  box.innerHTML = "";
  S.data.transcript.forEach((w) => {
    const mid = (w.start + w.end) / 2;
    const hit = S.data.cuts.find((c) => mid >= c.start && mid <= c.end);
    const el = document.createElement("w");
    el.textContent = w.text + " ";
    if (hit) { el.className = "gone"; el.title = "cut as " + hit.kind.replace("_", " "); }
    box.appendChild(el);
  });
}

$("#apply").onclick = async () => {
  const text = $("#fb").value.trim();
  if (!text) return;
  const b = $("#apply");
  b.disabled = true;
  b.textContent = "applying…";
  msg("re-cutting");

  const fd = new FormData();
  fd.append("text", text);
  try {
    const r = await fetch(`/api/feedback/${S.job}`, { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "that didn't work");

    if (!j.ok) {
      $("#fb-out").textContent = j.message;
      $("#fb-out").className = "ask-out miss";
    } else {
      S.data = j.result;
      paint();
      $("#out").src = `/api/result/${S.job}?t=${Date.now()}`;
      $("#fb-out").textContent = "→ " + j.notes.join(", ");
      $("#fb-out").className = "ask-out hit";
      $("#fb").value = "";
      msg("done");
    }
  } catch (e) {
    bail(e.message);
  } finally {
    b.disabled = false;
    b.textContent = "apply";
  }
};

$("#again").onclick = () => {
  S.job = S.data = null;
  $("#fb").value = "";
  $("#fb-out").textContent = "";
  document.querySelectorAll(".nv").forEach((x) => x.classList.toggle("on", x.dataset.go === "rec"));
  page("rec");
  camera();
};