/* onetake — record or upload, get a cut back, ask for changes. */

const $ = (s) => document.querySelector(s);
const S = { job: null, data: null, rec: null, ws: null, stream: null, tick: null };

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

/* Recording used to silently do nothing because this was hardcoded to
   video/webm — unsupported in Safari, so the constructor threw and onstop
   never fired. Probe instead. */
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

/* ── record ── */
async function camera() {
  if (S.stream) return;
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 },
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    $("#preview").srcObject = S.stream;
    $("#idle").style.display = "none";
    msg("camera ready");
  } catch {
    bail("camera or mic blocked — allow it in browser settings");
  }
}
camera();

let recording = false;

$("#btn").onclick = async () => {
  if (recording) return stop();

  await camera();
  if (!S.stream) return;
  const mime = pickMime();
  if (!mime) return bail("this browser can't record — try Chrome");

  S.job = Math.random().toString(16).slice(2, 14);

  // Stream to disk while recording, so the file is already whole at stop.
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
    held.push(e.data);                 // always keep a copy — the socket can drop
    if (live && S.ws.readyState === 1) e.data.arrayBuffer().then((b) => S.ws.send(b));
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
      send(new File([whole], "take." + mime.ext, { type: mime.type }));
    } else {
      bail("nothing was recorded — check camera permissions");
    }
  };

  S.rec.start(1000);
  recording = true;
  const t0 = Date.now();
  S.tick = setInterval(() => ($("#rt").textContent = clock((Date.now() - t0) / 1000)), 400);
  $("#dot").classList.add("live");
  $("#btn").textContent = "stop & cut";
  msg(live ? "recording — streaming live" : "recording — buffering locally");
};

function stop() {
  if (!S.rec || S.rec.state === "inactive") return;
  S.rec.stop();
  clearInterval(S.tick);
  recording = false;
  $("#dot").classList.remove("live");
  $("#btn").textContent = "start recording";
  S.stream?.getTracks().forEach((t) => t.stop());
  S.stream = null;
}

/* ── upload ── */
$("#zone").onclick = () => $("#file").click();
$("#file").onchange = (e) => e.target.files[0] && send(e.target.files[0]);

async function send(file) {
  const fd = new FormData();
  fd.append("file", file, file.name || "take.webm");
  page("work");
  meter("uploading", 0.02);
  msg("uploading");

  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const raw = await r.text();
    let j = {};
    try { j = JSON.parse(raw); } catch { /* server sent a stack trace */ }
    if (!r.ok) throw new Error(j.detail || raw.slice(0, 180) || `server said ${r.status}`);
    S.job = j.job_id;
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
    if (j.stage === "error") {
      page("up");
      bail(j.error || "processing failed");
      return alert("onetake failed:\n\n" + (j.error || "unknown"));
    }
    meter(j.stage || "working", j.progress || 0);
    msg(j.stage || "working");
    if (j.ready && j.result) { S.data = j.result; result(); return; }
  } catch { /* mid-render, retry */ }
  setTimeout(poll, 700);
}

/* ── result ── */
function result() {
  page("out");
  msg("done");
  $("#out").src = `/api/result/${S.job}?t=${Date.now()}`;
  $("#dl").href = `/api/result/${S.job}`;
  paint();
}

function paint() {
  const s = S.data.stats;
  $("#stat").innerHTML =
    `${clock(s.original_seconds)} → <b>${clock(s.final_seconds)}</b>` +
    ` · ${s.cut_count} cuts · ${Math.round(s.percent_removed)}% removed`;

  const reel = $("#reel");
  reel.innerHTML = "";
  S.data.cuts.forEach((c) => {
    const m = document.createElement("div");
    m.className = "mk " + c.kind;
    m.style.left = (c.start / S.data.duration) * 100 + "%";
    m.style.width = Math.max((c.dur / S.data.duration) * 100, 0.4) + "%";
    reel.appendChild(m);
  });
}

$("#fb").onkeydown = (e) => { if (e.key === "Enter") $("#apply").click(); };

$("#apply").onclick = async () => {
  const text = $("#fb").value.trim();
  if (!text) return;
  const b = $("#apply");
  b.disabled = true;
  b.textContent = "…";
  msg("re-cutting");

  const fd = new FormData();
  fd.append("text", text);
  try {
    const r = await fetch(`/api/feedback/${S.job}`, { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "that didn't work");

    if (!j.ok) {
      $("#note").textContent = j.message;
      $("#note").className = "note miss";
    } else {
      S.data = j.result;
      paint();
      $("#out").src = `/api/result/${S.job}?t=${Date.now()}`;
      $("#note").textContent = "→ " + j.notes.join(", ");
      $("#note").className = "note hit";
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
  $("#note").textContent = "";
  document.querySelectorAll(".nv").forEach((x) => x.classList.toggle("on", x.dataset.go === "rec"));
  page("rec");
  camera();
};