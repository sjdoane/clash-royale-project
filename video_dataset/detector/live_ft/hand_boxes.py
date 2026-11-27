#!/usr/bin/env python3
"""Hand-drawn boxes for units the detector misses: a one-page local tool.

    ./draw_boxes.sh                      (= .venv/bin/python video_dataset/detector/live_ft/hand_boxes.py)
        [--dir runs/det_live/hand_v1] [--port 8765] [--no-open]

Serves <dir> (tasks.json, img/, ref/ from hand_pick.py) on 127.0.0.1 only and opens the page. Every answer is
written at once to <dir>/answers.json ({task id: {boxes [[x0, y0, x1, y1] in live-frame pixels], ms, at}}), so the
tab can be closed at any time and the work resumes where it stopped. Standard library only. Ctrl-C stops it.
The frames can show an opponent's name in places: local use only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOCK = threading.Lock()
SAFE = re.compile(r"^(img|ref)/[A-Za-z0-9_.-]+\.jpg$")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Box the missed units</title>
<style>
:root{--bg:#14161a;--panel:#1d2026;--line:#2c313a;--text:#eceef2;--dim:#9aa1ad;--ok:#3ddc84;--ours:#4c9bff;--enemy:#ff5d5d;--hint:#ffd84a}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}
#top{height:56px;display:flex;align-items:center;gap:18px;padding:0 20px;border-bottom:1px solid var(--line)}
#top h1{font-size:16px;margin:0;font-weight:650;white-space:nowrap}
#bar{flex:1;height:10px;background:var(--panel);border-radius:5px;overflow:hidden;border:1px solid var(--line)}
#fill{height:100%;width:0;background:var(--ok);transition:width .2s}
#count{font-variant-numeric:tabular-nums;white-space:nowrap}
#count b{font-size:17px}
#saved{color:var(--dim);width:92px;text-align:right;white-space:nowrap}
#main{display:flex;gap:20px;padding:16px 20px;height:calc(100% - 56px);justify-content:center}
#stage{position:relative;flex:none}
canvas{display:block;border-radius:8px;cursor:crosshair;background:#000;touch-action:none}
#side{width:320px;flex:none;display:flex;flex-direction:column;gap:14px;min-height:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
#who{display:inline-block;font-size:12px;font-weight:700;letter-spacing:.08em;padding:2px 9px;border-radius:999px;color:#0b0d10}
#name{font-size:26px;font-weight:700;line-height:1.15;margin:8px 0 4px}
#kind{color:var(--dim)}
#refrow{display:flex;gap:12px;align-items:flex-start}
#ref{width:150px;height:150px;object-fit:contain;border-radius:6px;background:#000;flex:none}
#mini{height:150px;border-radius:6px;flex:none}
.cap{color:var(--dim);font-size:12.5px;margin-top:8px}
ol{margin:0;padding-left:20px}
li{margin:3px 0}
kbd{display:inline-block;min-width:22px;text-align:center;padding:1px 7px;border:1px solid #4a505c;border-bottom-width:2px;border-radius:5px;background:#262a32;font:12.5px ui-monospace,Menlo,monospace}
#btns{display:flex;gap:8px;margin-top:auto}
button{flex:1;padding:11px 8px;border-radius:8px;border:1px solid var(--line);background:#262a32;color:var(--text);font:inherit;cursor:pointer}
button:hover{background:#30353f}
#next{flex:2;background:var(--ok);color:#06210f;border-color:var(--ok);font-weight:700}
#next:hover{filter:brightness(1.08);background:var(--ok)}
#over{position:fixed;inset:0;background:rgba(10,11,14,.86);display:none;align-items:center;justify-content:center;z-index:5}
#over .card{max-width:440px;text-align:center;padding:28px}
#over h2{margin:0 0 8px;font-size:22px}
#over button{margin-top:16px;flex:none;padding:11px 22px}
#toast{position:absolute;left:50%;top:14px;transform:translateX(-50%);background:rgba(0,0,0,.72);padding:5px 12px;border-radius:999px;font-size:13px;opacity:0;transition:opacity .15s;pointer-events:none}
</style></head><body>
<div id="top"><h1>Box the missed units</h1><div id="bar"><div id="fill"></div></div>
<div id="count"></div><div id="saved"></div></div>
<div id="main">
 <div id="stage"><canvas id="cv"></canvas><div id="toast"></div></div>
 <div id="side">
  <div class="card"><span id="who"></span><div id="name"></div><div id="kind"></div></div>
  <div class="card"><div id="refrow"><img id="ref" alt=""><canvas id="mini"></canvas></div>
   <div class="cap"><b style="color:var(--ok)">Green</b> = the same unit in a nearby frame, boxed the way yours should be. Right: where you are on the screen.</div></div>
  <div class="card"><ol>
   <li><b>Drag a tight box</b> around the unit inside the <b style="color:var(--hint)">yellow corners</b>. Body only: no health bar, no level number.</li>
   <li><kbd>Space</kbd> next. Wrong box? Just drag again.</li>
   <li>Not there, cannot tell, or it is a different card? <kbd>Space</kbd> without a box.</li>
  </ol><div class="cap"><kbd>Z</kbd> back &nbsp; <kbd>X</kbd> remove box &nbsp; <kbd>H</kbd> hide the yellow corners. Thin white boxes are already done.</div></div>
  <div id="btns"><button id="back">Back</button><button id="clear">Remove box</button><button id="next">Next</button></div>
 </div>
</div>
<div id="over"><div class="card"><h2 id="oh"></h2><div id="ob"></div><button id="ok"></button></div></div>
<script>
"use strict";
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d"), mini = $("mini"), mctx = mini.getContext("2d");
const KIND = {deploy: "It has just been played and is dropping in.", gap: "The detector lost it for a moment.",
              bodies: n => "A group of " + n + ". Box each one that has no box yet."};
let W = 549, H = 976, tasks = [], goal = 0, ans = {}, i = 0, boxes = [], view = null, img = new Image(), drag = null,
    mouse = null, t0 = 0, hint = true, CS = 600, goalShown = false, times = [];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function size() {
  CS = Math.max(360, Math.floor(Math.min(window.innerHeight - 56 - 32, window.innerWidth - 320 - 60)));
  const d = window.devicePixelRatio || 1;
  cv.style.width = cv.style.height = CS + "px"; cv.width = cv.height = Math.round(CS * d);
  ctx.setTransform(d, 0, 0, d, 0, 0);
  const mh = 150, mw = Math.round(mh * W / H);
  mini.style.width = mw + "px"; mini.style.height = mh + "px"; mini.width = mw * d; mini.height = mh * d;
  mctx.setTransform(d, 0, 0, d, 0, 0);
  draw();
}
function region(t) { const b = t.box, gw = (b[2] - b[0]) * t.grow, gh = (b[3] - b[1]) * t.grow;
  return [b[0] - gw, b[1] - gh, b[2] + gw, b[3] + gh]; }
function done() { return Object.keys(ans).length; }
function head() {
  const n = done(), tot = n >= goal ? tasks.length : goal;
  $("fill").style.width = (100 * Math.min(n / tot, 1)) + "%";
  let left = "";
  if (times.length >= 3 && n < tot) { const s = times.slice(-15).reduce((a, b) => a + b, 0) / Math.min(times.length, 15);
    left = " · about " + Math.max(1, Math.round(s * (tot - n) / 60000)) + " min left"; }
  $("count").innerHTML = "<b>" + n + "</b> of " + tot + left;
}
function load(k) {
  i = clamp(k, 0, tasks.length - 1);
  const t = tasks[i];
  boxes = (ans[t.id] ? ans[t.id].boxes : []).map(b => b.slice());
  const r = region(t), S = clamp(Math.max(r[2] - r[0], r[3] - r[1]) * 2.3, 230, 460);
  view = {s: S, x: clamp((r[0] + r[2]) / 2 - S / 2, 0, W - S), y: clamp((r[1] + r[3]) / 2 - S / 2, 0, H - S)};
  $("who").textContent = t.side ? "ENEMY" : "YOURS";
  $("who").style.background = t.side ? "var(--enemy)" : "var(--ours)";
  $("name").textContent = t.name;
  $("kind").textContent = t.kind === "bodies" ? KIND.bodies(t.n_body) : KIND[t.kind];
  $("ref").src = "ref/" + t.id + ".jpg";
  img = new Image(); img.onload = draw; img.src = "img/" + t.stem + ".jpg";
  if (tasks[i + 1]) { new Image().src = "img/" + tasks[i + 1].stem + ".jpg"; new Image().src = "ref/" + tasks[i + 1].id + ".jpg"; }
  t0 = performance.now(); drag = null; head(); draw();
}
const fx = x => (x - view.x) * CS / view.s, fy = y => (y - view.y) * CS / view.s;
function rect(b, color, lw, dash) { ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash || []);
  ctx.strokeRect(fx(b[0]), fy(b[1]), (b[2] - b[0]) * CS / view.s, (b[3] - b[1]) * CS / view.s); ctx.restore(); }
function corners(b) { const x0 = fx(b[0]), y0 = fy(b[1]), x1 = fx(b[2]), y1 = fy(b[3]), L = Math.min(26, (x1 - x0) / 3, (y1 - y0) / 3);
  ctx.save(); ctx.strokeStyle = "#ffd84a"; ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.beginPath();
  for (const [x, y, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
    ctx.moveTo(x + dx * L, y); ctx.lineTo(x, y); ctx.lineTo(x, y + dy * L); }
  ctx.stroke(); ctx.restore(); }
function draw() {
  if (!view) return;
  const t = tasks[i];
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, CS, CS);
  if (img.complete && img.naturalWidth) { ctx.imageSmoothingQuality = "high"; ctx.drawImage(img, view.x, view.y, view.s, view.s, 0, 0, CS, CS); }
  for (const l of t.labels) rect(l[0], "rgba(255,255,255,.55)", 1, [4, 4]);
  if (hint) corners(region(t));
  for (const b of boxes) rect(b, "#3ddc84", 2.5);
  if (drag) rect([Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)], "#fff", 2);
  else if (mouse) { ctx.save(); ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(mouse.x, 0); ctx.lineTo(mouse.x, CS); ctx.moveTo(0, mouse.y); ctx.lineTo(CS, mouse.y); ctx.stroke(); ctx.restore(); }
  const mw = mini.clientWidth, mh = mini.clientHeight, k = mh / H;
  mctx.clearRect(0, 0, mw, mh);
  if (img.complete && img.naturalWidth) mctx.drawImage(img, 0, 0, mw, mh);
  mctx.strokeStyle = "#ffd84a"; mctx.lineWidth = 2; mctx.strokeRect(view.x * k, view.y * k, view.s * k, view.s * k);
}
function pos(e) { const r = cv.getBoundingClientRect(); return {x: clamp(e.clientX - r.left, 0, CS), y: clamp(e.clientY - r.top, 0, CS)}; }
cv.addEventListener("pointerdown", e => { if (e.button) return; const p = pos(e); cv.setPointerCapture(e.pointerId);
  drag = {x0: view.x + p.x * view.s / CS, y0: view.y + p.y * view.s / CS}; drag.x1 = drag.x0; drag.y1 = drag.y0; draw(); });
cv.addEventListener("pointermove", e => { mouse = pos(e);
  if (drag) { drag.x1 = view.x + mouse.x * view.s / CS; drag.y1 = view.y + mouse.y * view.s / CS; } draw(); });
cv.addEventListener("pointerleave", () => { mouse = null; draw(); });
cv.addEventListener("pointerup", () => { if (!drag) return;
  const b = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)].map(v => Math.round(v * 10) / 10);
  drag = null;
  if (b[2] - b[0] >= 4 && b[3] - b[1] >= 4) { if (tasks[i].kind !== "bodies") boxes = []; boxes.push(b); }
  draw(); });
function toast(s) { const t = $("toast"); t.textContent = s; t.style.opacity = 1; clearTimeout(toast.h); toast.h = setTimeout(() => t.style.opacity = 0, 900); }
async function next() {
  const t = tasks[i], fresh = !ans[t.id], ms = Math.round(performance.now() - t0);
  ans[t.id] = {boxes: boxes.map(b => b.slice()), ms};
  if (fresh) times.push(Math.min(ms, 60000));
  $("saved").textContent = "Saving…";
  try { const r = await fetch("save", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: t.id, boxes, ms})});
    if (!r.ok) throw new Error(r.status); $("saved").textContent = "Saved ✓"; $("saved").style.color = "var(--dim)"; }
  catch (err) { delete ans[t.id]; $("saved").textContent = "NOT SAVED"; $("saved").style.color = "var(--enemy)";
    over("Not saved", "The tool in the terminal has stopped. Start <b>./draw_boxes.sh</b> again and reload this page. Nothing saved before is lost.", "OK", () => {}); return; }
  toast(boxes.length ? "Saved " + boxes.length + (boxes.length > 1 ? " boxes" : " box") : "Skipped");
  const n = done();
  if (n >= tasks.length) return over("All done", n + " units answered. You can close this tab and tell Claude “boxes done”.", "Review", () => load(i));
  if (n >= goal && !goalShown) { goalShown = true;
    return over("Goal reached", "That is the " + goal + " I asked for. Everything is saved. Close this tab and tell Claude “boxes done”, or keep going: every extra box helps (" + (tasks.length - n) + " more ready).", "Keep going", go); }
  go();
  function go() { let k = i + 1; while (k < tasks.length && ans[tasks[k].id]) k++; if (k >= tasks.length) k = tasks.findIndex(x => !ans[x.id]); load(k < 0 ? i : k); }
}
function over(h, b, btn, fn) { $("oh").textContent = h; $("ob").innerHTML = b; $("ok").textContent = btn; $("over").style.display = "flex";
  $("ok").onclick = () => { $("over").style.display = "none"; fn(); }; }
function back() { if (i > 0) load(i - 1); }
function clearBox() { boxes.pop(); draw(); }
$("next").onclick = next; $("back").onclick = back; $("clear").onclick = clearBox;
window.addEventListener("keydown", e => {
  if ($("over").style.display === "flex") { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("ok").click(); } return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === " " || k === "enter") { e.preventDefault(); if (!e.repeat) next(); }
  else if (k === "z") back();
  else if (k === "x" || k === "backspace") { e.preventDefault(); clearBox(); }
  else if (k === "h") { hint = !hint; draw(); }
});
window.addEventListener("resize", size);
fetch("state").then(r => r.json()).then(s => {
  tasks = s.tasks; goal = s.goal; ans = s.answers; W = s.size[0]; H = s.size[1]; goalShown = done() >= goal;
  size(); const k = tasks.findIndex(t => !ans[t.id]); load(k < 0 ? 0 : k);
  if (!done()) over("Box the missed units", "Each screen shows one unit the detector missed, and names it for you. Drag a tight box around it, then press <b>Space</b>. " +
    "If it is not there, or it is not the card named, press <b>Space</b> without drawing.<br><br>The goal is " + goal + " units, about 10 minutes. Everything saves as you go, so you can stop at any time.", "Start", () => { t0 = performance.now(); });
});
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "runs" / "det_live" / "hand_v1"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    tasks = json.load(open(d / "tasks.json"))
    ids = {t["id"] for t in tasks["tasks"]}
    apath = d / "answers.json"
    answers = json.load(open(apath)) if apath.exists() else {}

    class Hd(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith(("text", "application")) else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0].lstrip("/")
            if p == "":
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if p == "state":
                with LOCK:
                    return self.send(200, json.dumps(tasks | {"answers": answers}).encode(), "application/json")
            if SAFE.match(p) and (d / p).is_file():
                return self.send(200, (d / p).read_bytes(), "image/jpeg")
            self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path.lstrip("/") != "save":
                return self.send(404, b"not found", "text/plain")
            try:
                r = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                assert r["id"] in ids
                bx = [[round(float(v), 1) for v in b] for b in r["boxes"]]
                assert all(len(b) == 4 and b[2] > b[0] and b[3] > b[1] for b in bx)
            except Exception as e:  # noqa: BLE001
                return self.send(400, str(e).encode(), "text/plain")
            with LOCK:
                answers[r["id"]] = {"boxes": bx, "ms": int(r.get("ms", 0)), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                tmp = apath.with_suffix(".tmp")
                tmp.write_text(json.dumps(answers, indent=0))
                os.replace(tmp, apath)
            self.send(200, b"{}", "application/json")

    port = a.port
    for port in range(a.port, a.port + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break
    srv = ThreadingHTTPServer(("127.0.0.1", port), Hd)
    url = f"http://127.0.0.1:{port}/"
    print(f"Box tool: {url}   ({len(answers)} of {len(ids)} answered so far, goal {tasks['goal']})", flush=True)
    print(f"Answers are saved to {apath} after every unit. Press Ctrl-C here when you are done.", flush=True)
    if not a.no_open:
        threading.Timer(0.4, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. {len(answers)} answered, {sum(bool(v['boxes']) for v in answers.values())} with a box.")


if __name__ == "__main__":
    sys.exit(main())
