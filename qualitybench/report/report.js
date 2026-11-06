const D = /*__DATA__*/null;
const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const fmt = (v, d = 1) => (v == null || isNaN(v)) ? "–" : Number(v).toFixed(d);
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sign = v => (v >= 0 ? "+" : "") + fmt(v, 2);
const pfmt = p => p < 1e-3 ? p.toExponential(1) : fmt(p, 3);
function el(tag, at = {}, parent) { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); if (parent) parent.appendChild(e); return e; }
function txt(parent, x, y, s, at = {}) { const t = el("text", { x, y, ...at }, parent); t.textContent = s; return t; }
function svgIn(node, w, h) { node.innerHTML = ""; return el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", role: "img" }, node); }
const short = (s, w) => w < 560 ? s.replace("LiveCodeBench", "LCB") : s;
const Wn = n => Math.max(300, Math.min(1040, n.clientWidth || 600));
const W = id => Wn($(id));
const hide = id => { const s = $(id); if (s) s.style.display = "none"; };
function table(id, head, rows, numCols = []) {
  const h = head.map((x, i) => `<th class="${numCols.includes(i) ? "num" : ""}">${x}</th>`).join("");
  const b = rows.map(r => "<tr>" + r.map((x, i) => `<td class="${numCols.includes(i) ? "num" : ""}">${x}</td>`).join("") + "</tr>").join("");
  const html = `<table><thead><tr>${h}</tr></thead><tbody>${b}</tbody></table>`;
  if (typeof id === "string") $(id).innerHTML = html; else return html;
}
const tip = $("#tip");
document.addEventListener("pointermove", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (!t) { tip.style.display = "none"; return; }
  tip.innerHTML = t.getAttribute("data-tip"); tip.style.display = "block";
  const r = tip.getBoundingClientRect(); let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14; if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
});
function status(kind, label) {
  const c = { good: "var(--good)", crit: "var(--crit)", warn: "var(--warn)", wait: "var(--axis)" }[kind];
  const icon = { good: "✓", crit: "✗", warn: "!", wait: "…" }[kind];
  return `<span class="status"><span class="dot" style="background:${c}"></span>${icon} ${esc(label)}</span>`;
}
function xAxis(g, y, ticks, sx, f = v => v) {
  for (const v of ticks) { const X = sx(v); el("line", { x1: X, x2: X, y1: 0, y2: y, stroke: "var(--grid)", "stroke-width": 1 }, g); txt(g, X, y + 16, f(v), { class: "tick", "text-anchor": "middle" }); }
}
const ARMS = Object.entries(D.arms || {});
const ARMC = ["var(--s1)", "var(--s2)", "var(--s3)"];
const LP = D.logprobs || {};
const SETS = [["ppl-wiki", "var(--s1)"], ["ppl-code", "var(--s2)"], ["kld", "var(--s3)"]].filter(([k]) => LP[k]);
const THINK = D.tasks.some(t => t.on);

// ---------- header + tiles
function header() {
  $("#title").textContent = D.title;
  const e = D.env || {};
  const where = [e.precision, e.engine && (e.gpu ? `${e.engine} on ${e.gpu}` : e.engine) || e.gpu].filter(Boolean).join(", ");
  $("#subtitle").textContent = [D.model, where, "greedy, seed 608"].filter(Boolean).join(" · ");
  const chips = [`QualityBench ${D.qualitybench_version || ""}`.trim(), e.lm_eval && `lm-eval ${e.lm_eval}`, e.evalplus && `evalplus ${e.evalplus}`,
    e.bfcl_eval && `BFCL ${e.bfcl_eval}`, D.generated && `generated ${D.generated}`].filter(Boolean);
  $("#chips").innerHTML = chips.map((c, i) => `<span class="chip ${i ? "" : "proto"}">${esc(c)}</span>`).join("");
  const nd = D.needle ? D.needle.items : [];
  const tiles = [[`${D.tasks.length}`, "tasks scored", THINK ? "thinking off and on" : "thinking off is the reference"]];
  if (D.tasks.length) {
    const best = D.tasks.reduce((a, t) => a + t.off.acc, 0) / D.tasks.length;
    tiles.push([fmt(best) + "%", "mean task accuracy", "unweighted, over the tasks above"]);
  }
  if (LP["ppl-wiki"]) tiles.push([fmt(LP["ppl-wiki"].ppl, 3), "perplexity, wikitext-2", `${LP["ppl-wiki"].windows} × ${LP["ppl-wiki"].window_tokens || LP["ppl-wiki"].positions + 1} tokens`]);
  if (LP["ppl-code"]) tiles.push([fmt(LP["ppl-code"].ppl, 3), "perplexity, code", `${LP["ppl-code"].windows} × ${LP["ppl-code"].window_tokens || LP["ppl-code"].positions + 1} tokens`]);
  if (nd.length) tiles.push([`${nd.filter(x => x.correct).length}/${nd.length}`, "needle retrievals", [...new Set(nd.map(x => x.length / 1024 + "k"))].join(" and ") + " context"]);
  if (LP.kld) tiles.push([fmt(100 * LP.kld.mass_mean, 1) + "%", `probability in top-${(D.eval_meta || {}).topk || 64}`, `KLD set mean; 1st pct ${fmt(100 * LP.kld.mass_p01, 1)}%`]);
  for (const [k, a] of ARMS) if (a.kld) tiles.push([fmt(a.kld.mean, 4), `mean KLD, ${a.label}`, a.top1 ? `top-1 agreement ${fmt(a.top1.agree, 2)}%` : "nats"]);
  $("#tiles").innerHTML = tiles.map(t => `<div class="tile"><div class="v">${esc(t[0])}</div><div class="l">${esc(t[1])}</div><div class="n">${esc(t[2])}</div></div>`).join("");
}

// ---------- task scoreboard: grouped horizontal bars + Wilson whiskers
function tasksChart() {
  if (!D.tasks.length) { hide("#s-tasks"); return; }
  const rows = THINK ? [["off", "var(--s1)", 4], ["on", "var(--s2)", 22]] : [["off", "var(--s1)", 8]];
  if (!THINK) $("#s-tasks .legend").style.display = "none";
  const w = W("#c-tasks"), L = Math.min(150, w * .3), R = 56, rowH = THINK ? 46 : 32, top = 6, h = top + D.tasks.length * rowH + 24;
  const svg = svgIn($("#c-tasks"), w, h), sx = v => L + (w - L - R) * v / 100;
  const g = el("g", {}, svg);
  xAxis(g, h - 22, [0, 20, 40, 60, 80, 100], sx, v => v + "%");
  D.tasks.forEach((t, i) => {
    const y = top + i * rowH;
    txt(g, L - 10, y + (THINK ? 22 : 20), short(t.label, w), { class: "lbl", "text-anchor": "end" });
    rows.forEach(([k, c, dy]) => {
      const r = t[k];
      if (!r) { if (k === "on") txt(g, sx(0) + 4, y + dy + 12, "thinking on: not run", { class: "tick" }); return; }
      const x0 = sx(0), x1 = sx(r.acc), bh = 14;
      const tipS = `<b>${esc(t.label)}</b>${THINK ? " · thinking " + k : ""}<br>${r.k}/${r.n} = ${fmt(r.acc)}%<br>95% CI ${fmt(r.ci[0])}–${fmt(r.ci[1])}%`;
      el("path", { d: `M${x0},${y + dy}H${Math.max(x0, x1 - 4)}a4,4 0 0 1 4,4v${bh - 8}a4,4 0 0 1 -4,4H${x0}Z`, fill: c, "data-tip": tipS }, g);
      el("line", { x1: sx(r.ci[0]), x2: sx(r.ci[1]), y1: y + dy + bh / 2, y2: y + dy + bh / 2, stroke: "var(--ink)", "stroke-width": 1.5, opacity: .55 }, g);
      txt(g, Math.max(sx(r.ci[1]), x1) + 6, y + dy + 11, fmt(r.acc) + "%", { class: "val" });
      el("rect", { x: x0, y: y + dy - 2, width: Math.max(8, x1 - x0) + 40, height: bh + 4, fill: "transparent", "data-tip": tipS }, g);
    });
  });
  const head = ["Task", "Scored as", THINK ? "Off" : "Accuracy", "95% CI"].concat(THINK ? ["On", "On 95% CI"] : []);
  table("#t-tasks", head, D.tasks.map(t => [esc(t.label), esc(t.how), `${t.off.k}/${t.off.n} (${fmt(t.off.acc)}%)`, `${fmt(t.off.ci[0])}–${fmt(t.off.ci[1])}`]
    .concat(THINK ? [t.on ? `${t.on.k}/${t.on.n} (${fmt(t.on.acc)}%)` : '<span class="pending">not run</span>', t.on ? `${fmt(t.on.ci[0])}–${fmt(t.on.ci[1])}` : "–"] : [])),
    THINK ? [2, 3, 4, 5] : [2, 3]);
}

// ---------- paired deltas (thinking on vs off, or an arm vs the reference) + discordant pairs
function deltaDots(node, rows, color, labA, labB) {
  const rowH = 34, top = 8, w = Wn(node), L = 120, R = 70, h = top + Math.max(1, rows.length) * rowH + 26, svg = svgIn(node, w, h), g = el("g", {}, svg);
  const ext = Math.max(4, ...rows.map(([, q]) => Math.max(Math.abs(q.ci[0]), Math.abs(q.ci[1])))) * 1.15;
  const sx = v => L + (w - L - R) * (v + ext) / (2 * ext);
  const step = ext > 8 ? 4 : 2, ticks = []; for (let v = -Math.floor(ext / step) * step; v <= ext; v += step) ticks.push(v);
  xAxis(g, h - 24, ticks, sx, v => (v > 0 ? "+" : "") + v);
  el("line", { x1: sx(0), x2: sx(0), y1: 0, y2: h - 24, stroke: "var(--axis)", "stroke-width": 1.5 }, g);
  if (D.gates && D.gates.task_delta_pts != null && ARMS.length && labA !== "thinking on") {
    const X = sx(Math.max(-ext, D.gates.task_delta_pts));
    el("line", { x1: X, x2: X, y1: 0, y2: h - 24, stroke: "var(--crit)", "stroke-width": 1, "stroke-dasharray": "4 3", opacity: .7 }, g);
  }
  rows.forEach(([label, q], i) => {
    const y = top + i * rowH + 12, sig = q.ci[0] > 0 || q.ci[1] < 0;
    const tipS = `<b>${esc(label)}</b><br>change ${sign(q.delta)} pts<br>95% CI ${sign(q.ci[0])} to ${sign(q.ci[1])}<br>McNemar p = ${pfmt(q.p)}<br>right only ${esc(labA)}: ${q.test_only} · only ${esc(labB)}: ${q.ref_only}`;
    txt(g, L - 10, y + 4, short(label, w), { class: "lbl", "text-anchor": "end" });
    el("line", { x1: sx(q.ci[0]), x2: sx(q.ci[1]), y1: y, y2: y, stroke: "var(--ink)", "stroke-width": 2, "stroke-linecap": "round", opacity: .7 }, g);
    el("circle", { cx: sx(q.delta), cy: y, r: 5.5, fill: color, stroke: "var(--surface)", "stroke-width": 2, "data-tip": tipS }, g);
    txt(g, w - R + 8, y + 4, `${sign(q.delta)}${sig ? "" : " n.s."}`, { class: "val" });
    el("rect", { x: L, y: y - 12, width: w - L - R, height: 24, fill: "transparent", "data-tip": tipS }, g);
  });
}
function thinkCharts() {
  const P = D.tasks.filter(t => t.paired);
  if (!P.length) { hide("#s-think"); return; }
  deltaDots($("#c-delta"), P.map(t => [t.label, t.paired]), "var(--s2)", "thinking on", "thinking off");
  { // discordant mirrored bars
    const rowH = 34, top = 8, w = W("#c-disc"), L = 120, R = 16, h = top + P.length * rowH + 26, svg = svgIn($("#c-disc"), w, h), g = el("g", {}, svg);
    const mx = Math.max(10, ...P.map(t => Math.max(t.paired.ref_only, t.paired.test_only))) * 1.2;
    const mid = L + (w - L - R) / 2, sx = v => (w - L - R) / 2 * v / mx;
    const st = [5, 10, 20, 25, 50, 100, 200].find(v => mx / v <= 3) || 500, ticks = [];
    for (let v = -Math.floor(mx / st) * st; v <= mx; v += st) ticks.push(v);
    xAxis(g, h - 24, ticks, v => mid + sx(v), v => Math.abs(v));
    el("line", { x1: mid, x2: mid, y1: 0, y2: h - 24, stroke: "var(--axis)", "stroke-width": 1.5 }, g);
    P.forEach((t, i) => {
      const y = top + i * rowH + 4, q = t.paired, bh = 16;
      const tipS = `<b>${esc(t.label)}</b><br>right only with thinking on: ${q.test_only}<br>right only with thinking off: ${q.ref_only}<br>${q.n} items`;
      txt(g, L - 10, y + 12, short(t.label, w), { class: "lbl", "text-anchor": "end" });
      const a = sx(q.ref_only), b = sx(q.test_only);
      if (a > 0) el("path", { d: `M${mid - 1},${y}H${mid - a + 4}a4,4 0 0 0 -4,4v${bh - 8}a4,4 0 0 0 4,4H${mid - 1}Z`, fill: "var(--s1)", "data-tip": tipS }, g);
      if (b > 0) el("path", { d: `M${mid + 1},${y}H${mid + b - 4}a4,4 0 0 1 4,4v${bh - 8}a4,4 0 0 1 -4,4H${mid + 1}Z`, fill: "var(--s2)", "data-tip": tipS }, g);
      txt(g, mid - a - 6, y + 12, q.ref_only, { class: "val", "text-anchor": "end" });
      txt(g, mid + b + 6, y + 12, q.test_only, { class: "val" });
    });
  }
  table("#t-think", ["Task", "Items", "Off", "On", "Change (pts)", "95% CI", "Right only off", "Right only on", "McNemar p"],
    P.map(t => { const q = t.paired; return [esc(t.label), q.n, fmt(t.off.acc) + "%", fmt(t.on.acc) + "%", sign(q.delta), `${sign(q.ci[0])} to ${sign(q.ci[1])}`, q.ref_only, q.test_only, pfmt(q.p)]; }), [1, 2, 3, 4, 6, 7, 8]);
}

// ---------- logprob panels
function lpCharts() {
  if (!SETS.length) { hide("#s-lp"); return; }
  $("#lg-pos").innerHTML = SETS.map(([k, c]) => `<span><i style="background:${c}"></i>${esc(LP[k].label)}</span>`).join("");
  { // mean NLL by position, crosshair tooltip
    const w = W("#c-pos"), L = 40, R = 70, T = 8, h = 230, B = 26, svg = svgIn($("#c-pos"), w, h), g = el("g", {}, svg);
    const all = SETS.flatMap(([k]) => LP[k].nll_by_pos.map(b => b[2])), ymax = Math.max(.5, Math.ceil(Math.max(...all) * 2) / 2), n = LP[SETS[0][0]].positions;
    const sx = p => L + (w - L - R) * p / n, sy = v => T + (h - T - B) * (1 - v / ymax);
    for (let v = 0; v <= ymax + 1e-9; v += ymax > 3 ? 1 : .5) { el("line", { x1: L, x2: w - R, y1: sy(v), y2: sy(v), stroke: "var(--grid)" }, g); txt(g, L - 6, sy(v) + 4, fmt(v, 1), { class: "tick", "text-anchor": "end" }); }
    [0, .25, .5, .75, 1].forEach(f => txt(g, sx(Math.round(f * n)), h - 8, Math.round(f * n), { class: "tick", "text-anchor": "middle" }));
    const ends = [];
    SETS.forEach(([k, c]) => {
      const pts = LP[k].nll_by_pos.map(b => [sx((b[0] + b[1]) / 2), sy(b[2])]);
      el("path", { d: "M" + pts.map(p => p.join(",")).join("L"), fill: "none", stroke: c, "stroke-width": 2, "stroke-linejoin": "round" }, g);
      const last = pts[pts.length - 1]; ends.push({ x: last[0] + 6, y: last[1] + 4, s: LP[k].label });
    });
    ends.sort((a, b) => a.y - b.y).forEach((e, i, A) => { if (i && e.y - A[i - 1].y < 14) e.y = A[i - 1].y + 14; txt(g, e.x, e.y, e.s, { class: "val" }); });
    const cross = el("line", { y1: T, y2: h - B, stroke: "var(--axis)", "stroke-width": 1, opacity: 0 }, g);
    const hit = el("rect", { x: L, y: T, width: w - L - R, height: h - T - B, fill: "transparent" }, g);
    const bins = LP[SETS[0][0]].nll_by_pos;
    hit.addEventListener("pointermove", e => {
      const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * w / r.width, i = Math.max(0, Math.min(bins.length - 1, Math.floor((px - L) / (w - L - R) * bins.length)));
      const b0 = bins[i]; cross.setAttribute("x1", sx((b0[0] + b0[1]) / 2)); cross.setAttribute("x2", sx((b0[0] + b0[1]) / 2)); cross.setAttribute("opacity", 1);
      hit.setAttribute("data-tip", `<b>positions ${b0[0]}–${b0[1] - 1}</b><br>` + SETS.map(([k]) => `${esc(LP[k].label)}: ${fmt(LP[k].nll_by_pos[i][2], 3)} nats`).join("<br>"));
    });
    hit.addEventListener("pointerleave", () => cross.setAttribute("opacity", 0));
  }
  { // per-window perplexity strip plot (log scale)
    const w = W("#c-win"), L = 80, R = 16, rowH = 56, h = 8 + SETS.length * rowH + 26, svg = svgIn($("#c-win"), w, h), g = el("g", {}, svg);
    const all = SETS.flatMap(([k]) => LP[k].win_ppl), lo = Math.log10(Math.min(...all) * .9), hi = Math.log10(Math.max(...all) * 1.1);
    const sx = v => L + (w - L - R) * (Math.log10(v) - lo) / (hi - lo);
    const ticks = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500].filter(v => Math.log10(v) >= lo && Math.log10(v) <= hi);
    xAxis(g, h - 24, ticks, sx, v => v);
    SETS.forEach(([k, c], i) => {
      const y = 8 + i * rowH + rowH / 2 - 6, s = LP[k];
      txt(g, L - 10, y + 4, s.label, { class: "lbl", "text-anchor": "end" });
      let rng = 608; const jit = () => ((rng = (rng * 16807) % 2147483647) / 2147483647 - .5) * 22;
      s.win_ppl.forEach((v, j) => el("circle", { cx: sx(v), cy: y + jit(), r: 3, fill: c, opacity: .75, "data-tip": `<b>${esc(s.label)}</b> window ${j}<br>perplexity ${fmt(v, 3)}` }, g));
      el("line", { x1: sx(s.ppl), x2: sx(s.ppl), y1: y - 16, y2: y + 16, stroke: "var(--ink)", "stroke-width": 2 }, g);
      txt(g, sx(s.ppl) + 5, y - 16, `pooled ${fmt(s.ppl, 2)}`, { class: "val" });
    });
  }
  const k = (D.eval_meta || {}).topk || 64;
  table("#t-lp", ["Set", "Windows × positions", "Perplexity", `Top-${k} mass, mean`, `Top-${k} mass, 1st pct`, "Loss, first bin", "Loss, last bin"],
    SETS.map(([key]) => { const s = LP[key], b = s.nll_by_pos; return [esc(s.label), `${s.windows} × ${s.positions}`, fmt(s.ppl, 4), fmt(100 * s.mass_mean, 2) + "%", fmt(100 * s.mass_p01, 2) + "%", fmt(b[0][2], 3), fmt(b[b.length - 1][2], 3)]; }), [1, 2, 3, 4, 5, 6]);
}

// ---------- needle heatmap (status cells, icon + label)
function needleChart() {
  if (!D.needle || !D.needle.items.length) { hide("#s-needle"); return; }
  const it = D.needle.items, lens = [...new Set(it.map(x => x.length))].sort((a, b) => a - b), deps = [...new Set(it.map(x => x.depth))].sort((a, b) => a - b);
  const w = W("#c-needle"), L = 70, top = 22, ch = 34, h = top + lens.length * ch + 8, svg = svgIn($("#c-needle"), w, h), g = el("g", {}, svg);
  const cw = (w - L) / deps.length;
  deps.forEach((d, j) => txt(g, L + j * cw + cw / 2, 14, Math.round(100 * d) + "%", { class: "tick", "text-anchor": "middle" }));
  lens.forEach((Ln, i) => {
    txt(g, L - 10, top + i * ch + ch / 2 + 4, (Ln / 1024) + "k", { class: "lbl", "text-anchor": "end" });
    deps.forEach((d, j) => {
      const x = it.find(q => q.length === Ln && q.depth === d);
      if (!x) return;
      const ok = x.correct;
      const tipS = `<b>${Ln / 1024}k context, depth ${Math.round(100 * d)}%</b><br>${ok ? "✓ retrieved" : "✗ missed"}${x.prompt_tokens ? `<br>${x.prompt_tokens} prompt tokens` : ""}`;
      el("rect", { x: L + j * cw + 1, y: top + i * ch + 1, width: cw - 2, height: ch - 2, rx: 4, fill: ok ? "var(--good)" : "var(--crit)", opacity: .9, "data-tip": tipS }, g);
      txt(g, L + j * cw + cw / 2, top + i * ch + ch / 2 + 5, ok ? "✓" : "✗", { "text-anchor": "middle", style: "fill:#fff;font-weight:700;font-size:14px;pointer-events:none" });
    });
  });
  table("#t-needle", ["Context", ...deps.map(d => Math.round(100 * d) + "%")], lens.map(Ln => [(Ln / 1024) + "k", ...deps.map(d => { const x = it.find(q => q.length === Ln && q.depth === d); return !x ? "–" : x.correct ? "✓" : "✗"; })]));
}

// ---------- thresholds
function thresholds() {
  const T = D.thresholds || [];
  if (!T.length) { hide("#s-thr"); return; }
  const h = (D.hashes || {})["THRESHOLDS.md"];
  $("#thr-lede").innerHTML = `Fixed before the comparison ran${h ? ` (thresholds file <span class="mono">${esc(h.slice(0, 12))}…</span>)` : ""}. Each row is scored on a compared run against this reference; none is a verdict on the reference itself.`;
  const keys = ARMS.map(([k]) => k);
  table("#t-thr", ["#", "Metric", "Pass if", ...ARMS.map(([, a]) => esc(a.label))], T.map(t => [esc(t.id), esc(t.metric), esc(t.rule),
    ...(keys.length ? keys.map(k => { const r = (t.results || {})[k]; return r ? status(r.pass ? "good" : "crit", r.value || (r.pass ? "pass" : "fail")) : status("wait", "not measured"); }) : [])]));
}

// ---------- compared runs (arms)
function armHist(node, a, c) {
  const hst = a.kld.hist, w = Wn(node), L = 36, R = 10, T = 8, B = 26, h = 170, svg = svgIn(node, w, h), g = el("g", {}, svg);
  const n = hst.counts.length, mx = Math.max(...hst.counts, 1), bw = (w - L - R) / n, sy = v => T + (h - T - B) * (1 - v / mx);
  hst.counts.forEach((v, i) => {
    if (!v) return;
    el("rect", { x: L + i * bw + .5, y: sy(v), width: Math.max(1, bw - 1), height: h - B - sy(v), fill: c, opacity: .85,
      "data-tip": `<b>KLD ${hst.edges[i].toExponential(1)}–${hst.edges[i + 1].toExponential(1)} nats</b><br>${v.toLocaleString()} positions` }, g);
  });
  const lx = v => L + (w - L - R) * (Math.log10(v) - Math.log10(hst.edges[0])) / (Math.log10(hst.edges[n]) - Math.log10(hst.edges[0]));
  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100].filter(v => v >= hst.edges[0] && v <= hst.edges[n]).forEach(v => txt(g, lx(v), h - 8, v >= 1 ? v : v.toExponential(0), { class: "tick", "text-anchor": "middle" }));
  const m = lx(Math.max(hst.edges[0], a.kld.mean));
  el("line", { x1: m, x2: m, y1: T, y2: h - B, stroke: "var(--ink)", "stroke-width": 1.5 }, g);
  txt(g, m + 4, T + 10, `mean ${fmt(a.kld.mean, 4)}`, { class: "val" });
}
function arms() {
  if (!ARMS.length) { hide("#s-compare"); return; }
  $("#arms-lede").textContent = D.arms_note || "Each compared run answered the same items and scored the same token windows as this reference.";
  const box = $("#arms"); box.innerHTML = "";
  ARMS.forEach(([k, a], i) => {
    const c = ARMC[i % 3], div = document.createElement("div"); div.className = "arm"; box.appendChild(div);
    const kv = [];
    if (a.kld) kv.push(["Mean KLD", `${fmt(a.kld.mean, 4)} nats (median ${fmt(a.kld.p50, 4)}, p99 ${fmt(a.kld.p99, 3)})`]);
    if (a.top1) kv.push(["Top-1 agreement", `${fmt(a.top1.agree, 2)}%${a.top1.n ? ` of ${a.top1.n.toLocaleString()} positions` : ""}`]);
    for (const [s, p] of Object.entries(a.ppl || {})) kv.push([`Perplexity, ${esc((LP[s] || {}).label || s)}`, `${fmt(p.ref, 4)} → ${fmt(p.arm, 4)} (${sign(p.delta_pct)}%)`]);
    if (a.spec_parity) kv.push(["Spec parity", `${a.spec_parity.n - a.spec_parity.mismatches.length}/${a.spec_parity.n} prompts identical`]);
    div.innerHTML = `<h3><i style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${c};margin-right:6px"></i>${esc(a.label)}</h3>` +
      (a.description ? `<p class="lede" style="margin-bottom:8px">${esc(a.description)}</p>` : "") +
      `<div class="kv">${kv.map(([x, y]) => `<div>${x}</div><div>${y}</div>`).join("")}</div>` +
      `<div class="grid2" style="margin-top:12px"><div>${a.kld ? "<h3>KLD per position (log bins)</h3><div class=\"chart hist\"></div>" : ""}</div>` +
      `<div>${a.tasks ? "<h3>Task accuracy change vs reference (points)</h3><div class=\"chart dd\"></div>" : ""}</div></div>`;
    if (a.kld) armHist(div.querySelector(".hist"), a, c);
    if (a.tasks) {
      const lab = Object.fromEntries(D.tasks.map(t => [t.id, t.label]));
      const rows = Object.entries(a.tasks).map(([t, q]) => [lab[t] || t, q]);
      deltaDots(div.querySelector(".dd"), rows, c, a.label, "reference");
      div.insertAdjacentHTML("beforeend", `<details><summary>Table view</summary>${table(null, ["Task", "Items", "Change (pts)", "95% CI", "Right only ref", "Right only this run", "McNemar p"],
        rows.map(([l, q]) => [esc(l), q.n, sign(q.delta), `${sign(q.ci[0])} to ${sign(q.ci[1])}`, q.ref_only, q.test_only, pfmt(q.p)]), [1, 2, 4, 5, 6])}</details>`);
    }
    if (a.spec_parity && a.spec_parity.mismatches.length)
      div.insertAdjacentHTML("beforeend", `<details><summary>Spec parity mismatches</summary>${table(null, ["Prompt", "Token", "This run", "Reference", "Top-2 gap"],
        a.spec_parity.mismatches.map(m => [m.i, m.token, `<code>${esc(m.a)}</code>`, `<code>${esc(m.b)}</code>`, m.a_top2_gap == null ? "–" : fmt(m.a_top2_gap, 4)]), [0, 1, 4])}</details>`);
  });
}

// ---------- notes, repro
function notes() {
  const n = [...(D.caveats || []), ...(D.partials || []).map(p => `Incomplete: ${p}.`)];
  if (!n.length) { hide("#s-notes"); return; }
  $("#notes").innerHTML = n.map(x => `<li>${esc(x)}</li>`).join("");
}
function repro() {
  const e = D.env || {}, harness = [e.lm_eval && `lm-eval ${e.lm_eval}`, e.evalplus && `evalplus ${e.evalplus}`, e.bfcl_eval && `BFCL ${e.bfcl_eval}`].filter(Boolean).join(" · ");
  const kv = [["Model", esc(D.model)], ["Date", esc(e.date)], ["Precision", esc(e.precision)], ["Engine", esc(e.engine)], ["GPU", esc(e.gpu)], ["Harnesses", esc(harness)],
    ["evalplus image", e.evalplus_image ? `<span class="mono">${esc(e.evalplus_image)}</span>` : ""], ["Scoring sandbox", esc(e.sandbox)],
    ["Eval windows", SETS.length ? esc(SETS.map(([k]) => `${LP[k].label} ${LP[k].windows} × ${LP[k].window_tokens || LP[k].positions + 1}`).join(" · ") + ` · top-${(D.eval_meta || {}).topk || 64}`) : ""],
    ["Thinking-on runs", esc((D.think_dirs || []).join(", "))]].filter(([, v]) => v);
  $("#kv-env").innerHTML = kv.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join("");
  $("#kv-hash").innerHTML = Object.entries(D.hashes || {}).filter(([, v]) => v).map(([k, v]) => `<div>${esc(k)}</div><div class="mono">${esc(v)}</div>`).join("");
  $("#foot").textContent = `QualityBench ${D.qualitybench_version || ""} report · github.com/launch80/QualityBench · a self-contained file, no network access needed to view.`;
}
// theme toggle: auto -> light -> dark
const modes = ["auto", "light", "dark"]; let mi = 0;
try { const s = localStorage.getItem("qb-theme"); if (s) mi = Math.max(0, modes.indexOf(s)); } catch (e) {}
function applyTheme() { const m = modes[mi]; if (m === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", m); $("#theme").textContent = "Theme: " + m; }
$("#theme").addEventListener("click", () => { mi = (mi + 1) % 3; try { localStorage.setItem("qb-theme", modes[mi]); } catch (e) {} applyTheme(); });
applyTheme();
function draw() { tasksChart(); thinkCharts(); lpCharts(); needleChart(); arms(); }
header(); draw(); thresholds(); notes(); repro();
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(draw, 120); });
