// Crop: draw one 3-D box on three maximum projections of the raw scan. The
// converter cuts it out before anything else runs (an NRRD, which is not
// converted, is cut as the segmentation loads it), so every later step works
// on less data.

import { h, icon, replace, fmtBytes, errorToast, toast } from "../lib.js";
import { api, enc } from "../api.js";
import { on, previewJob } from "../store.js";
import { buildLut } from "../colormaps.js";
import { loadVolume, histogramPercentile } from "../volume.js";
import { go } from "../app.js";

// For each pane: which box entries its horizontal and vertical axes edit.
const PANES = [
  { plane: 0, title: "From above (projected through Z)", axes: [4, 5, 2, 3], names: ["x", "y"] },
  { plane: 1, title: "From the front (projected through Y)", axes: [4, 5, 0, 1], names: ["x", "z"] },
  { plane: 2, title: "From the side (projected through X)", axes: [2, 3, 0, 1], names: ["y", "z"] },
];
const EDGE = 9;

export async function mount(root, [name]) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  const sub = h("div", { class: "sub" }, "Drag a box in any view; the three views share it.");
  const head = h("div", { class: "page-head" },
    h("div", null, h("a", { href: `#/sample/${enc(name)}`, class: "muted", style: { fontSize: "13px" } }, name), h("h1", null, "Crop the scan"), sub));
  page.appendChild(head);
  const body = h("div");
  page.appendChild(body);

  let record, converted;
  try {
    const detail = await api.get(`/api/samples/${enc(name)}`);
    record = detail.record;
    converted = detail.summary.source?.format !== "nrrd";
  } catch (err) {
    replace(body, h("div", { class: "notice critical" }, err.message));
    return;
  }
  // What happens to the box: the converter cuts it out, or, for an NRRD, the segmentation.
  const verb = converted ? "converted" : "segmented";
  sub.textContent = `Drag a box in any view; the three views share it. Only what is inside is ${verb}.`;

  let vol = null, box = null, lut = null;
  // Set by "Whole scan": the converter should not crop to the specimen on its own either.
  let keepWhole = false;
  const panes = [];

  async function ensurePreview() {
    const views = await api.get(`/api/samples/${enc(name)}/views`);
    const info = views.find((v) => v.view === "source");
    if (!info) {
      replace(body, h("div", { class: "empty" }, h("h3", null, "There is no raw scan to crop"),
        h("span", null, "Cropping applies to a raw scan, before it is converted or segmented.")));
      return false;
    }
    if (info.built && info.current) return true;
    if (!previewJob(name, "source")) await api.post(`/api/samples/${enc(name)}/views/source/build`, { force: info.built });
    replace(body, h("div", { class: "empty" }, h("h3", null, "Reading a sample of the scan"),
      h("span", null, `Every few slices are read to build the preview; the full scan is read later, only once, by the ${converted ? "converter" : "segmentation"}.`),
      h("div", { class: "progress indeterminate", style: { width: "260px" } }, h("i"))));
    return false;
  }

  async function open() {
    if (!(await ensurePreview())) return;
    vol = await loadVolume(name, "source");
    const f = vol.meta.factors;
    const full = vol.meta.full_shape;
    const saved = record.crop_box;
    box = saved
      ? [saved[0] / f[0], saved[1] / f[0], saved[2] / f[1], saved[3] / f[1], saved[4] / f[2], saved[5] / f[2]]
      : [0, vol.Z, 0, vol.Y, 0, vol.X];
    const hist = vol.meta.histogram;
    let peak = 0;
    hist.counts.forEach((c, i) => { if (c > hist.counts[peak]) peak = i; });
    const tissue = { edges: hist.edges, counts: hist.counts.map((c, i) => (i <= peak ? 0 : c)) };
    lut = buildLut("gray", vol.q, histogramPercentile(tissue, 0.02), histogramPercentile(tissue, 0.999), 1, false);
    render(full);
  }

  function toFull() {
    const f = vol.meta.factors, full = vol.meta.full_shape;
    const out = [];
    for (let a = 0; a < 3; a++) {
      const lo = Math.max(0, Math.floor(Math.min(box[2 * a], box[2 * a + 1]) * f[a]));
      const hi = Math.min(full[a], Math.ceil(Math.max(box[2 * a], box[2 * a + 1]) * f[a]));
      out.push(lo, Math.max(hi, lo + 1));
    }
    return out;
  }

  const inputs = [];
  const summary = h("div", { class: "soft num" });
  function syncInputs() {
    const fb = toFull();
    inputs.forEach((inp, i) => { if (document.activeElement !== inp) inp.value = fb[i]; });
    const full = vol.meta.full_shape;
    const n = (fb[1] - fb[0]) * (fb[3] - fb[2]) * (fb[5] - fb[4]);
    const total = full[0] * full[1] * full[2];
    summary.textContent = `${fb[1] - fb[0]} × ${fb[3] - fb[2]} × ${fb[5] - fb[4]} voxels (z × y × x), ` +
      `${(100 * n / total).toFixed(1)}% of the scan, about ${fmtBytes(n * 2)} at 16 bits`;
  }

  function render(full) {
    const grid = h("div", { class: "crop-grid" });
    panes.length = 0;
    for (const spec of PANES) {
      const canvas = h("canvas");
      const holder = h("div", { class: "cv" }, canvas);
      grid.appendChild(h("div", { class: "crop-pane" }, h("h4", null, spec.title), holder));
      const pane = { spec, canvas, holder, layout: null, mip: vol.project(spec.plane, "max") };
      panes.push(pane);
      wire(pane);
    }
    const axisInputs = ["z", "y", "x"].map((axis, a) => {
      const lo = h("input", { type: "number", min: 0, max: full[a], "aria-label": `${axis} start` });
      const hi = h("input", { type: "number", min: 0, max: full[a], "aria-label": `${axis} end` });
      inputs[2 * a] = lo; inputs[2 * a + 1] = hi;
      const apply = () => {
        const f = vol.meta.factors[a];
        box[2 * a] = Math.max(0, Number(lo.value) / f);
        box[2 * a + 1] = Math.min(vol.meta.shape[a], Number(hi.value) / f);
        drawAll();
      };
      lo.addEventListener("change", apply);
      hi.addEventListener("change", apply);
      return h("div", { class: "field" }, h("span", { class: "lbl" }, `${axis.toUpperCase()} range (of ${full[a]})`),
        h("div", { class: "ax" }, lo, h("span", { class: "muted" }, "to"), hi));
    });
    const runAll = h("input", { type: "checkbox", checked: true });
    replace(body,
      grid,
      h("div", { class: "crop-inputs" }, axisInputs),
      h("div", { style: { marginTop: "10px" } }, summary),
      h("div", { class: "row wrap", style: { marginTop: "14px" } },
        h("button", { class: "btn", onclick: detect }, icon("target"), "Find the specimen"),
        h("button", { class: "btn", onclick: () => { box = [0, vol.Z, 0, vol.Y, 0, vol.X]; keepWhole = true; drawAll(); } }, "Whole scan"),
        h("span", { class: "spacer" }),
        h("button", { class: "btn", onclick: () => save(false) }, "Save crop box"),
        h("label", { class: "check" }, runAll, "and run the whole pipeline"),
        h("button", { class: "btn primary", onclick: () => save(true, runAll.checked) }, icon("play"), converted ? "Save and convert" : "Save and segment")));
    requestAnimationFrame(drawAll);
  }

  async function detect() {
    try {
      const res = await api.post(`/api/samples/${enc(name)}/crop/suggest`, { margin_mm: 0.5 });
      const f = vol.meta.factors;
      box = res.box.map((v, i) => v / f[Math.floor(i / 2)]);
      drawAll();
      toast("Box set around the largest piece of the specimen, with a 0.5 mm margin.", "info");
    } catch (err) { errorToast(err); }
  }

  async function save(andRun, runAll = false) {
    const full = toFull();
    const whole = full.every((v, i) => v === (i % 2 ? vol.meta.full_shape[Math.floor(i / 2)] : 0));
    // With no box, the converter still crops to the specimen, unless "Whole scan" said not to.
    const autocrops = whole && converted && !keepWhole;
    try {
      await api.post(`/api/samples/${enc(name)}/crop`, { box: whole ? null : full });
      if (!andRun) {
        toast(!whole ? "Crop box saved."
          : converted ? "No crop box. The converter crops to the specimen on its own unless Autocrop is off in its options."
            : `No crop: the whole scan will be ${verb}.`);
        return;
      }
      // A new box changes everything after the step that cuts it, so the whole
      // chain runs again rather than only what is missing.
      const chain = ["convert", "segment", "cmpr", "incisor_quant", "molar_quant"].filter((s) => converted || s !== "convert");
      const params = whole && converted && keepWhole ? { convert: { autocrop: false } } : {};
      const res = await api.post("/api/run", { samples: [name], steps: runAll ? chain : chain.slice(0, 1), params });
      if (res.skipped.length) toast(res.skipped[0].reason, "warning");
      else toast(`Queued ${res.jobs.map((j) => j.title).join(", ")}.${autocrops ? " With no box drawn, the converter crops to the specimen automatically."
        : whole ? ` The whole scan will be ${verb}.` : ""}`);
      go(`#/sample/${encodeURIComponent(name)}`);
    } catch (err) { errorToast(err); }
  }

  // ------------------------------------------------------------ drawing
  function drawAll() {
    for (const pane of panes) drawPane(pane);
    syncInputs();
  }

  function drawPane(pane) {
    const { canvas, holder, spec, mip } = pane;
    const rect = holder.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const [w, hh] = vol.dims(spec.plane);
    const [sx, sy] = vol.pixelMm(spec.plane);
    const k = Math.min((rect.width * 0.94) / (w * sx), (rect.height * 0.94) / (hh * sy));
    const dw = w * sx * k, dh = hh * sy * k, dx = (rect.width - dw) / 2, dy = (rect.height - dh) / 2;
    pane.layout = { dx, dy, dw, dh, w, h: hh };
    const off = document.createElement("canvas");
    off.width = w; off.height = hh;
    const img = off.getContext("2d").createImageData(w, hh);
    const b32 = new Uint32Array(img.data.buffer);
    for (let i = 0; i < w * hh; i++) b32[i] = lut[mip[i]];
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.fillStyle = "#0e1213";
    ctx.fillRect(0, 0, rect.width, rect.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, dx, dy, dw, dh);
    const [a0, a1, b0, b1] = spec.axes;
    const x0 = dx + (Math.min(box[a0], box[a1]) / w) * dw, x1 = dx + (Math.max(box[a0], box[a1]) / w) * dw;
    const y0 = dy + (Math.min(box[b0], box[b1]) / hh) * dh, y1 = dy + (Math.max(box[b0], box[b1]) / hh) * dh;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(dx, dy, dw, y0 - dy);
    ctx.fillRect(dx, y1, dw, dy + dh - y1);
    ctx.fillRect(dx, y0, x0 - dx, y1 - y0);
    ctx.fillRect(x1, y0, dx + dw - x1, y1 - y0);
    ctx.strokeStyle = "#7cc6cb";
    ctx.lineWidth = 2;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillStyle = "#7cc6cb";
    for (const [hx, hy] of [[x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2], [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1]]) {
      ctx.fillRect(hx - 4, hy - 4, 8, 8);
    }
    ctx.fillStyle = "rgba(226,233,231,0.8)";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(spec.names[0], dx + dw - 12, dy + dh + 12 > rect.height ? dy + dh - 6 : dy + dh + 12);
    ctx.fillText(spec.names[1], Math.max(4, dx - 12), dy + 12);
  }

  function wire(pane) {
    let drag = null;
    const toImage = (e) => {
      const r = pane.canvas.getBoundingClientRect();
      const { dx, dy, dw, dh, w, h: hh } = pane.layout;
      return { u: ((e.clientX - r.left - dx) / dw) * w, v: ((e.clientY - r.top - dy) / dh) * hh, sx: e.clientX - r.left, sy: e.clientY - r.top };
    };
    pane.canvas.addEventListener("pointerdown", (e) => {
      if (!pane.layout) return;
      const p = toImage(e);
      const [a0, a1, b0, b1] = pane.spec.axes;
      const { dx, dy, dw, dh, w, h: hh } = pane.layout;
      const X0 = dx + (box[a0] / w) * dw, X1 = dx + (box[a1] / w) * dw, Y0 = dy + (box[b0] / hh) * dh, Y1 = dy + (box[b1] / hh) * dh;
      const near = (a, b) => Math.abs(a - b) < EDGE;
      const inY = p.sy > Y0 - EDGE && p.sy < Y1 + EDGE, inX = p.sx > X0 - EDGE && p.sx < X1 + EDGE;
      const edges = { l: near(p.sx, X0) && inY, r: near(p.sx, X1) && inY, t: near(p.sy, Y0) && inX, b: near(p.sy, Y1) && inX };
      if (edges.l || edges.r || edges.t || edges.b) drag = { kind: "edges", edges };
      else if (p.sx > X0 && p.sx < X1 && p.sy > Y0 && p.sy < Y1) drag = { kind: "move", u: p.u, v: p.v, start: [...box] };
      else {
        const u = Math.max(0, Math.min(w, p.u)), v = Math.max(0, Math.min(hh, p.v));
        box[a0] = u; box[a1] = u; box[b0] = v; box[b1] = v;
        drag = { kind: "edges", edges: { r: true, b: true } };
      }
      pane.canvas.setPointerCapture(e.pointerId);
    });
    pane.canvas.addEventListener("pointermove", (e) => {
      if (!pane.layout) return;
      const p = toImage(e);
      const [a0, a1, b0, b1] = pane.spec.axes;
      const { w, h: hh } = pane.layout;
      if (!drag) {
        const inside = p.u >= 0 && p.v >= 0 && p.u <= w && p.v <= hh;
        pane.canvas.style.cursor = inside ? "crosshair" : "default";
        return;
      }
      const cu = Math.max(0, Math.min(w, p.u)), cv = Math.max(0, Math.min(hh, p.v));
      if (drag.kind === "move") {
        const du = p.u - drag.u, dv = p.v - drag.v;
        const spanU = drag.start[a1] - drag.start[a0], spanV = drag.start[b1] - drag.start[b0];
        box[a0] = Math.max(0, Math.min(w - spanU, drag.start[a0] + du)); box[a1] = box[a0] + spanU;
        box[b0] = Math.max(0, Math.min(hh - spanV, drag.start[b0] + dv)); box[b1] = box[b0] + spanV;
      } else {
        if (drag.edges.l) box[a0] = cu;
        if (drag.edges.r) box[a1] = cu;
        if (drag.edges.t) box[b0] = cv;
        if (drag.edges.b) box[b1] = cv;
      }
      drawAll();
    });
    pane.canvas.addEventListener("pointerup", () => {
      if (!drag) return;
      drag = null;
      const [a0, a1, b0, b1] = pane.spec.axes;
      for (const [i, j] of [[a0, a1], [b0, b1]]) {
        if (box[i] > box[j]) [box[i], box[j]] = [box[j], box[i]];
        if (box[j] - box[i] < 1) box[j] = box[i] + 1;
      }
      drawAll();
    });
  }

  const ro = new ResizeObserver(() => { if (vol && panes.length) drawAll(); });
  ro.observe(body);
  const off = on("jobs", (job) => {
    if (job && job.sample === name && job.step === "preview" && job.params?.view === "source" && job.status === "done") open();
  });
  try { await open(); } catch (err) { errorToast(err); }
  return () => { ro.disconnect(); off(); };
}
