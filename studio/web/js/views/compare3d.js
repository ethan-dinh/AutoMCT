// Compare: specimens side by side in 3-D, on one heatmap scale, turning together.
//
// Each panel ray-marches one specimen's preview, as the viewer's 3-D mode
// does. The colour window, the surface level and every other display setting
// are shared, and set on one scale -- mg HA/cm³, unless no scan has an HA
// calibration: each specimen's own calibration turns the window back into its
// stored values (scale.js), so a colour means one density in every panel,
// whatever setting each scan was made with. Dragging any panel turns them
// all, all the way round.
//
// Specimens sit differently in their scans (the segmentation only permutes
// and flips axes, so each keeps its tilt in the holder), so each one is turned
// about its own tissue's centre and, as it arrives, turned and moved to fit
// the first one (pose.js). Alt-drag, or a panel's own move button, turns and
// moves one specimen alone, to finish the job by hand.
//
// For differences finer than a voxel's noise (about 45 mg HA/cm³ in 6 µm
// enamel), the colour can be averaged over a little tissue beneath the
// surface, cut into steps at round densities with their edges drawn, and read
// against a reference density on the above/below scale; turning the shading
// down keeps the light from moving colours off the scale.

import { h, icon, replace, sig, numFmt, calibrationFormula, basename, ordinal } from "../lib.js";
import { api, enc } from "../api.js";
import { state, on, previewJob } from "../store.js";
import { COLORMAPS, SEQUENTIAL, gradientCanvas, colorAt, RANDOM, MOST_COLOURS, stepPalette, stepsCanvas } from "../colormaps.js";
import { loadVolume, histogramPercentile, tissueHistogram } from "../volume.js";
import { Render3D } from "../render3d.js";
import { register, identity } from "../pose.js";
import { histWindow } from "../histwin.js";
import { niceTicks } from "../charts.js";
import { levelOrder, levelOf } from "../compare.js";
import { toScale, fromScale, sharedWindow, gridColumns, rampPosition, stepWidth, stepChoices, stepsFor, randomSteps, stepColour } from "../scale.js";
import { DEFAULT_CMAP, OVERLAY_COLORS } from "./viewer.js";

const MAX_PANELS = 4;
const VIEW_ORDER = ["molar", "incisor", "cmpr", "mandible", "bone", "converted", "source"];
// One colour per panel, for its caption and its histogram line: the dark stage's first four categorical steps.
const HUES = ["#3987e5", "#d95926", "#199e70", "#c98500"];
const SIDES = [["side", "Side"], ["other", "Other side"], ["top", "Top"], ["bottom", "Below"], ["end", "End"]];
const SCALES = [["mg", "mg HA/cm³"], ["raw", "Raw intensity"]];
// Blue below the centre, grey at it, red above: the studio's two-armed scale.
const DIVERGING = "ratio";
// Cut into steps, each its own hue on the Rainbow scale: about this many across the window...
const STEP_COUNTS = [6, 12, 24, 48, 96];
const DEFAULT_STEPS = "count:12";
// ...and never more than this many, however fine a width is typed.
const MAX_STEPS = 500;
const RAINBOW = "rainbow";
// How far the colour averages around each surface point, in the coarsest preview's voxels.
const SMOOTHING = [0, 1, 2, 3];
// How far the scan is smoothed in 3-D first -- a Gaussian's sigma -- in the same voxels,
// and what the random scale arrives with: its colours scatter on the scan's own noise.
const SOFTEN = [0, 1, 2, 3];
const RANDOM_SOFTEN = 2;
const STAGE = "#0e1213";
const MAC = /Mac/.test(navigator.platform);
const ALT = MAC ? "Option" : "Alt";
const ROLL = MAC ? "⌘" : "Ctrl";
// How specimens are placed: turned and moved to fit the first, or as each sits in its scan.
const ALIGNS = [["match", "Fitted to the first"], ["scan", "As scanned"]];

const plain = (v) => sig(v, 5).replace(/,/g, "");
// A length in mm as µm, as plainly as it reads: 0.006 -> "6", 0.0125 -> "12.5".
const um = (mm) => String(Number((mm * 1000).toPrecision(3)));
const copyCam = (c) => ({ rot: [...c.rot], dist: c.dist, target: [...c.target] });
const deg = (rad) => Math.round((rad * 180) / Math.PI);

// What the page remembers per workspace: what it compares and how it looks.
const storeKey = () => `studio.compare3d:${state.app?.workspace?.root || ""}`;
function loadSaved() {
  try { return JSON.parse(localStorage.getItem(storeKey()) || "{}") || {}; } catch { return {}; }
}

// A colour bar with its ticks, for the legend and the saved image: every
// colour as the panels give it (`colour` of a value, steps and all), a tick at
// each step's edge and labels on as many as fit.
function colorbar(c, x, y, w, barH, scale, colour, ink, font) {
  const { lo, hi, steps } = scale;
  for (let i = 0; i < w; i++) {
    const [r, g, b] = colour(lo + ((i + 0.5) / w) * (hi - lo));
    c.fillStyle = `rgb(${r},${g},${b})`;
    c.fillRect(x + i, y, 1.5, barH);  // a hair wider than a pixel: no seams at fractional scales
  }
  let ticks = niceTicks(lo, hi, 4), step = ticks.length > 1 ? ticks[1] - ticks[0] : null, labelled = ticks;
  if (steps) {
    ticks = [];
    for (let k = Math.ceil((lo - steps.origin) / steps.width - 1e-9); ticks.length < 200; k++) {
      const v = steps.origin + k * steps.width;
      if (v > hi + 1e-9 * Math.max(1, Math.abs(hi))) break;
      ticks.push(v);
    }
    const every = Math.max(1, Math.ceil(ticks.length / 7));
    labelled = ticks.filter((_, i) => i % every === 0);
    step = steps.width * every;
    if (ticks.length > 40) ticks = labelled;  // fine steps: their colours mark them, not a comb of ticks
  }
  c.fillStyle = ink;
  c.font = font;
  c.textAlign = "center";
  c.textBaseline = "top";
  for (const v of ticks) {
    const px = x + ((v - lo) / (hi - lo || 1)) * w;
    if (px < x - 0.5 || px > x + w + 0.5) continue;
    const major = labelled.includes(v);
    c.fillRect(px - 0.5, y + barH, 1, barH * (major ? 0.4 : 0.25));
    if (major) c.fillText(numFmt(v, step), px, y + barH * 1.6);
  }
}

const cmapIsDiverging = (ui) => ui.cmap === DIVERGING;

// A saved steps setting, as the page reads it: "off", "count:N", a width, or the old "auto" (about six).
function stepsSetting(v) {
  if (v === "off" || (typeof v === "number" && v > 0)) return v;
  if (typeof v === "string" && /^count:\d+$/.test(v)) return v;
  return v === "auto" ? "count:6" : DEFAULT_STEPS;
}
const stepCount = (setting) => Number(String(setting).split(":")[1]) || 12;

// Specimens by group, controls first, as the results page orders them.
function grouped() {
  const all = [...state.samples.values()];
  const order = levelOrder(all);
  const rank = (s) => [order.genotype.indexOf(levelOf(s.meta, "genotype")), order.age.indexOf(levelOf(s.meta, "age"))];
  all.sort((a, b) => {
    const [ga, aa] = rank(a), [gb, ab] = rank(b);
    return ga - gb || aa - ab || a.name.localeCompare(b.name, undefined, { numeric: true });
  });
  const out = new Map();
  for (const s of all) {
    if (!out.has(s.group)) out.set(s.group, []);
    out.get(s.group).push(s);
  }
  return [...out];
}

export async function mount(root, _params, query) {
  const views = state.app?.views || {};
  const saved = loadSaved();
  const ui = {
    view: views[query.view] ? query.view : views[saved.view] ? saved.view : "molar",
    scale: saved.scale === "raw" ? "raw" : "mg",
    cmap: SEQUENTIAL.includes(saved.cmap) || [DIVERGING, RAINBOW, RANDOM].includes(saved.cmap) ? saved.cmap : null,  // null: the view's own default
    style: saved.style === "xray" ? "xray" : "surface",
    linked: saved.linked !== false,
    align: saved.align === "scan" ? "scan" : "match",
    // The window, on the shared scale. fit: what it follows as specimens come and go --
    // "enamel" (the middle of every one's enamel), "tissue" (all of it), "centre" (the
    // above/below scale's) -- or null once set by hand.
    lo: 0, hi: 1, gamma: 1, fit: null,
    // isoAuto: the page chose the surface level (on the enamel, say) and may move it as specimens come.
    // The surface level on the shared scale (null follows the window's low end).
    iso: null, isoAuto: true, cut: 0, tint: 0.7, structures: {},
    side: "side",
    // Finer differences: steps ("off", "count:N" -- about N across the window -- or a width on the
    // shared scale), each one colour or not; how much light shapes the surface; how far the colour
    // averages, in voxels.
    steps: stepsSetting(saved.steps),
    stepFill: saved.stepFill !== false,
    // Which order the random scale deals its colours in (0: the order they were picked).
    seed: Number.isInteger(saved.seed) && saved.seed >= 0 ? saved.seed : 0,
    shade: Number.isFinite(saved.shade) ? Math.min(1, Math.max(0, saved.shade)) : 0.4,
    smooth: SMOOTHING.includes(saved.smooth) ? saved.smooth : 2,
    soften: SOFTEN.includes(saved.soften) ? saved.soften : 0,
  };
  ui.fit = cmapIsDiverging(ui) ? "centre" : ui.view === "molar" ? "enamel" : "tissue";
  if (ui.cmap === RANDOM && ui.steps === "off") ui.steps = DEFAULT_STEPS;
  const wanted = (query.s != null ? String(query.s).split(",") : saved.names || [])
    .filter((n, i, all) => state.samples.has(n) && all.indexOf(n) === i).slice(0, MAX_PANELS);

  // ------------------------------------------------------------ layout
  const shell = h("div", { class: "viewer c3d" });
  root.appendChild(shell);
  const topbar = h("div", { class: "topbar" });
  const grid = h("div", { class: "c3d-grid" });
  const empty = h("div", { class: "c3d-empty", hidden: true });
  const stage = h("div", { class: "c3d-stage" }, grid, empty);
  const bar = h("canvas", { class: "c3d-bar", "aria-hidden": "true" });
  const barNote = h("span", { class: "c3d-note" });
  const legend = h("div", { class: "c3d-legend" }, bar, barNote, h("span", { class: "spacer" }),
    h("span", { class: "c3d-hint" }, `Drag turns them all · Shift-drag moves · ${ROLL}-drag rolls · ${ALT} with any of them: one alone · Scroll zooms · Double-click resets`));
  const side = h("aside", { class: "side", "aria-label": "Display settings" });
  shell.append(h("div", { class: "stage-wrap" }, topbar, stage, legend), side);

  // ------------------------------------------------------------ panels
  const panels = [];
  let cam = null;         // the shared camera
  let fitted = true;      // no one has turned or zoomed since the views were fitted: a new specimen refits them
  let lastMoved = null;
  let cols = 1;

  const ready = (p) => Boolean(p.vol && p.r3d.ok && p.r3d.textures);
  const onScale = (p) => ui.scale === "raw" || Boolean(p.cal);
  const shown = () => panels.filter((p) => ready(p) && onScale(p));
  const calOf = (p) => (ui.scale === "mg" ? p.cal : null);
  const toShared = (p, v) => toScale(v, calOf(p));
  const fromShared = (p, v) => fromScale(v, calOf(p));
  const cmap = () => ui.cmap || DEFAULT_CMAP[ui.view] || "mineral";
  const diverging = () => cmap() === DIVERGING;
  const random = () => cmap() === RANDOM;
  const unitText = () => (ui.scale === "mg" ? "mg HA/cm³" : "raw intensity");
  // The coarsest preview's voxel, in mm: what the colour's depth and averaging are measured in, alike in every panel.
  const voxelMm = () => { const v = shown().map((p) => p.r3d.voxel); return v.length ? Math.max(...v) : null; };
  // How far beneath the surface its colour is read: two of those voxels.
  const sharedDeep = () => { const v = voxelMm(); return v == null ? null : 2 * v; };
  const smoothMm = () => (voxelMm() || 0) * ui.smooth;
  const softenMm = () => (voxelMm() || 0) * ui.soften;
  // The steps on the shared scale, their edges on round values: multiples of
  // the width, or the centre plus multiples on the above/below scale.
  // No edge lines on the surfaces: each step shows by its own colour.
  function sharedSteps() {
    if (ui.steps === "off") return null;
    const typed = typeof ui.steps === "number" ? ui.steps : stepWidth(ui.lo, ui.hi, stepCount(ui.steps));
    const width = Math.max(typed, Math.abs(ui.hi - ui.lo) / MAX_STEPS);
    if (!(width > 0)) return null;
    if (random()) return randomSteps(ui.lo, ui.hi, width, ui.seed);
    return { origin: diverging() ? (ui.lo + ui.hi) / 2 : 0, width, fill: ui.stepFill, lines: false };
  }
  const scaleState = () => ({ lo: ui.lo, hi: ui.hi, gamma: ui.gamma, steps: sharedSteps() });
  const colourOf = (v) => {
    const scale = scaleState();
    return scale.steps?.palette ? stepColour(v, scale) : colorAt(cmap(), rampPosition(v, scale));
  };

  function makePanel(name) {
    const used = new Set(panels.map((p) => p.color));
    const canvas = h("canvas", { class: "gl3d", tabindex: 0, hidden: true,
      "aria-label": `${name} in 3-D. Drag to turn every specimen, ${ALT}-drag to turn this one alone, Shift-drag to move, scroll to zoom.` });
    // adjusted: turned or moved alone by hand since it was lined up; fit: how it was lined up (pose.js).
    const p = { name, color: HUES.find((c) => !used.has(c)) || HUES[0], canvas, cap: h("div", { class: "c3d-cap" }),
      msg: h("div", { class: "loading", hidden: true }), vol: null, cal: null, molar: null, token: 0,
      adjusted: false, fit: null, reg: null };
    p.aloneBtn = h("button", { class: "btn small icon c3d-alone", "aria-pressed": "false", "aria-label": `Turn and move ${name} alone`,
      title: "Turn and move this one alone: drag turns it, Shift-drag moves it", onclick: () => setAlone(p, !p.r3d.alone) }, icon("move"));
    p.el = h("div", { class: "stage c3d-panel" }, canvas, p.cap, p.msg, p.aloneBtn,
      h("button", { class: "btn small icon c3d-x", "aria-label": `Remove ${name}`, title: "Take it out of the comparison",
        onclick: () => removePanel(p) }, icon("x")));
    p.el.style.setProperty("--hue", p.color);
    p.r3d = new Render3D(canvas);
    p.r3d.alignable = true;
    p.r3d.onMove = (how) => moved(p, how);
    return p;
  }

  // A panel whose plain drags turn and move its specimen alone, not the camera.
  function setAlone(p, on) {
    p.r3d.alone = on;
    p.aloneBtn.setAttribute("aria-pressed", on ? "true" : "false");
    p.el.classList.toggle("alone", on);
    renderCaption(p);
  }

  function show(p, text, progress = null) {
    p.canvas.hidden = true;
    if (!p.msg.hidden && progress == null && p.msg.textContent === text) return;  // already saying it
    p.msg.hidden = false;
    replace(p.msg, h("div", null, text,
      progress != null ? h("div", { class: "progress" }, h("i", { style: { width: `${Math.round(progress * 100)}%` } })) : null));
  }

  async function openPanel(p) {
    const token = ++p.token, view = ui.view;
    const stale = () => token !== p.token || !panels.includes(p);
    Object.assign(p, { vol: null, cal: null, molar: null, adjusted: false, fit: null, reg: null });
    renderCaption(p);
    if (!p.r3d.ok) { show(p, p.r3d.error); return; }
    show(p, "Loading…");
    try {
      const list = await api.get(`/api/samples/${enc(p.name)}/views`);
      if (stale()) return;
      const info = list.find((v) => v.view === view);
      if (!info) { show(p, `${p.name} has no ${(views[view] || view).toLowerCase()} view yet.`); return; }
      if (!info.built || !info.current) {
        if (!previewJob(p.name, view)) await api.post(`/api/samples/${enc(p.name)}/views/${view}/build`, { force: info.built && !info.current });
        if (!stale()) show(p, info.built ? "The preview is out of date; rebuilding it." : "Building the preview. Large scans take a minute.", 0);
        return;
      }
      const vol = await loadVolume(p.name, view);
      const molar = view === "molar" ? await api.get(`/api/samples/${enc(p.name)}/molar`).then((r) => r.summary, () => null) : null;
      if (stale()) return;
      p.r3d.setVolume(vol);
      // The renderer holds the voxels now: the page keeps only what it reads, so
      // four panels do not keep four scans' worth of arrays alive as well.
      Object.assign(p, { vol: { meta: vol.meta, q: vol.q }, cal: vol.meta.calibration || null, molar });
      if (!fitted && ui.linked && cam) p.r3d.setCamera(cam);
    } catch (err) {
      if (!stale()) show(p, err.message || String(err));
      return;
    }
    settle();
  }

  // After a specimen arrives or leaves, or the scale changes: the window, the
  // camera and everything drawn from them follow.
  function settle() {
    const loaded = panels.filter(ready);
    // With nothing to convert, stored values are the only scale there is.
    if (ui.scale === "mg" && loaded.length && !loaded.some((p) => p.cal)) ui.scale = "raw";
    lineUpAll();
    if (ui.fit) autoWindow();
    if (fitted) fitAll(ui.side);
    renderTopbar();
    panels.forEach(renderCaption);
    renderSide();
    drawAll();
  }

  // What the window follows until it is set by hand: on the molars, the middle
  // of the enamel -- the finest differences there are the point -- else all the
  // tissue; on the above/below scale, a window centred on their references.
  const defaultFit = () => (diverging() ? "centre" : ui.view === "molar" ? "enamel" : "tissue");

  function autoWindow() {
    const list = shown();
    const enamel = ui.view === "molar" && list.length &&
      list.every((p) => p.molar?.enamel_low_raw != null && p.molar?.enamel_high_raw != null && p.molar?.threshold != null);
    if (ui.fit === "centre") centredWindow();
    else if (ui.fit === "enamel" && enamel) {
      ui.lo = Math.min(...list.map((p) => toShared(p, p.molar.enamel_low_raw)));
      ui.hi = Math.max(...list.map((p) => toShared(p, p.molar.enamel_high_raw)));
    } else {
      const win = sharedWindow(list.map((p) => ({ hist: p.vol.meta.histogram, masked: p.vol.meta.masked, cal: calOf(p) })));
      if (win) { ui.lo = win.lo; ui.hi = win.hi; }
    }
    // Around the enamel, the surface is the enamel's; over all the tissue it follows the window's low end.
    if (ui.isoAuto) ui.iso = ui.fit === "centre" || (ui.fit === "enamel" && enamel) ? enamelSurface() : null;
  }

  // The surface on the enamel: the lowest enamel threshold among them, so every
  // one's enamel is inside it. Null (follow the window) off the molar view.
  function enamelSurface() {
    const list = shown();
    if (ui.view !== "molar" || !list.length || !list.every((p) => p.molar?.threshold != null)) return null;
    return Math.min(...list.map((p) => toShared(p, p.molar.threshold)));
  }

  // Densities the above/below scale can be centred on: each specimen's enamel
  // mean on the molar view, else the median of its tissue.
  function references() {
    return shown().map((p) => {
      const enamel = ui.view === "molar" && p.molar?.enamel_mean_raw != null;
      const raw = enamel ? p.molar.enamel_mean_raw : histogramPercentile(tissueHistogram(p.vol.meta.histogram, p.vol.meta.masked), 0.5);
      return { name: p.name, value: toShared(p, raw), what: enamel ? "enamel mean" : "median" };
    });
  }

  // The above/below window: centred on the specimens' average reference, and
  // wide enough for the middle of every one's enamel (or of its tissue).
  function centredWindow() {
    const refs = references();
    if (!refs.length) return;
    const centre = refs.reduce((a, r) => a + r.value, 0) / refs.length;
    const reach = shown().flatMap((p) => {
      if (ui.view === "molar" && p.molar?.enamel_low_raw != null && p.molar?.enamel_high_raw != null) {
        return [toShared(p, p.molar.enamel_low_raw), toShared(p, p.molar.enamel_high_raw)];
      }
      const t = tissueHistogram(p.vol.meta.histogram, p.vol.meta.masked);
      return [toShared(p, histogramPercentile(t, 0.25)), toShared(p, histogramPercentile(t, 0.75))];
    });
    const half = Math.max(...reach.map((v) => Math.abs(v - centre))) || Math.abs(centre) * 0.1 || 1;
    ui.lo = centre - half;
    ui.hi = centre + half;
  }

  // Every specimen seen from one side, from a distance at which the largest fits:
  // the same distance for all, so they are drawn at one scale.
  function fitAll(sideName) {
    const loaded = panels.filter(ready);
    const sized = shown().length ? shown() : loaded;
    if (!sized.length) return;
    const dist = Math.max(...sized.map((p) => p.r3d.fitDistance()));
    for (const p of loaded) p.r3d.setView(sideName, dist);
    cam = copyCam(sized[0].r3d.cam);
  }

  // Where each specimen sits. The first one loaded sits as scanned, turning
  // about its tissue's centre; each other one is turned and moved to fit it,
  // unless they are all to sit as scanned. One turned or moved by hand since
  // keeps where it was put.
  function lineUpAll() {
    const loaded = panels.filter(ready);
    const first = loaded[0];
    for (const p of loaded) {
      if (p.adjusted) continue;
      p.fit = ui.align === "match" && p !== first ? fitTo(first, p) : null;
      p.r3d.setPose(p.fit || { rot: identity(), shift: [0, 0, 0] });
    }
  }

  // How p fits `first`, worked out once for each pair of previews.
  function fitTo(first, p) {
    const key = `${first.name}|${first.r3d.volumeKey}|${p.r3d.volumeKey}`;
    if (p.reg?.key !== key) {
      const fit = register(first.r3d.cloud, p.r3d.cloud);
      p.reg = { key, fit: fit && { ...fit, to: first.name } };
    }
    return p.reg.fit;
  }

  function setAlign(align) {
    ui.align = align;
    remember();
    for (const p of panels) p.adjusted = false;
    lineUpAll();
    if (fitted) fitAll(ui.side);
    panels.forEach(renderCaption);
    renderSide();
  }

  function moved(p, how) {
    lastMoved = p;
    if (how === "align") {
      if (!p.adjusted) { p.adjusted = true; renderCaption(p); renderSide(); }
      return;
    }
    if (how === "reset") {
      if (ui.linked) { ui.side = "side"; fitted = true; fitAll("side"); }
      return;
    }
    fitted = false;
    if (!ui.linked) return;
    cam = copyCam(p.r3d.cam);
    for (const q of panels) if (q !== p && ready(q)) q.r3d.setCamera(cam);
  }

  function drawAll() {
    const deep = sharedDeep(), smooth = smoothMm(), steps = sharedSteps(), blur = softenMm();
    for (const p of panels) {
      if (!ready(p)) continue;
      if (!onScale(p)) {
        show(p, "No HA calibration for this scan, so it has no place on the mg HA/cm³ scale. Raw intensity shows it.");
        continue;
      }
      p.canvas.hidden = false;
      p.msg.hidden = true;
      p.r3d.setOptions({
        style: ui.style, level: fromShared(p, ui.iso ?? ui.lo), lo: fromShared(p, ui.lo), hi: fromShared(p, ui.hi),
        gamma: ui.gamma, cmap: cmap(), cut: ui.cut, tint: ui.tint, deep, smooth, shade: ui.shade, steps: stepsFor(steps, calOf(p)), blur,
        overlays: p.vol.meta.overlays.map((o) => ({ on: Boolean(ui.structures[o.name]), color: OVERLAY_COLORS[o.name] || "#ffffff" })),
      });
      p.r3d.render();
    }
    drawLegend();
  }

  function drawLegend() {
    const w = 260, hgt = 30, dpr = window.devicePixelRatio || 1;
    bar.width = Math.round(w * dpr);
    bar.height = Math.round(hgt * dpr);
    bar.style.width = `${w}px`;
    bar.style.height = `${hgt}px`;
    const c = bar.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, hgt);
    if (!shown().length) { replace(barNote); return; }
    colorbar(c, 18, 2, w - 36, 10, scaleState(), colourOf, "#a3aeac", "10.5px system-ui, sans-serif");
    const steps = sharedSteps();
    replace(barNote, h("b", null, ui.scale === "mg" ? "mg HA/cm³" : "Raw intensity"),
      diverging() ? [" · centred on ", h("b", null, sig((ui.lo + ui.hi) / 2, 4))] : null,
      steps ? [" · steps of ", h("b", null, numFmt(steps.width, steps.width))] : null,
      random() ? " · grey: outside the window" : null,
      ui.style === "xray" ? " · the densest value along each ray"
        : [" · surface at ", h("b", null, sig(ui.iso ?? ui.lo, 4)), ui.iso == null ? ", the window's low end" : ""]);
  }

  // ------------------------------------------------------------ adding and removing
  function addPanel(name) {
    if (panels.length >= MAX_PANELS || panels.some((p) => p.name === name) || !state.samples.has(name)) return;
    const p = makePanel(name);
    panels.push(p);
    grid.appendChild(p.el);
    arranged();
    openPanel(p);
  }

  function removePanel(p) {
    const i = panels.indexOf(p);
    if (i < 0) return;
    panels.splice(i, 1);
    p.token++;  // anything still loading for it is dropped
    p.r3d.dispose();
    p.el.remove();
    if (lastMoved === p) lastMoved = null;
    arranged();
    settle();
  }

  // After a specimen is added or taken away: the grid, the address and what is remembered.
  function arranged() {
    layout();
    empty.hidden = panels.length > 0;
    if (!panels.length) renderEmpty();
    remember();
    renderTopbar();
    panels.forEach(renderCaption);
  }

  function layout() {
    const n = panels.length, W = stage.clientWidth || 1, H = stage.clientHeight || 1;
    // A layout that barely wins keeps the one in place: panels do not jump when the legend grows a line.
    const room = (c) => Math.min(W / c, (H / Math.ceil(n / c)) * 1.3);
    const best = gridColumns(n, W, H);
    if (!(cols <= n && cols >= 1 && room(best) < room(cols) * 1.06)) cols = best;
    grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    grid.style.gridTemplateRows = `repeat(${Math.max(1, Math.ceil(n / cols))}, minmax(0, 1fr))`;
    if (fitted) fitAll(ui.side);
    for (const p of panels) if (ready(p)) p.r3d.render();
  }
  const ro = new ResizeObserver(layout);
  ro.observe(stage);

  function remember() {
    const names = panels.map((p) => p.name);
    try {
      localStorage.setItem(storeKey(), JSON.stringify({ view: ui.view, names, scale: ui.scale, cmap: ui.cmap, style: ui.style,
        linked: ui.linked, align: ui.align, steps: ui.steps, stepFill: ui.stepFill, seed: ui.seed, shade: ui.shade, smooth: ui.smooth, soften: ui.soften }));
    } catch { /* private window */ }
    // The address says what is compared, so it can be reloaded or sent on.
    const next = `#/compare?view=${enc(ui.view)}${names.length ? `&s=${names.map(enc).join(",")}` : ""}`;
    if (location.hash !== next) history.replaceState(null, "", next);
  }

  // ------------------------------------------------------------ changing what is shown
  function setView(view) {
    if (view === ui.view) return;
    Object.assign(ui, { view, iso: null, isoAuto: true, structures: {}, side: "side" });
    ui.fit = defaultFit();
    fitted = true;
    remember();
    for (const p of panels) openPanel(p);
    renderTopbar();
    renderSide();
    drawLegend();
  }

  function setScale(scale) {
    if (scale === ui.scale) return;
    Object.assign(ui, { scale, iso: null, isoAuto: true });
    ui.fit = defaultFit();
    if (typeof ui.steps === "number") ui.steps = DEFAULT_STEPS;  // a width in one scale's units means nothing in the other's
    remember();
    settle();
  }

  function setLinked(linked) {
    ui.linked = linked;
    remember();
    // Turning together again: every view takes the one moved last.
    const loaded = panels.filter(ready);
    const lead = loaded.includes(lastMoved) ? lastMoved : loaded[0];
    if (linked && lead) {
      cam = copyCam(lead.r3d.cam);
      for (const p of loaded) if (p !== lead) p.r3d.setCamera(cam);
    }
    renderSide();
  }

  function resetView() {
    ui.side = "side";
    fitted = true;
    fitAll("side");
  }

  function lineUp() {
    for (const p of panels) if (p.adjusted) { p.adjusted = false; renderCaption(p); }
    lineUpAll();
    renderSide();
  }

  // ------------------------------------------------------------ the window
  const hist = histWindow((lo, hi) => { ui.lo = lo; ui.hi = hi; ui.fit = null; syncInputs(); drawAll(); });
  const loIn = h("input", { type: "number", step: "any", "aria-label": "Window low", onchange: () => windowTo(Number(loIn.value), ui.hi) });
  const hiIn = h("input", { type: "number", step: "any", "aria-label": "Window high", onchange: () => windowTo(ui.lo, Number(hiIn.value)) });
  const centreIn = h("input", { type: "number", step: "any", "aria-label": "Centre", onchange: () => {
    const half = (ui.hi - ui.lo) / 2, c = Number(centreIn.value);
    windowTo(c - half, c + half);
  } });
  const reachIn = h("input", { type: "number", step: "any", min: 0, "aria-label": "Either side of the centre", onchange: () => {
    const c = (ui.lo + ui.hi) / 2, half = Math.abs(Number(reachIn.value));
    windowTo(c - half, c + half);
  } });
  function syncInputs() {
    loIn.value = plain(ui.lo); hiIn.value = plain(ui.hi);
    centreIn.value = plain((ui.lo + ui.hi) / 2); reachIn.value = plain((ui.hi - ui.lo) / 2);
  }
  function windowTo(lo, hi) {
    if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) Object.assign(ui, { lo, hi, fit: null });
    syncInputs();
    hist.set(histState());
    drawAll();
  }

  function histState() {
    const list = shown();
    return {
      series: list.map((p) => {
        const t = tissueHistogram(p.vol.meta.histogram, p.vol.meta.masked);
        return { edges: t.edges.map((e) => toShared(p, e)), counts: t.counts, color: p.color };
      }),
      lo: ui.lo, hi: ui.hi, gamma: ui.gamma, cmap: cmap(), color: colourOf,
      markers: list.filter((p) => p.molar?.threshold != null).map((p) => ({ value: toShared(p, p.molar.threshold), color: p.color })),
    };
  }

  // [label, what it does, set it]: each over every specimen shown, on the shared scale.
  // A preset with a fit keeps following the specimens; the others are set once, the surface following the window.
  function presets() {
    const list = shown();
    const pct = (q) => list.map((p) => toShared(p, histogramPercentile(tissueHistogram(p.vol.meta.histogram, p.vol.meta.masked), q)));
    const lowest = (v) => Math.min(...v), highest = (v) => Math.max(...v);
    const out = [
      ["All tissue", "From the dimmest to the brightest tissue among them; the surface follows the window's low end", () => {
        Object.assign(ui, { gamma: 1, fit: "tissue", isoAuto: true }); autoWindow();
      }],
      ["Full range", "Every value any of them holds", () => {
        ui.lo = lowest(list.map((p) => toShared(p, p.vol.q.lo))); ui.hi = highest(list.map((p) => toShared(p, p.vol.q.hi)));
      }],
    ];
    if (ui.view === "molar" && list.every((p) => p.molar?.threshold != null)) {
      out.push(["Enamel", "From the lowest enamel threshold among them to the brightest enamel; the surface at that threshold", () => {
        Object.assign(ui, { iso: null, isoAuto: false });
        ui.lo = lowest(list.map((p) => toShared(p, p.molar.threshold))); ui.hi = highest(pct(0.999));
      }]);
      if (list.every((p) => p.molar.dentin_mode != null)) {
        out.push(["Dentin to enamel", "From the lowest dentin peak among them to the brightest enamel; the surface at that peak", () => {
          Object.assign(ui, { iso: null, isoAuto: false });
          ui.lo = lowest(list.map((p) => toShared(p, p.molar.dentin_mode))); ui.hi = highest(pct(0.999));
        }]);
      }
      if (list.every((p) => p.molar.enamel_low_raw != null && p.molar.enamel_high_raw != null)) {
        const pc = list[0].molar.density_percentiles;
        const [a, b] = pc ? [ordinal(pc[0]), ordinal(pc[1])] : ["lower", "upper"];
        out.push(["Enamel core", `Every colour spent on the middle of the enamel -- from the lowest ${a} percentile among them to the highest ${b} -- with the surface on the enamel, at the lowest enamel threshold among them.`, () => {
          Object.assign(ui, { fit: "enamel", isoAuto: true }); autoWindow();
        }]);
      }
    } else {
      out.push(["Brightest 10%", "The brightest tenth of each", () => { ui.lo = lowest(pct(0.9)); ui.hi = highest(pct(0.999)); }]);
      out.push(["Middle 90%", "All but each one's dimmest and brightest 5%", () => { ui.lo = lowest(pct(0.05)); ui.hi = highest(pct(0.95)); }]);
    }
    return out;
  }

  // ------------------------------------------------------------ captions and bars
  function renderCaption(p) {
    const s = state.samples.get(p.name);
    let res = null;
    if (p.vol) {
      const sizes = p.vol.meta.spacing_preview_mm.map(um);
      res = p.vol.meta.spacing_known ? `${new Set(sizes).size === 1 ? sizes[0] : sizes.join(" × ")} µm preview` : "voxel size unknown";
    }
    replace(p.cap,
      h("div", { class: "c3d-name" }, h("i", { class: "sw", style: { background: p.color } }), h("span", { class: "specimen" }, p.name),
        s?.excluded ? h("span", { class: "c3d-flag", title: s.excluded.note || "Flagged" }, "flagged") : null),
      h("div", null, [s?.group, res].filter(Boolean).join(" · ")),
      p.r3d.alone ? h("div", { class: "c3d-alone-note" }, "Drag turns this one alone · Shift-drag moves it") : null,
      p.adjusted ? h("div", null, "Placed by hand · ", h("button", { class: "c3d-link",
        onclick: () => { p.r3d.resetModel(); p.adjusted = false; renderCaption(p); renderSide(); } }, "line it up again")) : null);
  }

  function renderTopbar() {
    const full = panels.length >= MAX_PANELS;
    replace(topbar,
      h("span", { class: "specimen" }, "Compare"),
      h("select", { "aria-label": "View", onchange: (e) => setView(e.target.value) },
        VIEW_ORDER.filter((v) => views[v]).map((v) => h("option", { value: v, selected: v === ui.view }, views[v]))),
      h("select", { "aria-label": "Add a specimen", disabled: full,
        onchange: (e) => { const name = e.target.value; e.target.value = ""; if (name) addPanel(name); } },
        h("option", { value: "" }, full ? "Four at most" : "Add a specimen…"),
        grouped().map(([group, list]) => h("optgroup", { label: group || "No group" }, list.map((s) =>
          h("option", { value: s.name, disabled: panels.some((p) => p.name === s.name) }, `${s.name}${s.excluded ? " (flagged)" : ""}`))))),
      h("span", { class: "spacer" }),
      h("label", { class: "check c3d-together", title: "Turn, move and zoom every specimen together" },
        h("input", { type: "checkbox", checked: ui.linked, onchange: (e) => setLinked(e.target.checked) }), "Turn together"),
      h("button", { class: "btn small", title: "Back to the first view (0)", disabled: !panels.some(ready), onclick: resetView }, "Reset"),
      h("button", { class: "btn small", disabled: !shown().length, onclick: saveImage }, icon("download"), "Save image"));
  }

  function renderEmpty() {
    const groups = grouped();
    replace(empty, h("div", { class: "c3d-empty-in" },
      h("h2", null, "Compare specimens in 3-D"),
      h("p", null, `Pick up to ${MAX_PANELS}. They turn together, with one heatmap scale in mg HA/cm³ across them all.`),
      groups.length ? groups.map(([group, list]) => h("div", { class: "c3d-pick" }, h("h3", null, group || "No group"),
        h("div", { class: "row wrap" }, list.map((s) => h("button", { class: "btn small", onclick: () => addPanel(s.name) },
          s.name, s.excluded ? h("span", { class: "c3d-flag" }, "flagged") : null)))))
        : h("p", null, "No specimens yet. ", h("a", { href: "#/import" }, "Import a scan"), ".")));
  }

  // ------------------------------------------------------------ side panel
  function renderSide() {
    const loaded = panels.filter(ready);
    if (!loaded.length) {
      replace(side, h("section", null, h("h3", null, "Heatmap"),
        h("div", { class: "lbl" }, panels.length ? "Loading the specimens…" : "Pick specimens to compare them.")));
      return;
    }
    const list = shown();
    replace(side, scaleSection(loaded), list.length ? heatSection(list) : null, list.length ? section3d(list) : null, specimensSection(loaded));
    if (list.length) hist.set(histState());
    syncInputs();
  }

  function scaleSection(loaded) {
    const missing = loaded.filter((p) => !p.cal).map((p) => p.name);
    const none = missing.length === loaded.length;
    return h("section", null, h("h3", null, "Scale"),
      h("div", { class: "seg", role: "group", "aria-label": "Scale" }, SCALES.map(([v, label]) =>
        h("button", { "aria-pressed": ui.scale === v ? "true" : "false", disabled: v === "mg" && none,
          title: v === "mg" && none ? "None of these scans has an HA calibration" : null, onclick: () => setScale(v) }, label))),
      h("div", { class: "lbl", style: { marginTop: "8px" } }, ui.scale === "mg"
        ? "Each specimen's values are converted with its own HA calibration, so one colour is one density in every panel."
        : "Stored values as they are: comparable only between scans made with one scanner setting and calibration."),
      none ? h("div", { class: "lbl c3d-warn" }, "None of these scans has an HA calibration. The converter copies it from a Scanco ISQ; for other scans, enter one in the Molar enamel step's Density calibration settings.")
        : missing.length ? h("div", { class: "lbl c3d-warn" }, `No HA calibration: ${missing.join(", ")}.`,
          ui.scale === "mg" ? " Left blank on this scale; Raw intensity shows every specimen." : "") : null);
  }

  function heatSection(list) {
    const unit = ui.scale === "mg" ? " (mg HA/cm³)" : "";
    const short = ui.scale === "mg" ? " mg HA/cm³" : "";
    const gammaVal = h("span", { class: "lbl num" }, ui.gamma.toFixed(2));
    const pick = (c) => {
      const was = diverging();
      ui.cmap = c;
      // Into or out of the above/below scale, the window starts again: centred, or spread over them all.
      // Above/below on the molars reads the enamel, so the surface goes onto it.
      if (was !== diverging()) { Object.assign(ui, { isoAuto: true }); ui.fit = defaultFit(); }
      if (ui.fit) autoWindow();
      // The rainbow is for cutting the scale into steps, and the random scale is nothing but steps: they arrive cut.
      if ((c === RAINBOW || c === RANDOM) && ui.steps === "off") ui.steps = DEFAULT_STEPS;
      // Random colours scatter wherever noise crosses a step's edge: the scan arrives smoothed.
      if (c === RANDOM && !ui.soften) ui.soften = RANDOM_SOFTEN;
      remember(); renderSide(); drawAll();
    };
    const widths = stepChoices(ui.lo, ui.hi);
    if (typeof ui.steps === "number" && !widths.includes(ui.steps)) widths.push(ui.steps);
    const steps = sharedSteps();
    const count = steps ? Math.round((ui.hi - ui.lo) / steps.width) : 0;
    const setSteps = (v) => { ui.steps = v; remember(); renderSide(); drawAll(); };
    return h("section", null, h("h3", null, "Heatmap"),
      h("div", { class: "cmaps", role: "group", "aria-label": "Colour scale" },
        SEQUENTIAL.map((c) => h("button", { "aria-pressed": cmap() === c ? "true" : "false", onclick: () => pick(c) },
          gradientCanvas(c, 90, 10), COLORMAPS[c].label)),
        h("button", { "aria-pressed": cmap() === RAINBOW ? "true" : "false", onclick: () => pick(RAINBOW),
          title: "Each step its own hue, blue (low) to red (high): steps tell apart however fine they are" },
          gradientCanvas(RAINBOW, 90, 10), COLORMAPS[RAINBOW].label),
        h("button", { "aria-pressed": random() ? "true" : "false", onclick: () => pick(RANDOM),
          title: "No gradient: each step a colour unrelated to its neighbours', so the pattern the steps make stands out. Read densities off the colour bar" },
          stepsCanvas(stepPalette(8, ui.seed), 90, 10), "Random"),
        h("button", { "aria-pressed": diverging() ? "true" : "false", onclick: () => pick(DIVERGING),
          title: "Blue below a density you choose, grey at it, red above: small differences either side of it show" },
          gradientCanvas(DIVERGING, 90, 10), "Above / below")),
      hist.el,
      diverging()
        ? h("div", { class: "win-inputs" },
          h("label", { class: "field" }, h("span", { class: "lbl" }, `Centre${unit}`), centreIn),
          h("label", { class: "field" }, h("span", { class: "lbl" }, "± either side"), reachIn))
        : h("div", { class: "win-inputs" },
          h("label", { class: "field" }, h("span", { class: "lbl" }, `Low${unit}`), loIn),
          h("label", { class: "field" }, h("span", { class: "lbl" }, `High${unit}`), hiIn)),
      diverging()
        ? h("div", { class: "presets" }, h("span", { class: "lbl c3d-centre" }, "Centre on"),
          h("button", { class: "btn", title: "The average of their references", onclick: () => { ui.fit = "centre"; autoWindow(); renderSide(); drawAll(); } }, "Their average"),
          references().map((r) => h("button", { class: "btn", title: `${r.name}'s ${r.what}: ${sig(r.value, 4)}${short}`, onclick: () => {
            const half = (ui.hi - ui.lo) / 2;
            Object.assign(ui, { lo: r.value - half, hi: r.value + half, fit: null });
            renderSide(); drawAll();
          } }, h("i", { class: "sw", style: { background: panels.find((x) => x.name === r.name)?.color } }), sig(r.value, 4))))
        : h("div", { class: "presets" }, presets().map(([label, title, set]) => h("button", { class: "btn", title,
          onclick: () => { ui.fit = null; set(); renderSide(); drawAll(); } }, label))),
      h("div", { class: "c3d-steps" },
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Steps"),
          h("select", { "aria-label": "Steps", onchange: (e) => {
            const v = e.target.value;
            setSteps(v === "off" || v.startsWith("count:") ? v : Number(v));
          } },
            random() ? null : h("option", { value: "off", selected: ui.steps === "off" }, "None: one smooth scale"),
            STEP_COUNTS.map((n) => {
              const w = stepWidth(ui.lo, ui.hi, n);
              return h("option", { value: `count:${n}`, selected: ui.steps === `count:${n}` }, `About ${n} (every ${numFmt(w, w)}${short})`);
            }),
            widths.sort((a, b) => a - b).map((w) => h("option", { value: w, selected: ui.steps === w }, `Every ${numFmt(w, w)}${short}`)))),
        h("label", { class: "field c3d-width" }, h("span", { class: "lbl" }, `Or every${unit}`),
          h("input", { type: "number", min: 0, step: "any", placeholder: "any width", "aria-label": "Step width",
            value: typeof ui.steps === "number" ? String(ui.steps) : "",
            onchange: (e) => { const v = Number(e.target.value); if (Number.isFinite(v) && v > 0) setSteps(v); } })),
        random()
          ? h("button", { class: "btn c3d-shuffle", title: "Deal the colours out to the steps again, in another order", onclick: () => {
            ui.seed = 1 + Math.floor(Math.random() * 999999); remember(); renderSide(); drawAll();
          } }, icon("refresh"), "Shuffle colours")
          : steps ? h("label", { class: "check lbl" }, h("input", { type: "checkbox", checked: ui.stepFill,
            onchange: (e) => { ui.stepFill = e.target.checked; remember(); renderSide(); drawAll(); } }), "One colour per step") : null),
      steps ? h("div", { class: "lbl", style: { marginTop: "6px" } },
        `${count} steps of ${numFmt(steps.width, steps.width)}${short}${diverging() ? " from the centre" : ""}, at the same densities in every panel.`,
        typeof ui.steps === "number" && steps.width > ui.steps ? ` (No finer than ${MAX_STEPS} across the window.)` : "",
        random() ? ` Each step a colour of its own, unrelated to its neighbours'${count > MOST_COLOURS ? `; past ${MOST_COLOURS} steps the colours come round again` : ""}. Grey is outside the window.` : "",
        random() && !ui.soften ? " Unsmoothed, the scan's noise scatters them: Smooth the scan (3-D, below) joins them into bands." : "",
        steps.fill && count > 7 && cmap() !== RAINBOW && !random() ? " On a one-colour scale neighbouring steps look alike past about seven; Rainbow or Random gives each its own colour." : "") : null,
      random() ? null : h("div", { class: "row", style: { marginTop: "10px" } }, h("span", { class: "lbl" }, "Gamma"),
        h("input", { type: "range", min: 0.3, max: 3, step: 0.05, value: ui.gamma, "aria-label": "Gamma",
          oninput: (e) => { ui.gamma = Number(e.target.value); gammaVal.textContent = ui.gamma.toFixed(2); hist.set(histState()); drawAll(); } }),
        gammaVal),
      h("div", { class: "lbl", style: { marginTop: "8px" } }, "One line per specimen, in its colour",
        list.some((p) => p.molar?.threshold != null) ? "; each tick is one's enamel threshold" : "", "."));
  }

  function section3d(list) {
    const unit = ui.scale === "mg" ? " (mg HA/cm³)" : "";
    const structures = new Map();
    for (const p of list) for (const o of p.vol.meta.overlays) structures.set(o.name, o.label);
    const tinting = [...structures.keys()].some((n) => ui.structures[n]);
    const surface = ui.style === "surface";
    const voxel = voxelMm() || 0, deep = sharedDeep() || 0, r = smoothMm();
    const shadeVal = h("span", { class: "lbl num" }, ui.shade.toFixed(2));
    return h("section", null, h("h3", null, "3-D"),
      h("div", { class: "seg", role: "group", "aria-label": "3-D style" },
        [["surface", "Surface"], ["xray", "X-ray"]].map(([v, label]) => h("button", { "aria-pressed": ui.style === v ? "true" : "false",
          onclick: () => { ui.style = v; remember(); renderSide(); drawAll(); } }, label))),
      surface ? h("div", { class: "field c3d-level" }, h("span", { class: "lbl" }, `Surface at${unit}`),
        h("div", { class: "row" },
          h("input", { type: "number", step: "any", "aria-label": "Surface level", value: plain(ui.iso ?? ui.lo),
            onchange: (e) => { const v = Number(e.target.value); if (Number.isFinite(v)) Object.assign(ui, { iso: v, isoAuto: false }); renderSide(); drawAll(); } }),
          h("button", { class: "btn small", disabled: ui.iso == null, onclick: () => { Object.assign(ui, { iso: null, isoAuto: false }); renderSide(); drawAll(); } },
            "Follow the window"))) : null,
      h("div", { class: "field c3d-level" }, h("span", { class: "lbl" }, "Smooth the scan"),
        h("div", { class: "seg", role: "group", "aria-label": "Smooth the scan",
          title: "Blur the scan in 3-D (a Gaussian this wide) before its surface and colours are read: steadier colours and a smoother surface, finer detail lost" },
        SOFTEN.map((n) => h("button", { "aria-pressed": ui.soften === n ? "true" : "false", onclick: () => { ui.soften = n; remember(); renderSide(); drawAll(); } },
          n ? `${um(n * voxel)} µm` : "As scanned")))),
      surface ? h("div", { class: "field c3d-level" }, h("span", { class: "lbl" }, "Average the colour over"),
        h("div", { class: "seg", role: "group", "aria-label": "Average the colour over" }, SMOOTHING.map((n) =>
          h("button", { "aria-pressed": ui.smooth === n ? "true" : "false", onclick: () => { ui.smooth = n; remember(); renderSide(); drawAll(); } },
            n ? `${um(n * voxel)} µm` : "One point")))) : null,
      surface ? h("div", { class: "row", style: { marginTop: "10px" } }, h("span", { class: "lbl" }, "Shading"),
        h("input", { type: "range", min: 0, max: 1, step: 0.05, value: ui.shade, "aria-label": "Shading",
          oninput: (e) => { ui.shade = Number(e.target.value); shadeVal.textContent = ui.shade.toFixed(2); drawAll(); }, onchange: remember }),
        shadeVal) : null,
      h("div", { class: "row", style: { marginTop: "10px" } }, h("span", { class: "lbl" }, "Cut away"),
        h("input", { type: "range", min: 0, max: 0.95, step: 0.01, value: ui.cut, "aria-label": "Cut away",
          oninput: (e) => { ui.cut = Number(e.target.value); drawAll(); } })),
      structures.size && surface ? h("div", { class: "c3d-structs" },
        h("span", { class: "lbl" }, "Tint what a structure covers"),
        [...structures].map(([name, label]) => h("label", { class: "check lbl" },
          h("input", { type: "checkbox", checked: Boolean(ui.structures[name]),
            onchange: (e) => { ui.structures[name] = e.target.checked; renderSide(); drawAll(); } }),
          h("i", { class: "sw", style: { background: OVERLAY_COLORS[name] || "#ffffff" } }), label)),
        tinting ? h("div", { class: "row" }, h("span", { class: "lbl" }, "Strength"),
          h("input", { type: "range", min: 0, max: 1, step: 0.05, value: ui.tint, "aria-label": "Tint strength",
            oninput: (e) => { ui.tint = Number(e.target.value); drawAll(); } })) : null) : null,
      h("div", { class: "row wrap", style: { marginTop: "10px" } },
        SIDES.map(([v, label]) => h("button", { class: "btn small", onclick: () => { ui.side = v; fitted = true; fitAll(v); } }, label))),
      h("div", { class: "field c3d-level" }, h("span", { class: "lbl" }, "Place them"),
        h("div", { class: "seg", role: "group", "aria-label": "Place them" }, ALIGNS.map(([v, label]) =>
          h("button", { "aria-pressed": ui.align === v ? "true" : "false", onclick: () => setAlign(v) }, label)))),
      h("div", { class: "lbl", style: { marginTop: "6px" } }, ui.align === "match"
        ? `Each turns about its own centre. The first sits as scanned; each other one is turned and moved to where its surface lies closest to the first one's.`
        : "Each turns about its own centre, sitting as it does in its scan."),
      h("div", { class: "row wrap", style: { marginTop: "8px" } },
        h("button", { class: "btn small", disabled: !panels.some((p) => p.adjusted), onclick: lineUp,
          title: "Undo every turn and move made to one specimen alone" }, "Line them up again")),
      h("div", { class: "lbl", style: { marginTop: "8px" } }, surface
        ? ["Each ray stops at the first voxel as dense as the surface level. Its colour is ",
          r ? `the mean density from ${um(deep)} to ${um(deep + 2 * r)} µm beneath it, within ${um(r)} µm around`
            : `the density ${um(deep)} µm beneath it`,
          ", straight in: the same in every panel.",
          r ? " Averaging quiets the voxel noise that would hide small differences." : " One point is as noisy as a voxel."]
        : "The densest value along each ray, on the heatmap's scale."),
      surface ? h("div", { class: "lbl", style: { marginTop: "6px" } },
        "Light shows the shape but moves colours off the scale; less shading keeps them true to it.") : null,
      h("div", { class: "lbl", style: { marginTop: "6px" } }, ui.linked
        ? "Drag any specimen to turn them all, all the way round."
        : "Each specimen turns on its own, all the way round.",
        ` Shift-drag or right-drag moves, ${ROLL}-drag rolls, scroll zooms, double-click resets, arrow keys turn.`,
        ` Hold ${ALT} (or press a panel's move button) and the same drags turn, move or roll that specimen alone.`));
  }

  function specimensSection(loaded) {
    const res = new Set(loaded.map((p) => um(Math.min(...p.vol.meta.spacing_preview_mm))));
    return h("section", null, h("h3", null, "Specimens"),
      panels.map((p) => {
        const cal = p.cal;
        const lines = [];
        if (p.vol) {
          lines.push(cal ? `Calibration from ${cal.source ? basename(cal.source) : "the Molar enamel settings"}${cal.description ? ` (${cal.description})` : ""}: ${calibrationFormula(cal)}.`
            : "No HA calibration for this scan.");
          if (onScale(p) && ui.scale === "mg") lines.push(`The window is ${sig(fromShared(p, ui.lo), 4)} to ${sig(fromShared(p, ui.hi), 4)} in its stored values.`);
          if (ready(p)) lines.push(placement(p, loaded));
        }
        return h("div", { class: "c3d-spec" },
          h("div", { class: "row" }, h("i", { class: "sw", style: { background: p.color } }),
            h("a", { class: "specimen", href: `#/sample/${enc(p.name)}` }, p.name), h("span", { class: "spacer" }),
            h("a", { class: "btn small ghost icon", href: `#/viewer/${enc(p.name)}/${ui.view}?plane=3d`,
              title: "Open in the viewer", "aria-label": `Open ${p.name} in the viewer` }, icon("viewer"))),
          h("div", { class: "lbl" }, state.samples.get(p.name)?.group || ""),
          lines.map((t) => h("div", { class: "lbl" }, t)));
      }),
      res.size > 1 ? h("div", { class: "lbl c3d-warn" }, `The previews differ in resolution (${[...res].join(" and ")} µm). `
        + "A coarser preview averages thin enamel with what surrounds it, which lowers its density and softens its surface. "
        + "The sharpest previews (Settings, Viewer previews) and a rebuild of each bring them level.") : null);
  }

  // Where a specimen sits, in a sentence.
  function placement(p, loaded) {
    if (p.adjusted) return "Placed by hand.";
    if (ui.align !== "match" || loaded.length < 2) return "As scanned, about its tissue's centre.";
    if (p === loaded[0]) return "As scanned, about its tissue's centre; the others are fitted to it.";
    if (!p.fit) return "As scanned: it has no tissue to fit by.";
    const mm = p.fit.moved;
    return `Turned ${deg(p.fit.turned)}° and moved ${mm < 1 ? `${um(mm)} µm` : `${sig(mm, 3)} mm`} to fit ${p.fit.to}.`;
  }

  // ------------------------------------------------------------ saving
  // The panels as they are laid out, each named, over the colour bar and its units.
  function saveImage() {
    const list = shown();
    if (!list.length) return;
    for (const p of list) { p.r3d.moving = false; p.r3d.drawNow(); }
    const first = list[0].canvas;
    const k = first.width / Math.max(first.clientWidth, 1);
    const cw = first.width, ch = first.height, rows = Math.ceil(panels.length / cols), foot = Math.round(64 * k);
    const out = document.createElement("canvas");
    out.width = cols * cw;
    out.height = rows * ch + foot;
    const c = out.getContext("2d");
    c.fillStyle = STAGE;
    c.fillRect(0, 0, out.width, out.height);
    const face = getComputedStyle(document.documentElement).getPropertyValue("--font-label").trim() || "sans-serif";
    panels.forEach((p, i) => {
      const x = (i % cols) * cw, y = Math.floor(i / cols) * ch;
      if (list.includes(p)) c.drawImage(p.canvas, x, y, cw, ch);
      c.fillStyle = p.color;
      c.fillRect(x + 12 * k, y + 14 * k, 10 * k, 10 * k);
      c.textAlign = "left";
      c.textBaseline = "top";
      c.fillStyle = "#e2e9e7";
      c.font = `600 ${16 * k}px ${face}`;
      c.fillText(p.name, x + 28 * k, y + 10 * k);
      c.fillStyle = "#a3aeac";
      c.font = `${12 * k}px system-ui, sans-serif`;
      c.fillText(list.includes(p) ? state.samples.get(p.name)?.group || "" : "No HA calibration: not on this scale", x + 28 * k, y + 31 * k);
    });
    c.fillStyle = "#2c3436";
    for (let i = 1; i < cols; i++) c.fillRect(i * cw - k / 2, 0, k, rows * ch);
    for (let j = 1; j < rows; j++) c.fillRect(0, j * ch - k / 2, out.width, k);
    const barW = Math.min(360 * k, out.width * 0.45);
    colorbar(c, 24 * k, rows * ch + 14 * k, barW, 12 * k, scaleState(), colourOf, "#a3aeac", `${11 * k}px system-ui, sans-serif`);
    c.textAlign = "left";
    c.textBaseline = "top";
    c.fillStyle = "#e2e9e7";
    c.font = `${12.5 * k}px system-ui, sans-serif`;
    const steps = sharedSteps(), r = smoothMm();
    const note = [`${views[ui.view] || ui.view}: colour in ${unitText()}`,
      diverging() ? `centred on ${sig((ui.lo + ui.hi) / 2, 4)}` : null,
      steps ? `steps of ${numFmt(steps.width, steps.width)}` : null,
      random() ? "each its own colour, grey outside the window" : null,
      ui.style === "surface" ? `surface at ${sig(ui.iso ?? ui.lo, 4)}` : "densest along each ray",
      ui.style === "surface" ? (r ? `mean ${um(sharedDeep())}-${um(sharedDeep() + 2 * r)} µm beneath` : `${um(sharedDeep())} µm beneath`) : null,
      ui.soften ? `scan smoothed (sigma ${um(softenMm())} µm)` : null];
    c.fillText(note.filter(Boolean).join(", "), 48 * k + barW, rows * ch + 12 * k);
    out.toBlob((blob) => {
      const a = h("a", { href: URL.createObjectURL(blob), download: `compare_${ui.view}_${panels.map((p) => p.name).join("_")}.png` });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    });
  }

  // ------------------------------------------------------------ keys and events
  const onKey = (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
    const loaded = panels.filter(ready);
    if (!loaded.length) return;
    const turn = { ArrowLeft: [-12, 0], ArrowRight: [12, 0], ArrowUp: [0, -12], ArrowDown: [0, 12] }[e.key];
    if (turn) {
      e.preventDefault();
      if (ui.linked) (loaded.includes(lastMoved) ? lastMoved : loaded[0]).r3d.nudge(...turn);
      else loaded.forEach((p) => p.r3d.nudge(...turn));
    } else if (e.key === "0" || e.key.toLowerCase() === "f") {
      e.preventDefault();
      resetView();
    }
  };
  window.addEventListener("keydown", onKey);

  const offJobs = on("jobs", (job) => {
    if (!job || job.step !== "preview" || job.params?.view !== ui.view) return;
    const p = panels.find((x) => x.name === job.sample);
    if (!p || p.vol) return;
    if (job.status === "done") openPanel(p);
    else if (["running", "queued", "waiting"].includes(job.status)) {
      const fill = p.msg.querySelector(".progress i");
      if (fill) fill.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    } else if (job.status === "failed") show(p, `The preview could not be built: ${job.error || "see the queue for the log"}`);
  });
  const offSamples = on("samples", () => {
    for (const p of [...panels]) if (!state.samples.has(p.name)) removePanel(p);
    renderTopbar();
    panels.forEach(renderCaption);
    if (!panels.length) renderEmpty();
  });

  // ------------------------------------------------------------ start
  for (const name of wanted) addPanel(name);
  if (!panels.length) arranged();
  renderSide();
  drawLegend();
  return () => {
    ro.disconnect();
    window.removeEventListener("keydown", onKey);
    offJobs();
    offSamples();
    for (const p of panels) { p.token++; p.r3d.dispose(); }
  };
}
