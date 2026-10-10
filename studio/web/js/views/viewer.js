// The viewer: any of a specimen's volumes as an adjustable heatmap, on a dark
// lightbox. Slices, projections and the unrolled incisor are computed in the
// browser from the preview; changing the colour scale is instant.

import { h, icon, replace, sig, fmtNum, errorToast, toast, rafThrottle, basename, relTime, slotColor } from "../lib.js";
import { api, enc } from "../api.js";
import { state, on, previewJob, libraryFolders } from "../store.js";
import { folderCrumbs, parentOf, leafOf, inFolder, track, groupLabel, flagTag } from "../components.js";
import { COLORMAPS, SEQUENTIAL, buildLut, table, gradientCanvas } from "../colormaps.js";
import { loadVolume, histogramPercentile, interp, tissueHistogram as tissueOf } from "../volume.js";
import { histWindow } from "../histwin.js";
import { Render3D } from "../render3d.js";
import { go } from "../app.js";

const VIEW_ORDER = ["mandible", "incisor", "molar", "cmpr", "bone", "converted", "source"];
export const DEFAULT_CMAP = { mandible: "gray", source: "gray", converted: "gray", incisor: "mineral", molar: "mineral", bone: "mineral", cmpr: "magma" };
// Structures keep one colour everywhere (dark-stage steps of the categorical palette).
export const OVERLAY_COLORS = {
  incisor: "#d95926", bone: "#3987e5", molar: "#199e70", enamel: "#e8b400",
  ml_incisor: "#d95926", ml_bone: "#3987e5", ml_molar: "#199e70",
  // The molar step's teeth, in their landmark colours.
  m1: "#1aa34a", m2: "#9085e9", m3: "#d55181",
};
const LANDMARKS = [
  { name: "eruption", label: "Eruption", keys: ["q"], kind: "event", color: "#e8b400" },
  { name: "M1", label: "M1", keys: ["g", "h"], kind: "span", color: "#1aa34a" },
  { name: "M2", label: "M2", keys: ["j", "k"], kind: "span", color: "#9085e9" },
  { name: "M3", label: "M3", keys: ["n", "w"], kind: "span", color: "#d55181" },
];

// A mark the Landmarks step placed ("teeth": the molar step's, before it did), and how it found each.
const isAuto = (m) => m?.source === "auto" || m?.source === "teeth";
const AUTO_HOW = {
  eruption: "the first slice toward the tip where no ray from the incisor's enamel touches bone",
  M1: "where its crown runs, from the tooth split", M2: "where its crown runs, from the tooth split",
  M3: "where its crown runs, from the tooth split",
};

const ROLL = /Mac/.test(navigator.platform) ? "⌘" : "Ctrl";
const hexRgb = (hex) => { const v = parseInt(hex.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };

export async function mount(root, [name, viewParam], query) {
  if (!name) return pickSample(root, query);

  let views = [];
  try {
    views = await api.get(`/api/samples/${enc(name)}/views`);
  } catch (err) {
    root.appendChild(h("div", { class: "page" }, h("div", { class: "notice critical" }, err.message)));
    return;
  }
  const view = viewParam || (VIEW_ORDER.find((v) => views.some((x) => x.view === v && x.built)) ||
    VIEW_ORDER.find((v) => views.some((x) => x.view === v)));

  const st = {
    name, view, vol: null, plane: query.plane === "3d" ? 4 : 0, idx: [0, 0, 0], proj: "slice", slab: 9,
    // 3-D (plane 4): the surface level (null follows the window's low end), style, cut-away, structure tint.
    iso: null, style3d: "surface", cut: 0, tint: 0.7, turn: query.turn || null,
    cmap: DEFAULT_CMAP[view] || "mineral", lo: 0, hi: 1, gamma: 1, smooth: false,
    ratio: false, rlo: 0.5, rhi: 1.8, ov: {}, zoom: 1, panX: 0, panY: 0,
    mode: query.mode === "landmarks" ? "landmarks" : null, marks: {}, mapped: [],
    hover: null, profile: null, molar: null, lutKey: "", lut: null,
  };

  // ------------------------------------------------------------ layout
  const shell = h("div", { class: "viewer" });
  root.appendChild(shell);
  const topbar = h("div", { class: "topbar" });
  const stage = h("div", { class: "stage", tabindex: 0, "aria-label": "Image. Arrow keys change the slice." });
  const canvas = h("canvas", { class: "main" });
  const glCanvas = h("canvas", { class: "gl3d", hidden: true, tabindex: 0,
    "aria-label": `3-D view. Drag to turn all the way round, shift-drag to move, ${ROLL}-drag to roll, scroll to zoom, arrow keys to turn.` });
  let r3d = null;
  const readout = h("div", { class: "readout" });
  const scalebar = h("div", { class: "scalebar" });
  const colorbar = h("div", { class: "landmark-hint" });
  stage.append(canvas, glCanvas, readout, scalebar, colorbar);
  const slider = h("input", { type: "range", min: 0, max: 0, value: 0, "aria-label": "Slice" });
  const sliceNum = h("span", { class: "num" });
  const slicebar = h("div", { class: "slicebar" }, slider, sliceNum);
  const side = h("aside", { class: "side", "aria-label": "Display settings" });
  shell.append(h("div", { class: "stage-wrap" }, topbar, stage, slicebar), side);
  const ctx = canvas.getContext("2d");
  const off = document.createElement("canvas");
  const offCtx = off.getContext("2d");

  // ------------------------------------------------------------ loading
  async function openView() {
    const info = views.find((v) => v.view === st.view);
    renderTopbar();
    if (!info) {
      showMessage(`${name} has no ${st.view} view.`, null);
      return;
    }
    if (!info.built || !info.current) {
      const job = previewJob(name, st.view);
      if (!job) {
        try { await api.post(`/api/samples/${enc(name)}/views/${st.view}/build`, { force: info.built && !info.current }); }
        catch (err) { errorToast(err); }
      }
      showMessage(info.built ? "The preview is out of date; rebuilding it." : "Building the preview. Large scans take a minute.", 0);
      return;
    }
    showMessage("Loading…", null);
    st.ready = false;
    try {
      st.vol = await loadVolume(name, st.view);
    } catch (err) {
      showMessage(err.message, null);
      return;
    }
    const vol = st.vol;
    st.idx = [Math.floor(vol.Z / 2), Math.floor(vol.Y / 2), Math.floor(vol.X / 2)];
    if (st.plane === 3 && !vol.unwrap) st.plane = 0;
    st.iso = null;
    st.ov = {};
    for (const o of vol.meta.overlays) {
      // ML masks share their structure's colour, so they default to outlines:
      // turned on over the pipeline's fills, the two segmentations stay apart.
      const ml = o.name.startsWith("ml_");
      st.ov[o.name] = { on: !ml, alpha: 0.4, style: st.view === "mandible" && !ml ? "fill" : "outline" };
    }
    if (st.view === "cmpr") {
      try {
        const inc = await api.get(`/api/samples/${enc(name)}/incisor`);
        const p = inc.profile;
        if (p) {
          const rows = p.z.map((z, i) => [z, p.dentin[i], p.z_percent[i]]).filter((r) => r[0] != null).sort((a, b) => a[0] - b[0]);
          const withDentin = rows.filter((r) => r[1] != null && r[1] > 0);
          st.profile = { z: rows.map((r) => r[0]), zp: rows.map((r) => r[2]), dz: withDentin.map((r) => r[0]), dentin: withDentin.map((r) => r[1]) };
          if (!st.profile.dz.length) st.profile.dentin = null;
        }
      } catch { st.profile = null; }
    }
    if (st.view === "molar") {
      try { st.molar = (await api.get(`/api/samples/${enc(name)}/molar`)).summary; } catch { st.molar = null; }
    }
    if (st.view === "mandible") {
      try {
        const m = await api.get(`/api/samples/${enc(name)}/marks`);
        st.marks = {};
        for (const mk of m.marks) st.marks[mk.landmark] = { start: mk.z_start, end: mk.z_end, source: mk.source };
        st.mapped = m.mapped || [];
        st.auto = m.auto || [];
        st.markWarnings = m.warnings || [];
      } catch { /* none yet */ }
    }
    autoWindow();
    stage.querySelectorAll(".loading").forEach((n) => n.remove());
    st.ready = true;
    renderTopbar();
    renderSide();
    fit();
    draw();
  }

  function showMessage(text, progress) {
    st.vol = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stage.querySelectorAll(".loading").forEach((n) => n.remove());
    stage.appendChild(h("div", { class: "loading" }, h("div", null, text,
      progress != null ? h("div", { class: "progress" }, h("i", { style: { width: `${Math.round(progress * 100)}%` } })) : null)));
    replace(readout); replace(scalebar); replace(colorbar);
    replace(side, sideViews());
  }

  // ------------------------------------------------------------ window
  const tissueHistogram = () => tissueOf(st.vol.meta.histogram, st.vol.meta.masked);

  function autoWindow() {
    const h2 = tissueHistogram();
    st.lo = histogramPercentile(h2, 0.01);
    st.hi = histogramPercentile(h2, 0.998);
    if (st.hi <= st.lo) { st.lo = st.vol.q.lo; st.hi = st.vol.q.hi; }
    st.gamma = 1;
  }

  function presets() {
    const hist = tissueHistogram();
    const p = (q) => histogramPercentile(hist, q);
    const list = [["Auto", () => autoWindow()], ["Full range", () => { st.lo = st.vol.q.lo; st.hi = st.vol.q.hi; }]];
    if (st.molar) {
      list.push(["Enamel", () => { st.lo = st.molar.threshold; st.hi = p(0.999); }]);
      if (st.molar.dentin_mode != null) list.push(["Dentin to enamel", () => { st.lo = st.molar.dentin_mode; st.hi = p(0.999); }]);
    } else {
      list.push(["Brightest 10%", () => { st.lo = p(0.9); st.hi = p(0.999); }]);
      list.push(["Middle 90%", () => { st.lo = p(0.05); st.hi = p(0.95); }]);
    }
    return list;
  }

  // ------------------------------------------------------------ geometry
  let W = 0, H = 0;
  const layout = { dx: 0, dy: 0, dw: 0, dh: 0, w: 1, h: 1, mmPerPx: null };

  function fit() { st.zoom = 1; st.panX = 0; st.panY = 0; }

  function resize() {
    const r = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    W = r.width; H = r.height;
    canvas.width = Math.max(1, Math.round(W * dpr));
    canvas.height = Math.max(1, Math.round(H * dpr));
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(stage);

  // ------------------------------------------------------------ image
  function currentImage() {
    const vol = st.vol;
    if (st.plane === 3) {
      const [depth, angles] = vol.meta.unwrap.shape;
      return { img: vol.unwrap, w: angles, h: depth, zOfRow: (r) => r, unwrap: true };
    }
    const [w, hh] = vol.dims(st.plane);
    const img = vol.image(st.plane, st.idx[st.plane], st.proj, st.slab);
    const zConst = vol.fullZ(st.idx[0]);
    return { img, w, h: hh, zOfRow: st.plane === 0 ? () => zConst : (r) => vol.fullZ(r), unwrap: false };
  }

  function dentinAt(zFull) {
    const p = st.profile;
    if (!p || !p.dentin) return null;
    return interp(p.dz, p.dentin, zFull);
  }

  function lutFor() {
    const key = `${st.cmap}|${st.lo}|${st.hi}|${st.gamma}|${st.vol.meta.masked}|${st.vol.meta.base.mtime}`;
    if (key !== st.lutKey) {
      st.lut = buildLut(st.cmap, st.vol.q, st.lo, st.hi, st.gamma, st.vol.meta.masked);
      st.lutKey = key;
    }
    return st.lut;
  }

  const draw = rafThrottle(() => {
    if (!st.vol || !W) return;
    // 3-D waits for the view to finish opening: its window and structures set, nothing still loading.
    if (st.plane === 4) { if (st.ready) draw3d(); return; }
    canvas.hidden = false;
    glCanvas.hidden = true;
    const vol = st.vol;
    const { img, w, h: hh, zOfRow, unwrap } = currentImage();
    if (off.width !== w || off.height !== hh) { off.width = w; off.height = hh; }
    const image = offCtx.createImageData(w, hh);
    const buf32 = new Uint32Array(image.data.buffer);
    const px = image.data;
    const ratioOn = st.ratio && st.profile?.dentin;
    if (ratioOn) {
      const tab = table("ratio");
      const span = st.rhi - st.rlo || 1;
      for (let r = 0; r < hh; r++) {
        const ref = dentinAt(zOfRow(r));
        for (let c = 0; c < w; c++) {
          const i = r * w + c, q = img[i];
          if (!q || !ref) continue;
          const t = Math.max(0, Math.min(1, (vol.value(q) / ref - st.rlo) / span));
          const k = Math.round(t * 255) * 3;
          px[i * 4] = tab[k]; px[i * 4 + 1] = tab[k + 1]; px[i * 4 + 2] = tab[k + 2]; px[i * 4 + 3] = 255;
        }
      }
    } else {
      const lut = lutFor();
      for (let i = 0; i < w * hh; i++) buf32[i] = lut[img[i]];
    }
    if (!unwrap) {
      for (const o of vol.meta.overlays) {
        const cfg = st.ov[o.name];
        if (!cfg || !cfg.on) continue;
        const ov = vol.overlayImage(o.name, st.plane, st.idx[st.plane], st.proj, st.slab);
        if (!ov) continue;
        const [cr, cg, cb] = hexRgb(OVERLAY_COLORS[o.name] || "#ffffff");
        if (cfg.style === "fill") {
          for (let i = 0; i < w * hh; i++) {
            const m = ov[i];
            if (!m) continue;
            const a = cfg.alpha * (m / 255);
            const j = i * 4;
            if (px[j + 3]) {
              px[j] = px[j] * (1 - a) + cr * a;
              px[j + 1] = px[j + 1] * (1 - a) + cg * a;
              px[j + 2] = px[j + 2] * (1 - a) + cb * a;
            } else {
              // Over empty background the colour sits on the stage itself.
              px[j] = cr; px[j + 1] = cg; px[j + 2] = cb; px[j + 3] = Math.round(255 * a);
            }
          }
        } else {
          for (let r = 0; r < hh; r++) {
            for (let c = 0; c < w; c++) {
              const i = r * w + c;
              if (ov[i] < 128) continue;
              const edge = c === 0 || c === w - 1 || r === 0 || r === hh - 1 ||
                ov[i - 1] < 128 || ov[i + 1] < 128 || ov[i - w] < 128 || ov[i + w] < 128;
              if (!edge) continue;
              const j = i * 4;
              px[j] = cr; px[j + 1] = cg; px[j + 2] = cb; px[j + 3] = 255;
            }
          }
        }
      }
    }
    offCtx.putImageData(image, 0, 0);

    // Placement: true physical aspect for slices; the unrolled map fills the stage.
    let dw, dh;
    if (unwrap) {
      dw = Math.min(W * 0.86, 1100) * st.zoom;
      dh = H * 0.8 * st.zoom;
      layout.mmPerPx = null;
    } else {
      const [sx, sy] = vol.pixelMm(st.plane);
      const k = Math.min((W * 0.94) / (w * sx), (H * 0.94) / (hh * sy)) * st.zoom;
      dw = w * sx * k; dh = hh * sy * k;
      layout.mmPerPx = 1 / k;
    }
    // The unrolled map keeps a band below it for its angle labels.
    const dx = (W - dw) / 2 + st.panX, dy = (H - dh) / 2 + st.panY - (unwrap ? 18 : 0);
    Object.assign(layout, { dx, dy, dw, dh, w, h: hh });
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = st.smooth;
    ctx.drawImage(off, dx, dy, dw, dh);
    drawAnnotations(unwrap);
    drawScale();
    drawReadout();
    slider.max = String(unwrap ? 0 : vol.depth(st.plane) - 1);
    slider.value = String(unwrap ? 0 : st.idx[st.plane]);
    slider.disabled = unwrap || st.proj === "max" || st.proj === "mean";
    sliceNum.textContent = unwrap ? `${vol.meta.unwrap.shape[0]} slices × ${vol.meta.unwrap.shape[1]}°`
      : (st.proj === "max" || st.proj === "mean") ? `${st.proj === "max" ? "Maximum" : "Mean"} through all ${vol.depth(st.plane)}`
        : `Slice ${st.idx[st.plane] + 1} of ${vol.depth(st.plane)}`;
  });

  // 3-D: the same preview, colour scale, window and structures, ray-marched on the GPU.
  function draw3d() {
    const vol = st.vol;
    if (!r3d) r3d = new Render3D(glCanvas);
    canvas.hidden = true;
    glCanvas.hidden = false;
    replace(scalebar);
    replace(colorbar);
    slider.disabled = true;
    sliceNum.textContent = `Drag to turn · Shift-drag to move · ${ROLL}-drag to roll · Scroll to zoom · Double-click to reset`;
    if (!r3d.ok) { replace(readout, r3d.error); return; }
    r3d.setVolume(vol);
    if (st.turn) { r3d.setView(st.turn); st.turn = null; }   // ?turn=top in the link: open on that side
    const level = st.iso ?? st.lo;
    r3d.setOptions({
      style: st.style3d, level, lo: st.lo, hi: st.hi, gamma: st.gamma, cmap: st.cmap, cut: st.cut, tint: st.tint,
      overlays: vol.meta.overlays.map((o) => ({ on: !!st.ov[o.name]?.on, color: OVERLAY_COLORS[o.name] || "#ffffff" })),
    });
    r3d.drawNow();   // draw() already runs once per animation frame
    replace(readout, st.style3d === "xray" ? "X-ray: the brightest value along each ray"
      : h("span", null, "Surface at ", h("b", null, sig(level, 4)), st.iso == null ? " (the window's low end)" : ""));
  }

  function setPlane(i) {
    st.plane = i;
    st.hover = null;
    renderTopbar();
    renderSide();
    draw();
  }

  function drawAnnotations(unwrap) {
    const { dx, dy, dw, dh, h: hh } = layout;
    ctx.font = "11.5px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    if (unwrap) {
      ctx.fillStyle = "rgba(226,233,231,0.8)";
      ctx.textAlign = "center";
      for (const deg of [0, 90, 180, 270, 360]) ctx.fillText(`${deg}°`, dx + (deg / 360) * dw, dy + dh + 12);
      const p = st.profile;
      if (p && p.z.length) {
        ctx.textAlign = "right";
        for (const [row, label] of [[0, "top"], [hh - 1, "bottom"]]) {
          const zp = interp(p.z, p.zp, row);
          ctx.fillText(zp != null ? `${Math.round(zp)}%` : "", dx - 8, dy + ((row + 0.5) / hh) * dh);
        }
        ctx.save();
        ctx.translate(dx - 34, dy + dh / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillText("Along the tooth (% from cervical loop)", 0, 0);
        ctx.restore();
      }
      return;
    }
    if (st.view === "mandible" && (st.plane === 1 || st.plane === 2)) {
      for (const lm of LANDMARKS) {
        const m = st.marks[lm.name];
        if (!m) continue;
        for (const [z, which] of [[m.start, lm.kind === "event" ? "" : " start"], [m.end, " end"]]) {
          if (z == null) continue;
          const row = st.vol.fromFullZ(z) + 0.5;
          const y = dy + (row / hh) * dh;
          ctx.strokeStyle = lm.color;
          ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(dx, y); ctx.lineTo(dx + dw, y); ctx.stroke();
          ctx.fillStyle = lm.color;
          ctx.textAlign = "left";
          ctx.fillText(`${lm.label}${which}`, dx + dw + 6, y);
        }
      }
    }
    if (st.hover && st.mode === "landmarks" && (st.plane === 1 || st.plane === 2)) {
      const y = dy + ((st.hover.row + 0.5) / hh) * dh;
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(dx, y); ctx.lineTo(dx + dw, y); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function drawScale() {
    if (!layout.mmPerPx || !st.vol.meta.spacing_known) { replace(scalebar); return; }
    const target = layout.mmPerPx * 110;
    const nice = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20].reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a));
    const px = nice / layout.mmPerPx;
    replace(scalebar, h("i", { style: { width: `${px}px` } }), nice < 1 ? `${Math.round(nice * 1000)} µm` : `${nice} mm`);
  }

  function drawReadout() {
    const vol = st.vol;
    const lines = [];
    const hv = st.hover;
    if (hv) {
      if (st.plane === 3) {
        lines.push(h("div", null, `Angle `, h("b", null, `${Math.round((hv.col / layout.w) * 360)}°`), `  slice `, h("b", null, String(hv.row))));
        const zp = st.profile ? interp(st.profile.z, st.profile.zp, hv.row) : null;
        if (zp != null) lines.push(h("div", null, "Position ", h("b", null, `${zp.toFixed(1)}%`), " along the tooth"));
      } else {
        const [z, y, x] = hv.vox;
        const [fz, fy, fx] = vol.toFull(z, y, x);
        lines.push(h("div", null, "Voxel ", h("b", null, `${fz}, ${fy}, ${fx}`), " (z, y, x)"));
        if (vol.meta.spacing_known) {
          const [sz, sy, sx] = vol.meta.spacing_mm;
          lines.push(h("div", null, h("b", null, `${(fz * sz).toFixed(3)}, ${(fy * sy).toFixed(3)}, ${(fx * sx).toFixed(3)}`), " mm"));
        }
        if (st.view === "cmpr" && st.profile) {
          const zp = interp(st.profile.z, st.profile.zp, fz);
          if (zp != null) lines.push(h("div", null, "Position ", h("b", null, `${zp.toFixed(1)}%`), " along the tooth"));
        }
      }
      if (hv.value != null) {
        lines.push(h("div", null, "Intensity ", h("b", null, fmtNum(hv.value, Math.abs(hv.value) < 10 ? 3 : 0))));
        if (st.ratio && hv.ratio != null) lines.push(h("div", null, "Enamel/dentin ", h("b", null, hv.ratio.toFixed(2))));
      } else lines.push(h("div", null, "Background"));
      for (const o of vol.meta.overlays) if (hv.ov?.[o.name]) lines.push(h("div", null, `Inside ${o.label}`));
    } else {
      lines.push(h("div", null, h("b", null, views.find((v) => v.view === st.view)?.label || st.view)));
      lines.push(h("div", null, `Preview ${vol.meta.shape.join(" × ")}, ${vol.meta.factors.every((f) => f === 1) ? "full resolution" : `1/${vol.meta.factors.join(", 1/")} of full`}`));
    }
    replace(readout, lines);
    const [lo, hi] = st.ratio ? [st.rlo, st.rhi] : [st.lo, st.hi];
    const grad = gradientCanvas(st.ratio ? "ratio" : st.cmap, 140, 8);
    grad.style.cssText = "width:140px;height:8px;display:block;border-radius:2px;margin-bottom:3px";
    replace(colorbar, grad, `${sig(lo, 4)} to ${sig(hi, 4)}${st.ratio ? " × dentin" : ""}`,
      st.mode === "landmarks" ? h("div", { style: { marginTop: "6px" } }, "Press a landmark key on a slice to mark it; U undoes.") : null);
  }

  // ------------------------------------------------------------ input
  function pick(e) {
    const r = stage.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const col = Math.floor(((mx - layout.dx) / layout.dw) * layout.w);
    const row = Math.floor(((my - layout.dy) / layout.dh) * layout.h);
    if (col < 0 || row < 0 || col >= layout.w || row >= layout.h) return null;
    const vol = st.vol;
    const { img, zOfRow } = currentImage();
    const q = img[row * layout.w + col];
    const value = vol.value(q);
    let vox = null;
    const ov = {};
    if (st.plane !== 3) {
      vox = st.plane === 0 ? [st.idx[0], row, col] : st.plane === 1 ? [row, st.idx[1], col] : [row, col, st.idx[2]];
      for (const o of vol.meta.overlays) {
        const a = vol.overlayImage(o.name, st.plane, st.idx[st.plane], st.proj, st.slab);
        if (a && a[row * layout.w + col] >= 128) ov[o.name] = true;
      }
    }
    const ref = st.ratio ? dentinAt(zOfRow(row)) : null;
    return { col, row, vox, value, ratio: value != null && ref ? value / ref : null, ov };
  }

  let drag = null;
  stage.addEventListener("pointerdown", (e) => {
    if (st.plane === 4) return;
    stage.focus();
    drag = { x: e.clientX, y: e.clientY, px: st.panX, py: st.panY };
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener("pointermove", (e) => {
    if (!st.vol || st.plane === 4) return;
    if (drag) {
      st.panX = drag.px + (e.clientX - drag.x);
      st.panY = drag.py + (e.clientY - drag.y);
      draw();
      return;
    }
    st.hover = pick(e);
    draw();
  });
  stage.addEventListener("pointerup", () => { drag = null; });
  stage.addEventListener("pointerleave", () => { st.hover = null; if (st.vol && st.plane !== 4) draw(); });
  stage.addEventListener("dblclick", () => { if (st.plane !== 4) { fit(); draw(); } });
  stage.addEventListener("wheel", (e) => {
    if (!st.vol || st.plane === 4) return;
    e.preventDefault();
    if (e.ctrlKey || e.metaKey || st.plane === 3) {
      const factor = Math.exp(-e.deltaY * 0.0015);
      const r = stage.getBoundingClientRect();
      const mx = e.clientX - r.left - W / 2, my = e.clientY - r.top - H / 2;
      const nz = Math.max(0.5, Math.min(16, st.zoom * factor));
      const k = nz / st.zoom;
      st.panX = mx - (mx - st.panX) * k;
      st.panY = my - (my - st.panY) * k;
      st.zoom = nz;
    } else {
      step(Math.sign(e.deltaY) * (e.shiftKey ? 10 : 1));
    }
    draw();
  }, { passive: false });

  function step(delta) {
    if (!st.vol || st.plane === 3) return;
    const d = st.vol.depth(st.plane);
    st.idx[st.plane] = Math.max(0, Math.min(d - 1, st.idx[st.plane] + delta));
    draw();
  }

  slider.addEventListener("input", () => { st.idx[st.plane] = Number(slider.value); draw(); });

  const onKey = (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (["input", "select", "textarea"].includes(tag) || e.metaKey || e.ctrlKey || !st.vol) return;
    const k = e.key.toLowerCase();
    if (st.plane === 4) {
      const turn = 12;
      if (e.key === "ArrowLeft") r3d?.nudge(-turn, 0);
      else if (e.key === "ArrowRight") r3d?.nudge(turn, 0);
      else if (e.key === "ArrowUp") r3d?.nudge(0, -turn);
      else if (e.key === "ArrowDown") r3d?.nudge(0, turn);
      else if (["1", "2", "3", "4"].includes(k)) { const p = Number(k) - 1; if (p < 3 || st.vol.unwrap) setPlane(p); }
      else if (k === "o") { for (const cfg of Object.values(st.ov)) cfg.on = !cfg.on; renderSide(); draw(); }
      else if (k === "0" || k === "f") r3d?.setView("side");
      else return;
      e.preventDefault();
      return;
    }
    if (st.mode === "landmarks") {
      const lm = LANDMARKS.find((l) => l.keys.includes(k));
      if (lm) { e.preventDefault(); mark(lm, lm.keys.indexOf(k)); return; }
      if (k === "u") { e.preventDefault(); undoMark(); return; }
    }
    if (e.key === "ArrowUp" || e.key === "ArrowRight" || k === "." ) { e.preventDefault(); step(e.shiftKey ? 10 : 1); }
    else if (e.key === "ArrowDown" || e.key === "ArrowLeft" || k === ",") { e.preventDefault(); step(e.shiftKey ? -10 : -1); }
    else if (["1", "2", "3", "4"].includes(k)) { const p = Number(k) - 1; if (p < 3 || st.vol.unwrap) setPlane(p); }
    else if (k === "5") setPlane(4);
    else if (k === "m") { st.proj = st.proj === "max" ? "slice" : "max"; renderTopbar(); draw(); }
    else if (k === "o") { for (const cfg of Object.values(st.ov)) cfg.on = !cfg.on; renderSide(); draw(); }
    else if (k === "r" && st.profile?.dentin) { st.ratio = !st.ratio; renderSide(); draw(); }
    else if (k === "0" || k === "f") { fit(); draw(); }
    else if (k === "=" || k === "+") { st.zoom = Math.min(16, st.zoom * 1.25); draw(); }
    else if (k === "-") { st.zoom = Math.max(0.5, st.zoom / 1.25); draw(); }
  };
  window.addEventListener("keydown", onKey);

  // ------------------------------------------------------------ landmarks
  const history = [];
  function currentZ() {
    if ((st.plane === 1 || st.plane === 2) && st.hover) return Math.round(st.vol.fullZ(st.hover.row));
    return Math.round(st.vol.fullZ(st.idx[0]));
  }
  function mark(lm, which) {
    const z = currentZ();
    history.push(JSON.stringify(st.marks));
    const m = st.marks[lm.name] || { start: null, end: null };
    if (lm.kind === "event" || which === 0) m.start = z; else m.end = z;
    m.source = "manual";
    st.marks[lm.name] = m;
    toast(`${lm.label}${lm.kind === "span" ? (which === 0 ? " start" : " end") : ""} at z ${z}`, "info", { timeout: 1600 });
    renderSide();
    draw();
  }
  function undoMark() {
    if (!history.length) return;
    st.marks = JSON.parse(history.pop());
    renderSide();
    draw();
  }
  // A landmark back where the Landmarks step found it; every one of them without a name.
  function resetToAuto(only = null) {
    const found = (st.auto || []).filter((a) => !only || a.landmark === only);
    if (!found.length) return;
    history.push(JSON.stringify(st.marks));
    for (const a of found) st.marks[a.landmark] = { start: a.z_start, end: a.z_end, source: "auto" };
    const label = LANDMARKS.find((lm) => lm.name === only)?.label || only;
    toast(`${only ? `${label} is` : "Every landmark is"} back where it was found; save to keep ${only ? "it" : "them"}.`, "info");
    renderSide();
    draw();
  }
  async function saveMarks(run) {
    const list = [];
    for (const lm of LANDMARKS) {
      const m = st.marks[lm.name];
      if (!m || (m.start == null && m.end == null)) continue;
      const source = isAuto(m) ? "auto" : "manual";
      if (lm.kind === "event") list.push({ landmark: lm.name, z_start: m.start, source });
      else if (m.start != null && m.end != null) list.push({ landmark: lm.name, z_start: m.start, z_end: m.end, source });
      else list.push({ landmark: lm.name, z_start: m.start ?? m.end, z_end: null, source });
    }
    try {
      const res = await api.post(`/api/samples/${enc(name)}/marks`, { marks: list, run });
      if (res.run?.skipped?.length) toast(res.run.skipped[0].reason, "warning");
      else toast(run ? `Saved ${res.saved} landmark${res.saved === 1 ? "" : "s"}; mapping them onto the density profile.` : `Saved ${res.saved} landmark${res.saved === 1 ? "" : "s"}.`);
    } catch (err) { errorToast(err); }
  }

  // ------------------------------------------------------------ top bar
  function renderTopbar() {
    const vol = st.vol;
    const planeLabels = st.view === "cmpr" ? ["Cross-section", "Lengthwise (Y)", "Lengthwise (X)", "Unrolled"] : ["Axial", "Coronal", "Sagittal"];
    const planes = planeLabels.filter((_, i) => i < 3 || vol?.unwrap);
    // The specimen's library folder, leading back to the others in it.
    const where = state.samples.get(name)?.folder || "";
    replace(topbar,
      h("a", { href: `#/sample/${enc(name)}`, title: "Back to the specimen" }, icon("back")),
      where ? h("a", { class: "viewer-folder", href: viewerHref(where), title: `The specimens in ${where.split("/").join(" › ")}` },
        icon("folder"), leafOf(where), h("span", { class: "sep", "aria-hidden": "true" }, icon("chevron"))) : null,
      h("span", { class: "specimen" }, name),
      h("select", { "aria-label": "View", onchange: (e) => go(`#/viewer/${enc(name)}/${e.target.value}${st.mode ? "?mode=landmarks" : st.plane === 4 ? "?plane=3d" : ""}`) },
        views.map((v) => h("option", { value: v.view, selected: v.view === st.view }, `${v.label}${v.built ? "" : " (not built)"}`))),
      h("div", { class: "seg", role: "group", "aria-label": "Plane" }, planes.map((label, i) =>
        h("button", { "aria-pressed": st.plane === i ? "true" : "false", title: `${label} (${i + 1})`,
          onclick: () => setPlane(i) }, label)),
        h("button", { "aria-pressed": st.plane === 4 ? "true" : "false", title: "Turn it in 3-D (5)", onclick: () => setPlane(4) }, "3-D")),
      st.plane < 3 ? h("select", { "aria-label": "Projection", onchange: (e) => { st.proj = e.target.value; draw(); renderTopbar(); } },
        [["slice", "Slice"], ["slab", "Thick slab (max)"], ["max", "Maximum projection"], ["mean", "Mean projection"]]
          .map(([v, l]) => h("option", { value: v, selected: st.proj === v }, l))) : null,
      st.proj === "slab" && st.plane < 3 ? h("label", { class: "row", style: { color: "var(--stage-ink-2)", fontSize: "12.5px" } }, "Thickness",
        h("input", { type: "number", min: 3, max: 101, step: 2, value: st.slab, style: { width: "64px" },
          onchange: (e) => { st.slab = Math.max(3, Number(e.target.value) || 9); draw(); } })) : null,
      h("span", { class: "spacer" }),
      h("button", { class: "btn small", title: st.plane === 4 ? "Back to the first view (0)" : "Fit to the window (0)",
        onclick: () => { if (st.plane === 4) r3d?.setView("side"); else { fit(); draw(); } } }, st.plane === 4 ? "Reset" : "Fit"));
  }

  // ------------------------------------------------------------ side panel
  function side3d() {
    const levelIn = h("input", { type: "number", step: "any", "aria-label": "Surface level",
      value: sig(st.iso ?? st.lo, 5).replace(/,/g, ""), onchange: (e) => { st.iso = Number(e.target.value); draw(); } });
    const views3d = [["side", "Side"], ["other", "Other side"], ["top", "Top"], ["bottom", "Below"], ["end", "End"]];
    return h("section", null, h("h3", null, "3-D"),
      h("div", { class: "seg", role: "group", "aria-label": "3-D style" },
        [["surface", "Surface"], ["xray", "X-ray"]].map(([v, label]) => h("button", { "aria-pressed": st.style3d === v ? "true" : "false",
          onclick: () => { st.style3d = v; renderSide(); draw(); } }, label))),
      st.style3d === "surface" ? h("div", { class: "win-inputs" },
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Surface at"), levelIn),
        h("button", { class: "btn small", disabled: st.iso == null, onclick: () => { st.iso = null; renderSide(); draw(); } }, "Follow the window")) : null,
      h("div", { class: "row", style: { marginTop: "10px" } }, h("span", { class: "lbl" }, "Cut away"),
        h("input", { type: "range", min: 0, max: 0.95, step: 0.01, value: st.cut, "aria-label": "Cut away",
          oninput: (e) => { st.cut = Number(e.target.value); draw(); } })),
      st.vol.meta.overlays.length && st.style3d === "surface" ? h("div", { class: "row", style: { marginTop: "8px" } },
        h("span", { class: "lbl" }, "Structure colour"),
        h("input", { type: "range", min: 0, max: 1, step: 0.05, value: st.tint, "aria-label": "Structure colour strength",
          oninput: (e) => { st.tint = Number(e.target.value); draw(); } })) : null,
      h("div", { class: "row wrap", style: { marginTop: "10px" } },
        views3d.map(([v, label]) => h("button", { class: "btn small", onclick: () => r3d?.setView(v) }, label)),
        h("button", { class: "btn small", onclick: () => r3d?.save(`${name}_${st.view}_3d.png`) }, icon("download"), "Save image")),
      h("div", { class: "lbl", style: { marginTop: "8px" } }, st.style3d === "surface"
        ? "Each ray stops at the first voxel as bright as the surface level. Its colour is the density just beneath, on the heatmap's scale; structures tint what they cover."
        : "The brightest value along each ray, on the heatmap's scale."),
      h("div", { class: "lbl", style: { marginTop: "6px" } }, `Drag to turn it all the way round, about its tissue's centre. Shift-drag or right-drag moves, ${ROLL}-drag rolls, scroll zooms, double-click resets. Arrow keys turn it.`));
  }

  function sideViews() {
    const info = views.find((v) => v.view === st.view);
    const job = previewJob(name, st.view);
    return h("section", null, h("h3", null, "View"),
      h("div", { class: "lbl" }, info ? `${info.label}, from ${basename(info.base)}` : ""),
      job ? h("div", { class: "lbl", style: { marginTop: "6px" } }, `Preview building: ${job.stage || "queued"}`) : null,
      h("div", { class: "row wrap", style: { marginTop: "8px" } },
        h("button", { class: "btn small", onclick: async () => {
          try { await api.post(`/api/samples/${enc(name)}/views/${st.view}/build`, { force: true }); toast("Rebuilding the preview."); showMessage("Rebuilding the preview…", 0); }
          catch (err) { errorToast(err); }
        } }, icon("refresh"), "Rebuild preview")));
  }

  const hist = histWindow((lo, hi) => { st.lo = lo; st.hi = hi; syncInputs(); draw(); });
  const loIn = h("input", { type: "number", step: "any", "aria-label": "Window low", onchange: () => { st.lo = Number(loIn.value); hist.set(histState()); draw(); } });
  const hiIn = h("input", { type: "number", step: "any", "aria-label": "Window high", onchange: () => { st.hi = Number(hiIn.value); hist.set(histState()); draw(); } });
  function syncInputs() { loIn.value = sig(st.lo, 5).replace(/,/g, ""); hiIn.value = sig(st.hi, 5).replace(/,/g, ""); }
  function histState() {
    const markers = [];
    if (st.molar) {
      markers.push({ value: st.molar.threshold, label: "enamel" });
      if (st.molar.dentin_mode != null) markers.push({ value: st.molar.dentin_mode, label: "dentin" });
    }
    return { hist: st.vol.meta.histogram, lo: st.lo, hi: st.hi, gamma: st.gamma, cmap: st.cmap, markers };
  }

  function renderSide() {
    const vol = st.vol;
    if (!vol) return;
    const cmapButtons = h("div", { class: "cmaps", role: "group", "aria-label": "Colour scale" },
      SEQUENTIAL.map((c) => {
        const g = gradientCanvas(c, 90, 10);
        return h("button", { "aria-pressed": st.cmap === c && !st.ratio ? "true" : "false",
          onclick: () => { st.cmap = c; st.ratio = false; renderSide(); draw(); } }, g, COLORMAPS[c].label);
      }));
    const gamma = h("input", { type: "range", min: 0.3, max: 3, step: 0.05, value: st.gamma, "aria-label": "Gamma",
      oninput: (e) => { st.gamma = Number(e.target.value); gammaVal.textContent = st.gamma.toFixed(2); hist.set(histState()); draw(); } });
    const gammaVal = h("span", { class: "lbl num" }, st.gamma.toFixed(2));
    const heat = h("section", null, h("h3", null, "Heatmap"), cmapButtons, hist.el,
      h("div", { class: "win-inputs" },
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Low"), loIn),
        h("label", { class: "field" }, h("span", { class: "lbl" }, "High"), hiIn)),
      h("div", { class: "presets" }, presets().map(([label, fn]) =>
        h("button", { class: "btn", onclick: () => { fn(); syncInputs(); hist.set(histState()); renderSide(); draw(); } }, label))),
      h("div", { class: "row", style: { marginTop: "10px" } }, h("span", { class: "lbl" }, "Gamma"), gamma, gammaVal),
      h("label", { class: "check lbl", style: { marginTop: "8px" } },
        h("input", { type: "checkbox", checked: st.smooth, onchange: (e) => { st.smooth = e.target.checked; draw(); } }), "Smooth pixels"));
    syncInputs();

    const sections = st.plane === 4 ? [sideViews(), side3d(), heat] : [sideViews(), heat];

    if (st.view === "cmpr" && st.plane !== 4) {
      const ok = !!st.profile?.dentin;
      const lo = h("input", { type: "number", step: 0.05, value: st.rlo, disabled: !ok, onchange: (e) => { st.rlo = Number(e.target.value); draw(); } });
      const hi = h("input", { type: "number", step: 0.05, value: st.rhi, disabled: !ok, onchange: (e) => { st.rhi = Number(e.target.value); draw(); } });
      const g = gradientCanvas("ratio", 260, 10);
      g.style.cssText = "width:100%;height:10px;border-radius:2px;margin-top:8px";
      sections.push(h("section", null, h("h3", null, "Enamel/dentin ratio"),
        ok ? h("label", { class: "check lbl" }, h("input", { type: "checkbox", checked: st.ratio,
          onchange: (e) => { st.ratio = e.target.checked; renderSide(); draw(); } }), "Colour by density relative to dentin (R)")
          : h("div", { class: "lbl" }, "Run Incisor enamel to colour each slice by its own dentin reference."),
        ok ? g : null,
        ok ? h("div", { class: "win-inputs" }, h("label", { class: "field" }, h("span", { class: "lbl" }, "Low ratio"), lo),
          h("label", { class: "field" }, h("span", { class: "lbl" }, "High ratio"), hi)) : null,
        ok ? h("div", { class: "lbl", style: { marginTop: "6px" } }, "Grey is dentin density (1.0). Blue is less mineralized, red more.") : null));
    }

    if (vol.meta.overlays.length && st.plane !== 3) {
      sections.push(h("section", null, h("h3", null, "Structures"),
        vol.meta.overlays.map((o) => {
          const cfg = st.ov[o.name];
          return h("div", { class: "ov-row" },
            h("i", { class: "sw", style: { background: OVERLAY_COLORS[o.name] || "#fff" } }),
            h("label", { class: "check lbl" }, h("input", { type: "checkbox", checked: cfg.on, onchange: (e) => { cfg.on = e.target.checked; draw(); } }), o.label),
            h("select", { style: { height: "26px" }, "aria-label": `${o.label} style`, onchange: (e) => { cfg.style = e.target.value; draw(); } },
              h("option", { value: "fill", selected: cfg.style === "fill" }, "Fill"), h("option", { value: "outline", selected: cfg.style === "outline" }, "Outline")),
            cfg.style === "fill" ? h("input", { type: "range", min: 0.05, max: 1, step: 0.05, value: cfg.alpha, "aria-label": `${o.label} opacity`,
              oninput: (e) => { cfg.alpha = Number(e.target.value); draw(); } }) : null);
        }),
        h("div", { class: "lbl", style: { marginTop: "6px" } }, "O toggles every structure.")));
    }

    if (st.view === "mandible" && st.plane !== 4) {
      // A hand mark with somewhere found to go back to gets a button for it.
      const resettable = (lm) => {
        const m = st.marks[lm.name];
        return Boolean(m && (m.start != null || m.end != null) && !isAuto(m) && (st.auto || []).some((a) => a.landmark === lm.name));
      };
      const withReset = LANDMARKS.some(resettable);
      const rows = LANDMARKS.map((lm) => {
        const m = st.marks[lm.name] || {};
        const zs = lm.kind === "event" ? (m.start != null ? `z ${m.start}` : "–")
          : m.start == null && m.end == null ? "–" : `z ${m.start ?? "–"}–${m.end ?? "–"}`;
        const found = (st.auto || []).find((a) => a.landmark === lm.name);
        const placed = m.start != null || m.end != null;
        return h("div", { class: "mark-row" },
          h("i", { class: "sw", style: { background: lm.color } }),
          h("span", null, lm.label, " ", lm.keys.map((k) => h("kbd", null, k.toUpperCase()))),
          h("span", { class: "z", title: !placed ? null : isAuto(m) ? `Found by the Landmarks step (${AUTO_HOW[lm.name] || "automatically"}). Press its key to move it by hand.`
            : found ? `Set by hand. Found at ${found.z_end != null ? `z ${found.z_start} to ${found.z_end}` : `z ${found.z_start}`}.` : "Set by hand." },
            zs, placed ? (isAuto(m) ? " · auto" : " · by hand") : ""),
          resettable(lm) ? h("button", { class: "btn small ghost icon", "aria-label": `Put ${lm.label} back where it was found`,
            title: "Back to where it was found", onclick: () => resetToAuto(lm.name) }, icon("refresh")) : withReset ? h("span") : null,
          h("button", { class: "btn small ghost icon", "aria-label": `Go to ${lm.label}`, disabled: m.start == null,
            onclick: () => { st.idx[0] = st.vol.fromFullZ(m.start); st.plane = 0; renderTopbar(); draw(); } }, icon("target")));
      });
      sections.push(h("section", null, h("h3", null, "Landmarks"),
        st.mode === "landmarks"
          ? h("div", { class: "lbl", style: { marginBottom: "8px" } }, "Scroll to where a landmark sits, or hover a lengthwise view, and press its key. The first key of a molar marks where it starts, the second where it ends.")
          : h("button", { class: "btn small", style: { marginBottom: "8px" }, onclick: () => { st.mode = "landmarks"; renderSide(); draw(); } },
            icon("target"), Object.keys(st.marks).length ? "Adjust landmarks" : "Pick landmarks"),
        h("div", { class: `marks ${withReset ? "with-reset" : ""}` }, rows),
        st.markWarnings?.length ? h("div", { class: "lbl", style: { marginTop: "8px" } }, st.markWarnings.join(" ")) : null,
        h("div", { class: "row wrap", style: { marginTop: "10px" } },
          h("button", { class: "btn small primary", onclick: () => saveMarks(true) }, "Save and map"),
          h("button", { class: "btn small", onclick: () => saveMarks(false) }, "Save"),
          h("button", { class: "btn small ghost", onclick: undoMark }, "Undo (U)")),
        h("button", { class: "btn small ghost", style: { marginTop: "6px" }, disabled: !st.auto?.length,
          title: st.auto?.length ? "Every landmark back where the Landmarks step found it"
            : "Run the Landmarks step on this specimen to find them",
          onclick: () => resetToAuto() }, "Back to the found landmarks"),
        st.mapped.length ? h("div", { class: "lbl", style: { marginTop: "8px" } }, "Mapped: ",
          st.mapped.map((r) => r.kind === "span" ? `${r.landmark} ${sig(r.start, 3)}–${sig(r.end, 3)}%` : `${r.landmark} ${sig(r.start, 3)}%`).join(", ")) : null));
    }

    const f = vol.meta.factors;
    sections.push(h("section", null, h("h3", null, "Preview"),
      h("div", { class: "lbl" }, `Full volume ${vol.meta.full_shape.join(" × ")} voxels${vol.meta.spacing_known ? ` at ${vol.meta.spacing_mm.map((s) => sig(s * 1000, 3)).join(" × ")} µm` : ", voxel size unknown"}.`),
      h("div", { class: "lbl" }, `Shown at ${vol.meta.shape.join(" × ")} (block ${f.join(" × ")}${vol.meta.sampling === "strided" ? ", strided" : ", averaged"}). The histogram is from the full volume.`),
      h("div", { class: "lbl", style: { marginTop: "6px" } }, "Keys: arrows or , . change slice (Shift ×10), 1–4 planes, 5 3-D, M maximum projection, 0 fit, +/− zoom. Drag to pan, Ctrl-scroll to zoom.")));
    replace(side, sections);
    hist.set(histState());
  }

  // ------------------------------------------------------------ live preview status
  const offJobs = on("jobs", async (job) => {
    if (!job || job.sample !== name || job.step !== "preview" || job.params?.view !== st.view) return;
    if (job.status === "done") {
      views = await api.get(`/api/samples/${enc(name)}/views`);
      openView();
    } else if (!st.vol && ["running", "queued", "waiting"].includes(job.status)) {
      const loading = stage.querySelector(".loading .progress i");
      if (loading) loading.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    } else if (job.status === "failed") {
      showMessage(`The preview could not be built: ${job.error || "see the queue for the log"}`, null);
    }
  });

  resize();
  await openView();
  return () => { ro.disconnect(); window.removeEventListener("keydown", onKey); offJobs(); r3d?.dispose(); };
}

// ---------------------------------------------------------------- no sample chosen
// The library's folders, as the Library files them: a folder's own specimens
// below the folders inside it, the path above leading back up, and a search
// that looks through every folder inside this one. As cards with a preview,
// or as a list; the browser remembers which.

const viewerHref = (folder) => (folder ? `#/viewer?folder=${encodeURIComponent(folder)}` : "#/viewer");
const LAYOUT_KEY = "studio.viewer.layout";
function savedLayout() {
  try { return localStorage.getItem(LAYOUT_KEY) === "list" ? "list" : "grid"; } catch { return "grid"; }
}
const plural = (n, one) => `${n} ${one}${n === 1 ? "" : "s"}`;
const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });

function pickSample(root, query = {}) {
  const folder = String(query.folder || "").split("/").map((p) => p.trim()).filter(Boolean).join("/");
  const ui = { q: "", groups: new Set(), layout: savedLayout() };
  const page = h("div", { class: "page viewer-pick" });
  root.appendChild(page);

  const crumbs = h("div");
  const title = h("h1");
  const sub = h("div", { class: "sub" });
  page.appendChild(h("div", { class: "page-head" }, h("div", null, crumbs, title, sub)));
  const search = h("input", { type: "search", placeholder: folder ? "Search this folder" : "Search specimens", "aria-label": "Search specimens",
    oninput: (e) => { ui.q = e.target.value.toLowerCase(); refresh(); } });
  const groupChips = h("div", { class: "row wrap" });
  const layoutSeg = h("div", { class: "seg layout-seg", role: "group", "aria-label": "Layout" });
  page.appendChild(h("div", { class: "filters" }, search, groupChips, h("span", { class: "spacer" }), layoutSeg));
  const body = h("div");
  page.appendChild(body);

  const all = () => [...state.samples.values()];
  const under = () => all().filter((s) => inFolder(s.folder || "", folder));
  const filtering = () => Boolean(ui.q || ui.groups.size);
  const matches = (s) => (!ui.q || [s.name, s.group, s.folder, s.meta?.study, s.meta?.line, s.meta?.notes]
    .join(" ").toLowerCase().includes(ui.q)) && (!ui.groups.size || ui.groups.has(s.group));
  const open = (s) => `#/viewer/${encodeURIComponent(s.name)}`;
  // A specimen found by a search in a folder inside this one says which.
  const elsewhere = (s) => ((s.folder || "") !== folder ? (s.folder || "").slice(folder ? folder.length + 1 : 0) : "");
  const inFolderTag = (s) => (elsewhere(s) ? h("span", { class: "in-folder" }, icon("folder"), elsewhere(s).split("/").join(" › ")) : null);

  // What a folder holds, however deep, and the groups among it.
  function folderInfo(path, samples) {
    const inside = samples.filter((s) => inFolder(s.folder || "", path));
    const folders = libraryFolders().filter((f) => parentOf(f) === path).length;
    const groups = [...new Map(inside.map((s) => [s.group, s])).values()].sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99));
    return {
      inside, groups,
      text: [inside.length ? plural(inside.length, "specimen") : "Empty", folders ? plural(folders, "folder") : null].filter(Boolean).join(", "),
      latest: inside.reduce((m, s) => (s.updated && s.updated > m ? s.updated : m), ""),
      swatches: h("span", { class: "folder-groups", title: groups.map((g) => g.group).join(", ") },
        groups.slice(0, 6).map((g) => h("i", { class: "sw", style: { background: slotColor(g.slot) } })),
        groups.length > 6 ? h("span", { class: "muted" }, `+${groups.length - 6}`) : null),
    };
  }

  // ---- as cards
  function grid(folders, rows, samples) {
    return h("div", { class: "views-grid" },
      folders.map((path) => {
        const info = folderInfo(path, samples);
        return h("a", { class: "view-card folder-card", href: viewerHref(path), title: path.split("/").join(" › ") },
          h("div", { class: "th" }, icon("folder")),
          h("div", { class: "cap" }, h("span", { class: "folder-name" }, leafOf(path)),
            h("small", { class: "folder-sub" }, info.text, info.swatches)));
      }),
      rows.map((s) => h("a", { class: "view-card", href: open(s) },
        h("div", { class: "th" }, s.thumb ? h("img", { src: s.thumb, alt: "", loading: "lazy" }) : h("span", null, "No preview yet")),
        h("div", { class: "cap" },
          h("div", { class: "specimen-line" }, h("span", { class: "specimen", style: { fontSize: "15px" } }, s.name),
            // A card has room for the flag alone; its note and scope on hover.
            s.excluded ? h("span", { class: "tag flagged", title: `${flagTag(s.excluded).textContent}: ${s.excluded.note || "no note"}` }, icon("flag"), "Flagged") : null),
          h("small", null, s.group), inFolderTag(s) ? h("small", null, inFolderTag(s)) : null))));
  }

  // ---- as a list
  function list(folders, rows, samples) {
    return h("div", { class: "table-wrap" }, h("table", { class: "data viewer-list" },
      h("thead", null, h("tr", null, h("th", { style: { width: "84px" } }, h("span", { class: "sr-only" }, "Preview")),
        h("th", null, "Specimen"), h("th", null, "Group"), h("th", null, "Pipeline"), h("th", { class: "hide-sm" }, "Updated"))),
      folders.length ? h("tbody", { class: "folders" }, folders.map((path) => {
        const info = folderInfo(path, samples);
        return h("tr", { class: "clickable folder-row", onclick: () => go(viewerHref(path)) },
          h("td", { class: "folder-icon" }, icon("folder")),
          h("td", null, h("a", { class: "folder-name", href: viewerHref(path), onclick: (e) => e.stopPropagation() }, leafOf(path)),
            h("div", { class: "specimen-sub" }, info.text)),
          h("td", null, info.swatches),
          h("td", { class: "muted" }, info.inside.length ? `${info.inside.filter((s) => s.thumb).length} of ${info.inside.length} with a preview` : null),
          h("td", { class: "hide-sm muted nowrap" }, relTime(info.latest)));
      })) : null,
      h("tbody", null, rows.map((s) => h("tr", { class: "clickable", onclick: () => go(open(s)) },
        h("td", null, h("div", { class: "list-thumb" }, s.thumb ? h("img", { src: s.thumb, alt: "", loading: "lazy" }) : null)),
        h("td", null, h("div", { class: "specimen-line" },
          h("a", { class: "specimen", href: open(s), onclick: (e) => e.stopPropagation() }, s.name), flagTag(s.excluded)),
          h("div", { class: "specimen-sub" }, inFolderTag(s), s.thumb ? null : "No preview yet")),
        h("td", null, groupLabel(s)),
        h("td", null, track(s)),
        h("td", { class: "hide-sm muted nowrap" }, relTime(s.updated)))))));
  }

  const refresh = rafThrottle(() => {
    const samples = under();
    const folders = libraryFolders();
    // The folder was renamed or removed elsewhere: show the nearest one still there.
    if (folder && !folders.includes(folder) && !samples.length) {
      let up = parentOf(folder);
      while (up && !folders.includes(up)) up = parentOf(up);
      go(viewerHref(up));
      return;
    }
    replace(crumbs, folder ? folderCrumbs(parentOf(folder), null, { href: viewerHref, root: "Viewer" }) : null);
    title.textContent = folder ? leafOf(folder) : "Viewer";
    const kids = folders.filter((f) => parentOf(f) === folder && f !== folder).sort((a, b) => natural(leafOf(a), leafOf(b)));
    sub.textContent = [samples.length ? plural(samples.length, "specimen") : null, kids.length ? plural(kids.length, "folder") : null]
      .filter(Boolean).join(", ") + (samples.length ? ". Choose a specimen: every volume opens as an adjustable heatmap." : "");
    const groups = [...new Map(samples.map((s) => [s.group, s])).values()].sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99));
    for (const g of [...ui.groups]) if (!groups.some((x) => x.group === g)) ui.groups.delete(g);
    replace(groupChips, groups.length > 1 ? groups.map((g) => h("button", {
      class: "chip", "aria-pressed": ui.groups.has(g.group) ? "true" : "false",
      onclick: () => { ui.groups.has(g.group) ? ui.groups.delete(g.group) : ui.groups.add(g.group); refresh(); },
    }, h("i", { class: "sw", style: { background: slotColor(g.slot) } }), g.group)) : null);
    replace(layoutSeg, [["grid", "Grid"], ["list", "List"]].map(([key, label]) => h("button", {
      "aria-pressed": ui.layout === key ? "true" : "false", title: key === "grid" ? "Cards with a preview" : "A row each",
      onclick: () => { ui.layout = key; try { localStorage.setItem(LAYOUT_KEY, key); } catch { /* private window */ } refresh(); },
    }, icon(key), label)));

    if (!all().length && !folders.length) {
      replace(body, h("div", { class: "empty" }, h("h3", null, "Nothing to view yet"), h("a", { class: "btn primary", href: "#/import" }, "Import a scan")));
      return;
    }
    const rows = (filtering() ? samples.filter(matches) : samples.filter((s) => (s.folder || "") === folder)).sort((a, b) => natural(a.name, b.name));
    const shownFolders = filtering() ? [] : kids;
    if (rows.length || shownFolders.length) {
      replace(body, ui.layout === "list" ? list(shownFolders, rows, samples) : grid(shownFolders, rows, samples),
        filtering() && kids.length ? h("p", { class: "muted", style: { fontSize: "12.5px", marginTop: "8px" } },
          folder ? "Searching this folder and the folders inside it." : "Searching every folder of the library.") : null);
    } else if (filtering()) {
      replace(body, h("div", { class: "empty" }, h("h3", null, folder ? "Nothing in this folder matches" : "Nothing matches"),
        h("button", { class: "btn", onclick: () => { ui.q = ""; search.value = ""; ui.groups.clear(); refresh(); } }, "Clear filters")));
    } else {
      replace(body, h("div", { class: "empty" }, h("h3", null, "This folder is empty"),
        h("span", null, "File specimens in it from the Library."),
        h("a", { class: "btn", href: viewerHref(parentOf(folder)) }, icon("back"), parentOf(folder) ? `Back to ${leafOf(parentOf(folder))}` : "Back to every folder")));
    }
  });

  const offs = [on("samples", refresh), on("app", refresh), on("folders", refresh)];
  refresh();
  return () => offs.forEach((f) => f());
}
