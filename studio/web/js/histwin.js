// The histogram window: a volume's intensity histogram with the heatmap's
// colour window over it, dragged at its edges to set the range, and a strip
// beneath showing which colour each value gets. Given several series instead
// of one histogram, it draws each as a line in its own colour on one axis:
// the compare page's specimens on their shared scale.

import { h, sig } from "./lib.js";
import { colorAt } from "./colormaps.js";

/**
 * onChange(lo, hi) as the window is dragged. set() takes
 *   { hist: {edges, counts} | series: [{edges, counts, color}], lo, hi, gamma, cmap,
 *     markers: [{value, label?, color?}], color?: (value) => [r, g, b] }
 * A marker with a colour is a short tick in it (one specimen's); one without
 * is a labelled dashed line.
 */
export function histWindow(onChange) {
  const canvas = h("canvas", { "aria-label": "Intensity histogram. Drag the edges to set the colour range." });
  const wrap = h("div", { class: "histwin" }, canvas);
  let s = null, dragging = null;
  const H = 118, PAD = 6, HIST_TOP = 6, HIST_BOTTOM = 86, STRIP_TOP = 92, STRIP_H = 9;
  const width = () => wrap.clientWidth || 280;
  const range = () => {
    if (s.series?.length) {
      return [Math.min(...s.series.map((r) => r.edges[0])), Math.max(...s.series.map((r) => r.edges[r.edges.length - 1]))];
    }
    return [s.hist.edges[0], s.hist.edges[s.hist.edges.length - 1]];
  };
  const x = (v) => { const [e0, e1] = range(); return PAD + ((v - e0) / (e1 - e0 || 1)) * (width() - 2 * PAD); };
  const v = (px) => { const [e0, e1] = range(); return e0 + ((px - PAD) / (width() - 2 * PAD)) * (e1 - e0); };

  function draw() {
    if (!s || !(s.hist || s.series?.length)) return;
    const w = width(), dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.height = `${H}px`;
    const c = canvas.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, H);
    const xl = x(s.lo), xh = x(s.hi);
    if (s.series?.length) {
      // One line per series, each scaled to its own tallest bin: specimens of
      // different sizes then compare by shape, not by voxel count.
      c.lineWidth = 1.5;
      c.lineJoin = "round";
      for (const r of s.series) {
        const max = Math.log1p(Math.max(...r.counts, 1));
        c.strokeStyle = r.color;
        c.beginPath();
        r.counts.forEach((n, i) => {
          const px = x((r.edges[i] + r.edges[i + 1]) / 2);
          const py = HIST_BOTTOM - (Math.log1p(n) / max) * (HIST_BOTTOM - HIST_TOP);
          if (i) c.lineTo(px, py); else c.moveTo(px, py);
        });
        c.stroke();
      }
    } else {
      const counts = s.hist.counts;
      const max = Math.log1p(Math.max(...counts, 1));
      const bw = (w - 2 * PAD) / counts.length;
      for (let i = 0; i < counts.length; i++) {
        if (!counts[i]) continue;
        const hh = (Math.log1p(counts[i]) / max) * (HIST_BOTTOM - HIST_TOP);
        const bx = PAD + i * bw;
        c.fillStyle = bx + bw >= xl && bx <= xh ? "rgba(226,233,231,0.75)" : "rgba(226,233,231,0.28)";
        c.fillRect(bx, HIST_BOTTOM - hh, Math.max(1, bw - 0.5), hh);
      }
    }
    // The colour each intensity gets, drawn under the histogram: by the page's
    // own rule when it gives one (steps, say), else by the window and gamma.
    for (let px = PAD; px < w - PAD; px++) {
      let t = (v(px) - s.lo) / (s.hi - s.lo || 1);
      t = t <= 0 ? 0 : t >= 1 ? 1 : Math.pow(t, 1 / (s.gamma || 1));
      const [r, g, b] = s.color ? s.color(v(px)) : colorAt(s.cmap, t);
      c.fillStyle = `rgb(${r},${g},${b})`;
      c.fillRect(px, STRIP_TOP, 1, STRIP_H);
    }
    c.fillStyle = "rgba(255,255,255,0.08)";
    c.fillRect(xl, HIST_TOP, Math.max(1, xh - xl), HIST_BOTTOM - HIST_TOP);
    for (const xx of [xl, xh]) {
      c.fillStyle = "#e2e9e7";
      c.fillRect(xx - 1, HIST_TOP - 2, 2, STRIP_TOP + STRIP_H - HIST_TOP + 2);
      c.fillRect(xx - 4, HIST_TOP - 4, 8, 6);
    }
    c.font = "10.5px system-ui, sans-serif";
    c.textBaseline = "top";
    for (const m of s.markers || []) {
      const xm = x(m.value);
      if (m.color) {
        c.fillStyle = m.color;
        c.fillRect(xm - 1, HIST_BOTTOM - 12, 2, 12);
        continue;
      }
      c.strokeStyle = "rgba(232,180,0,0.9)";
      c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(xm, HIST_TOP); c.lineTo(xm, HIST_BOTTOM); c.stroke();
      c.setLineDash([]);
      c.fillStyle = "#e8b400";
      c.textAlign = xm > w - 50 ? "right" : "left";
      c.fillText(m.label, xm + (xm > w - 50 ? -3 : 3), HIST_TOP);
    }
    const [e0, e1] = range();
    c.fillStyle = "#a3aeac";
    c.textAlign = "left";
    c.fillText(sig(e0, 4), PAD, STRIP_TOP + STRIP_H + 4);
    c.textAlign = "right";
    c.fillText(sig(e1, 4), w - PAD, STRIP_TOP + STRIP_H + 4);
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (!s) return;
    const r = canvas.getBoundingClientRect(), px = e.clientX - r.left;
    const xl = x(s.lo), xh = x(s.hi);
    if (Math.abs(px - xl) < 8) dragging = { kind: "lo" };
    else if (Math.abs(px - xh) < 8) dragging = { kind: "hi" };
    else if (px > xl && px < xh) dragging = { kind: "both", start: px, lo: s.lo, hi: s.hi };
    else dragging = { kind: px < xl ? "lo" : "hi" };
    canvas.setPointerCapture(e.pointerId);
    move(e);
  });
  const move = (e) => {
    if (!dragging || !s) return;
    const r = canvas.getBoundingClientRect(), px = e.clientX - r.left;
    const [e0, e1] = range();
    const clamp = (val) => Math.max(e0, Math.min(e1, val));
    if (dragging.kind === "lo") s.lo = Math.min(clamp(v(px)), s.hi - (e1 - e0) * 0.002);
    else if (dragging.kind === "hi") s.hi = Math.max(clamp(v(px)), s.lo + (e1 - e0) * 0.002);
    else {
      const d = v(px) - v(dragging.start);
      s.lo = dragging.lo + d; s.hi = dragging.hi + d;
    }
    draw();
    onChange(s.lo, s.hi);
  };
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerup", () => { dragging = null; });
  new ResizeObserver(draw).observe(wrap);
  return { el: wrap, set(next) { s = { ...next }; draw(); } };
}
