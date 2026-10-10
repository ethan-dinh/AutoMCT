// Figures drawn as GraphPad Prism draws them, for a paper: white ground, black
// axes on the left and bottom only, ticks pointing out, no grid, Arial, and
// the figure laid out in points at the size it will be printed. The results
// page hands over what it shows (a spec, below) and the export dialog turns
// the drawing into SVG, or a PNG or TIFF at a chosen DPI. No DOM here beyond
// measuring text, so the tests run the layout under Node.
//
// Two kinds of figure, as the page has two kinds of chart:
//
//   column  one value per specimen. Without splits, Prism's column scatter: a
//           column per group, each specimen a point, the mean a line, the SD
//           (or SEM) a capped bar. With splits, Prism's grouped layout: a
//           cluster per stratum along the axis, the groups side by side in each
//           and named in a legend. Or bars at the mean with the points on them.
//           Significance stars from the page's Welch tests, on brackets over
//           each pair of groups compared within a stratum.
//
//     { kind: "column", title, heading, yLabel, splits: [labels],
//       levels: [{level, name, color}],
//       strata: [{label, cells: [{level, name, color, values: [{label, value}]}]}],
//       tests: [{a, b, within, p}] }          (a, b: level names; within: a stratum's label)
//
//   xy      profiles along the incisor: a panel per stratum on one y scale,
//           each group's mean with a shaded ±1 SD band, the specimens' own
//           curves thin behind, and the landmark bands as the page draws them.
//
//     { kind: "xy", title, heading, xLabel, yLabel, xDomain, yDomain, rules: [{y, label}],
//       legend: [{label, color, slot, dentin}],
//       panels: [{title, spans, series: [{role: mean|specimen|dentin, slot, color, bandColor, points, band}]}] }
//
// `title` names the files and tables; `heading`, when given, is drawn above the
// plot (with the `title` option on). `ink`, when given, colours the rest:
// {mean, error, rule, brackets, title, axisTitle, text, axes, background}, any
// left out as INK has them. A landmark span's `color` overrides its tone.
//
// Colours arrive as the page's CSS colours (var(--s1)); `color` in the options
// turns them into the ones printed.

import { BAND_OPACITY, landmarkStops, spreadLabels } from "./charts.js";

export const FONT = "Arial, Helvetica, sans-serif";

export const FIGURE_DEFAULTS = {
  width: 3.5, height: 2.6,          // inches
  font: 10,                         // points
  bold: true,                       // axis titles and group names in bold, as Prism sets them
  palette: "group",                 // "group": the page's colours; "black": Prism's black symbols
  style: "scatter",                 // "scatter" or "bars" (column figures)
  error: "sd",                      // "sd" or "sem"
  zero: false,                      // start the y axis at 0 (bars always do)
  significance: "stars",            // "stars", "p" or "off"
  ns: true,                         // label pairs that differ by less than p = 0.05 "ns"
  title: true,                      // the spec's heading above the plot
  line: 1,                          // axis and error bar weight, points
  point: 5,                         // symbol size, points
  transparent: false,
};

// The colours of everything that is not a group's, as Prism prints them; a spec's `ink` overrides any.
export const INK = {
  mean: "#000000", error: "#000000", rule: "#6e6e6e", brackets: "#000000", title: "#000000",
  axisTitle: "#000000", text: "#000000", axes: "#000000", background: "#ffffff",
};

// Prism's symbols in turn, for the black palette: filled, then the same shapes open.
const SHAPES = ["circle", "square", "triangle", "down", "diamond"];
const GREYS = ["#ffffff", "#c8c8c8", "#8c8c8c", "#505050"];
const DASHES = [null, [4, 2], [1.2, 1.6], [6, 2, 1.5, 2]];
const GAP = 0.8;          // between clusters, in columns
const EDGE = 0.45;        // before the first column and after the last

// ---------------------------------------------------------------- significance

/** GraphPad's asterisks: **** below 0.0001, *** below 0.001, ** below 0.01, * below 0.05, else ns. */
export function stars(p) {
  if (p == null || !Number.isFinite(p)) return "";
  return p < 0.0001 ? "****" : p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "ns";
}

export function pText(p) {
  if (p == null || !Number.isFinite(p)) return "";
  return p < 0.0001 ? "p < 0.0001" : `p = ${p < 0.001 ? p.toFixed(4) : p.toFixed(3)}`;
}

// ---------------------------------------------------------------- text

// Arial's advance widths, in ems, for measuring without a canvas (Node).
function advance(ch) {
  if (/[0-9]/.test(ch)) return 0.556;
  if (/[ijl|'.,:;!]/.test(ch)) return 0.24;
  if (/[ frt()\-[\]/]/.test(ch)) return 0.31;
  if (/[mw]/.test(ch)) return 0.8;
  if (/[MW]/.test(ch)) return 0.9;
  if (/[A-Z]/.test(ch)) return 0.68;
  if (/[a-z]/.test(ch)) return 0.53;
  return 0.6;
}

let measureCtx = null;
/** How wide `text` is, in points, set in Arial at `size` points. */
export function textWidth(text, size, bold = false) {
  const str = String(text);
  if (typeof document !== "undefined") {
    measureCtx ||= document.createElement("canvas").getContext("2d");
    measureCtx.font = `${bold ? "bold " : ""}${size}px ${FONT}`;
    return measureCtx.measureText(str).width;
  }
  let w = 0;
  for (const ch of str) w += advance(ch);
  return w * size * (bold ? 1.06 : 1);
}

/** Text broken into lines no wider than `width`: units on a line of their own first, then word by word. */
export function wrap(text, width, size, bold = false, maxLines = 3) {
  const fits = (t) => textWidth(t, size, bold) <= width;
  if (!text || fits(text)) return text ? [text] : [];
  const paren = text.lastIndexOf(" (");
  if (paren > 0 && fits(text.slice(0, paren)) && fits(text.slice(paren + 1))) return [text.slice(0, paren), text.slice(paren + 1)];
  const lines = [];
  let line = "";
  for (const word of text.split(" ")) {
    const next = line ? `${line} ${word}` : word;
    if (line && !fits(next) && lines.length < maxLines - 1) { lines.push(line); line = word; } else line = next;
  }
  lines.push(line);
  return lines;
}

// ---------------------------------------------------------------- SVG

const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const r2 = (v) => String(Math.round(v * 100) / 100);
function el(tag, attrs, inner) {
  const a = Object.entries(attrs).filter(([, v]) => v != null && v !== false)
    .map(([k, v]) => ` ${k}="${typeof v === "number" ? r2(v) : esc(v)}"`).join("");
  return inner == null ? `<${tag}${a}/>` : `<${tag}${a}>${inner}</${tag}>`;
}
const dashArray = (d, lw) => (d ? d.map((v) => r2(v * lw)).join(" ") : null);

function text(x, y, str, { size, bold = false, anchor = "start", fill = "#000", rotate = null }) {
  return el("text", { x, y, "font-size": size, "font-weight": bold ? "bold" : null, "text-anchor": anchor === "start" ? null : anchor,
    fill, transform: rotate != null ? `rotate(${rotate} ${r2(x)} ${r2(y)})` : null }, esc(str));
}
const line = (x1, y1, x2, y2, stroke, width, extra = {}) => el("line", { x1, y1, x2, y2, stroke, "stroke-width": width, ...extra });

function symbol(shape, cx, cy, size, fill, stroke, sw) {
  const r = size / 2;
  const style = { fill, stroke: stroke || null, "stroke-width": stroke ? sw : null };
  if (shape === "circle") return el("circle", { cx, cy, r, ...style });
  if (shape === "square") return el("rect", { x: cx - r * 0.88, y: cy - r * 0.88, width: r * 1.76, height: r * 1.76, ...style });
  const pts = {
    triangle: [[0, -1.15], [1, 0.62], [-1, 0.62]],
    down: [[0, 1.15], [1, -0.62], [-1, -0.62]],
    diamond: [[0, -1.2], [1, 0], [0, 1.2], [-1, 0]],
  }[shape];
  return el("polygon", { points: pts.map(([x, y]) => `${r2(cx + x * r)},${r2(cy + y * r)}`).join(" "), ...style });
}

// ---------------------------------------------------------------- scales

function niceStep(span, count) {
  const raw = span / Math.max(count, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / mag;
  return mag * (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10);
}

/** An axis from one tick to another, as Prism autoscales it: {min, max, step, ticks}. */
export function niceScale(lo, hi, count = 5, fixed = false) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
  if (lo === hi) { const d = Math.abs(lo) * 0.1 || 1; lo -= d; hi += d; }
  const step = niceStep(hi - lo, count);
  const min = fixed ? lo : Math.floor(lo / step + 1e-9) * step;
  const max = fixed ? hi : Math.ceil(hi / step - 1e-9) * step;
  const ticks = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { min, max, step, ticks };
}

/** A tick label with as many decimals as its step needs, no thousands separators, a true minus. */
export function tickLabel(v, step) {
  // As many as the step's own digits: 0.25 apart needs two, 0.5 apart one.
  let decimals = 0;
  while (decimals < 6 && Math.abs(Math.round(step * 10 ** decimals) - step * 10 ** decimals) > 1e-6 * 10 ** decimals) decimals++;
  const s = Math.abs(v).toFixed(decimals);
  return v < 0 && Number(s) !== 0 ? `−${s}` : s;
}

function describe(values) {
  const n = values.length;
  if (!n) return { n, mean: null, sd: null };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : null;
  return { n, mean, sd };
}

// Points spread sideways just enough not to overlap, as Prism's scatter lays
// them out: each goes to the free spot nearest the column's centre.
export function swarm(ys, size, maxOffset) {
  const order = ys.map((_, i) => i).sort((a, b) => ys[a] - ys[b]);
  const placed = [];
  const xs = new Array(ys.length).fill(0);
  const d2 = size * size;
  for (const i of order) {
    const y = ys[i];
    const near = placed.filter((p) => Math.abs(p.y - y) < size);
    const options = [0];
    for (const p of near) {
      const dx = Math.sqrt(d2 - (p.y - y) ** 2);
      options.push(p.x + dx, p.x - dx);
    }
    options.sort((a, b) => Math.abs(a) - Math.abs(b) || a - b);
    const free = options.find((x) => near.every((p) => (p.x - x) ** 2 + (p.y - y) ** 2 >= d2 - 1e-6)) ?? 0;
    xs[i] = Math.max(-maxOffset, Math.min(maxOffset, free));
    placed.push({ x: xs[i], y });
  }
  return xs;
}

// ---------------------------------------------------------------- the figure

/**
 * The figure for `spec` with `options` (FIGURE_DEFAULTS): its size in points
 * and `svg(width, height)`, the document with those size attributes (inches
 * by default; pixels to rasterize it).
 */
export function renderFigure(spec, options = {}) {
  const o = { ...FIGURE_DEFAULTS, color: (c) => c, ...options };
  o.ink = { ...INK };
  for (const [k, v] of Object.entries(spec.ink || {})) if (v && k in INK) o.ink[k] = o.color(v);
  const W = o.width * 72, H = o.height * 72;
  const fig = spec.kind === "xy" ? xyFigure(spec, o, W, H) : columnFigure(spec, o, W, H);
  const svg = (width = `${o.width}in`, height = `${o.height}in`) => [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${r2(W)} ${r2(H)}" font-family="${FONT}">`,
    fig.defs.length ? `<defs>${fig.defs.join("")}</defs>` : "",
    o.transparent ? "" : el("rect", { width: W, height: H, fill: o.ink.background }),
    ...fig.body,
    "</svg>",
  ].join("\n");
  return { width: W, height: H, svg, layout: fig.layout };
}

/** A size to start from: wide enough for the columns or panels at the default font. */
export function defaultSize(spec) {
  const clamp = (v, lo, hi) => Math.round(Math.min(hi, Math.max(lo, v)) * 10) / 10;
  if (spec.kind === "xy") {
    const n = spec.panels.length, cols = n <= 3 ? n : n === 4 ? 2 : 3;
    return { width: cols === 1 ? 4.5 : 7.2, height: clamp(0.5 + 2.3 * Math.ceil(n / cols), 2.6, 9) };
  }
  if (!spec.splits.length) return { width: clamp(1.4 + 0.55 * (spec.strata[0]?.cells.length || 1), 2.2, 7.2), height: 2.6 };
  const slots = spec.strata.length * spec.levels.length;
  return { width: clamp(2.3 + 0.38 * slots + 0.25 * spec.strata.length, 3, 7.2), height: 2.6 };
}

// ---------------------------------------------------------------- the heading

// The heading above the plot, at 1.2 times the font in bold, on up to two lines: its lines and the height it takes.
function headingOf(spec, o, W) {
  const size = o.font * 1.2, pad = Math.max(2, o.font * 0.3);
  const lines = o.title && spec.heading ? wrap(spec.heading, W - 2 * pad, size, true, 2) : [];
  return { lines, size, height: lines.length ? lines.length * size * 1.15 + o.font * 0.35 : 0 };
}

// The heading drawn centred over the plot from x0 to x1, kept inside the figure.
function drawHeading(head, o, W, x0, x1) {
  const pad = Math.max(2, o.font * 0.3);
  return head.lines.map((l, i) => {
    const half = textWidth(l, head.size, true) / 2;
    const mid = Math.min(W - pad - half, Math.max(pad + half, (x0 + x1) / 2));
    return text(mid, pad + head.size * (0.85 + i * 1.15), l, { size: head.size, bold: true, anchor: "middle", fill: o.ink.title });
  });
}

// ---------------------------------------------------------------- column figures

function columnFigure(spec, o, W, H) {
  const fs = o.font, lw = o.line, pad = Math.max(2, fs * 0.3), tick = fs * 0.4;
  const black = o.palette === "black", bars = o.style === "bars";
  const ink = o.ink, head = headingOf(spec, o, W);
  const grouped = spec.splits.length > 0;
  const levels = spec.levels;
  const slotOf = new Map(levels.map((lv, i) => [lv.level, i]));
  const hue = (color) => o.color(color);
  const pointFill = (col) => (black ? (col.slot >= SHAPES.length ? "#ffffff" : "#000000") : hue(col.cell.color));
  const shapeOf = (col) => (black ? SHAPES[col.slot % SHAPES.length] : "circle");

  // The columns along the axis, in column units, and what labels each stretch.
  const cols = [], cats = [];
  let u = EDGE;
  if (grouped) {
    spec.strata.forEach((st, si) => {
      if (si) u += GAP;
      const from = u;
      levels.forEach((lv, li) => {
        const cell = st.cells.find((c) => c.level === lv.level);
        if (cell) cols.push({ at: u + 0.5, cell, si, slot: li });
        u += 1;
      });
      cats.push({ label: st.label, from, to: u });
    });
  } else {
    for (const cell of spec.strata[0]?.cells || []) {
      cols.push({ at: u + 0.5, cell, si: 0, slot: slotOf.get(cell.level) ?? 0 });
      cats.push({ label: cell.name, from: u, to: u + 1 });
      u += 1;
    }
  }
  const units = u + EDGE;
  for (const c of cols) {
    c.values = c.cell.values.map((v) => v.value).filter((v) => v != null && Number.isFinite(v));
    const d = describe(c.values);
    c.mean = d.mean;
    c.err = d.sd == null ? null : o.error === "sem" ? d.sd / Math.sqrt(d.n) : d.sd;
  }
  let lo = Infinity, hi = -Infinity;
  for (const c of cols) {
    for (const v of [...c.values, c.err != null ? c.mean - c.err : c.mean, c.err != null ? c.mean + c.err : c.mean]) {
      if (v != null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
  }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (bars || o.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }

  // The pairs compared within each stratum, each on a bracket.
  const pairs = [];
  if (o.significance !== "off") {
    spec.strata.forEach((st, si) => {
      for (const t of spec.tests || []) {
        if ((t.within || "") !== (st.label || "") || t.p == null) continue;
        if (!o.ns && t.p >= 0.05) continue;
        const a = cols.find((c) => c.si === si && c.cell.name === t.a);
        const b = cols.find((c) => c.si === si && c.cell.name === t.b);
        if (a && b) pairs.push({ a, b, si, label: o.significance === "p" ? pText(t.p) : stars(t.p), isStars: o.significance !== "p" && t.p < 0.05 });
      }
    });
  }

  // The legend, for clusters: a symbol or swatch and the group's name.
  const legendItems = grouped && levels.length > 1 ? levels.map((lv, i) => ({ lv, slot: i, w: fs * 1.25 + textWidth(lv.name, fs, false) })) : [];
  const legendW = legendItems.length ? Math.max(...legendItems.map((it) => it.w)) : 0;
  const legendRight = legendItems.length > 0 && legendW + fs < W * 0.3;
  let legendRows = [];
  if (legendItems.length && !legendRight) {
    let row = [], width = 0;
    for (const it of legendItems) {
      if (row.length && width + fs + it.w > W - 2 * pad) { legendRows.push(row); row = []; width = 0; }
      width += (row.length ? fs : 0) + it.w;
      row.push(it);
    }
    legendRows.push(row);
  }
  const legendTop = legendRows.length * fs * 1.35;

  // Axis labels: flat, on two lines, or turned 45° when neither fits.
  const labelBold = o.bold;
  const splitLabel = (t) => {
    const i = t.indexOf(", ");
    if (i > 0) return [t.slice(0, i + 1), t.slice(i + 2)];
    const words = t.split(" ");
    if (words.length < 2) return [t];
    let best = 1, bestDiff = Infinity;
    for (let k = 1; k < words.length; k++) {
      const diff = Math.abs(words.slice(0, k).join(" ").length - words.slice(k).join(" ").length);
      if (diff < bestDiff) { bestDiff = diff; best = k; }
    }
    return [words.slice(0, best).join(" "), words.slice(best).join(" ")];
  };
  const labelMode = (band) => {
    const room = (c) => (c.to - c.from) * band * 0.94;
    if (cats.every((c) => textWidth(c.label, fs, labelBold) <= room(c))) return { mode: "flat", height: fs * 1.25 };
    if (cats.every((c) => splitLabel(c.label).every((l) => textWidth(l, fs, labelBold) <= room(c)))) return { mode: "two", height: fs * 2.4 };
    const w = Math.max(0, ...cats.map((c) => textWidth(c.label, fs, labelBold)));
    return { mode: "turned", height: w * 0.7071 + fs * 1.0 };
  };

  const layout = (extraTop) => {
    let plotH = H * 0.6, left = 0, right = 0, top = 0, bottom = 0, scale, labels, yLines, band, mode, overhang = 0;
    for (let pass = 0; pass < 3; pass++) {
      scale = niceScale(lo, hi, Math.max(3, Math.min(8, Math.round(plotH / (fs * 2.6)))));
      labels = scale.ticks.map((t) => tickLabel(t, scale.step));
      yLines = wrap(spec.yLabel || "", Math.max(plotH, fs * 4), fs, o.bold);
      const tickW = Math.max(0, ...labels.map((l) => textWidth(l, fs)));
      left = pad + yLines.length * fs * 1.15 + fs * 0.35 + tickW + fs * 0.3 + tick + overhang;
      right = pad + (legendRight ? legendW + fs * 0.8 : fs * 0.3);
      band = (W - left - right) / units;
      mode = labelMode(band);
      if (mode.mode === "turned") {
        // A turned label reaches left of its column: keep the first inside the figure.
        const first = cats[0];
        const reach = textWidth(first.label, fs, labelBold) * 0.7071 + fs * 0.3;
        overhang = Math.max(overhang, reach - (left - overhang) - ((first.from + first.to) / 2) * band + pad);
        overhang = Math.max(0, overhang);
      }
      top = pad + fs * 0.5 + head.height + legendTop + extraTop;
      bottom = pad + fs * 0.35 + mode.height;
      plotH = H - top - bottom;
    }
    const x0 = left, x1 = W - right, y0 = top, y1 = H - bottom;
    const y = (v) => y1 - ((v - scale.min) / (scale.max - scale.min || 1)) * (y1 - y0);
    const cx = (at) => x0 + at * band;
    return { x0, x1, y0, y1, y, cx, band, scale, labels, yLines, mode };
  };

  // Brackets sit just over the data they compare, the shorter ones lower.
  const placeBrackets = (g) => {
    const size = o.point;
    const colTop = (c) => {
      const tops = [...c.values, c.err != null ? c.mean + c.err : c.mean].filter((v) => v != null);
      return Math.min(g.y(Math.max(...tops)) - size / 2, bars ? g.y(Math.max(c.mean, 0)) : Infinity);
    };
    const placed = [];
    const order = [...pairs].sort((p, q) => p.si - q.si || Math.abs(p.a.at - p.b.at) - Math.abs(q.a.at - q.b.at));
    // Each end stops a little short of its column's centre, so two brackets
    // meeting at one column sit side by side at one height.
    const inset = Math.min(g.band * 0.08, fs * 0.3);
    for (const p of order) {
      const ca = g.cx(Math.min(p.a.at, p.b.at)), cb = g.cx(Math.max(p.a.at, p.b.at));
      const xa = ca + inset, xb = cb - inset;
      // What the bracket takes up across: its line, or its label where that is wider.
      const half = textWidth(p.label, fs) / 2 + fs * 0.15, mid = (xa + xb) / 2;
      const left = Math.min(xa, mid - half), right = Math.max(xb, mid + half);
      let yb = Math.min(...cols.filter((c) => g.cx(c.at) >= ca - 0.1 && g.cx(c.at) <= cb + 0.1).map(colTop)) - fs * 0.45;
      for (const q of placed) if (q.left < right && q.right > left) yb = Math.min(yb, q.top - fs * 0.6);
      const baseline = p.isStars ? yb + fs * 0.3 : yb - fs * 0.22;
      const top = baseline - fs * (p.isStars ? 0.75 : 0.72);
      placed.push({ ...p, xa, xb, left, right, yb, baseline, top });
    }
    return placed;
  };

  let extra = 0, g = layout(0), brackets = placeBrackets(g);
  for (let i = 0; i < 5 && brackets.length; i++) {
    const over = pad + head.height - Math.min(...brackets.map((b) => b.top));
    if (over <= 0.25) break;
    extra += over;
    g = layout(extra);
    brackets = placeBrackets(g);
  }

  const body = [];
  const { x0, x1, y0, y1, y, cx, band, scale } = g;
  // Y axis: its line from the first tick to the last, ticks out, labels, title.
  body.push(...drawHeading(head, o, W, x0, x1));
  body.push(line(x0, y(scale.max), x0, y(scale.min), ink.axes, lw, { "stroke-linecap": "square" }));
  scale.ticks.forEach((t, i) => {
    body.push(line(x0 - tick, y(t), x0, y(t), ink.axes, lw));
    body.push(text(x0 - tick - fs * 0.3, y(t) + fs * 0.35, g.labels[i], { size: fs, anchor: "end", fill: ink.text }));
  });
  const titleX = pad + fs * 0.85;
  g.yLines.forEach((l, i) => {
    const tx = titleX + i * fs * 1.15, ty = (y0 + y1) / 2;
    body.push(text(tx, ty, l, { size: fs, bold: o.bold, anchor: "middle", rotate: -90, fill: ink.axisTitle }));
  });
  // X axis along the bottom, meeting the y axis at its corner.
  body.push(line(x0, y1, x1, y1, ink.axes, lw, { "stroke-linecap": "square" }));

  const size = o.point;
  for (const c of cols) {
    const at = cx(c.at);
    if (!c.values.length) continue;
    if (bars && c.mean != null) {
      const base = y(Math.max(scale.min, Math.min(scale.max, 0)));
      const half = Math.min(band * 0.36, fs * 2.4);
      const fill = black ? GREYS[c.slot % GREYS.length] : hue(c.cell.color);
      body.push(el("rect", { x: at - half, y: Math.min(base, y(c.mean)), width: half * 2, height: Math.abs(base - y(c.mean)),
        fill, "fill-opacity": black ? null : 0.28, stroke: black ? "#000" : hue(c.cell.color), "stroke-width": lw }));
    }
    const ys = c.values.map(y);
    const xs = swarm(ys, size * 1.08, band * (bars ? 0.3 : 0.38));
    const stroke = black && c.slot >= SHAPES.length ? "#000" : null;
    ys.forEach((py, i) => body.push(symbol(shapeOf(c), at + xs[i], py, size, pointFill(c), stroke, lw * 0.75)));
    if (c.mean == null) continue;
    const half = Math.min(band * 0.3, fs * 1.6), cap = half * 0.5;
    if (!bars) body.push(line(at - half, y(c.mean), at + half, y(c.mean), ink.mean, lw * 1.4));
    if (c.err != null && c.err > 0) {
      const ends = bars ? [c.mean >= 0 ? c.mean + c.err : c.mean - c.err] : [c.mean - c.err, c.mean + c.err];
      body.push(line(at, y(bars ? c.mean : c.mean - c.err), at, y(bars ? ends[0] : c.mean + c.err), ink.error, lw));
      for (const e of ends) body.push(line(at - cap, y(e), at + cap, y(e), ink.error, lw));
    }
  }

  // Group names (or strata, for clusters) under the axis.
  const labelY = y1 + fs * 0.35;
  for (const c of cats) {
    const mid = cx((c.from + c.to) / 2);
    if (g.mode.mode === "flat") body.push(text(mid, labelY + fs * 0.85, c.label, { size: fs, bold: labelBold, anchor: "middle", fill: ink.text }));
    else if (g.mode.mode === "two") {
      splitLabel(c.label).forEach((l, i) => body.push(text(mid, labelY + fs * (0.85 + i * 1.15), l, { size: fs, bold: labelBold, anchor: "middle", fill: ink.text })));
    } else {
      body.push(text(mid + fs * 0.3, labelY + fs * 0.45, c.label, { size: fs, bold: labelBold, anchor: "end", rotate: -45, fill: ink.text }));
    }
  }

  for (const b of brackets) {
    const drop = fs * 0.35;
    body.push(el("path", { d: `M${r2(b.xa)},${r2(b.yb + drop)}V${r2(b.yb)}H${r2(b.xb)}V${r2(b.yb + drop)}`, fill: "none", stroke: ink.brackets, "stroke-width": lw }));
    body.push(text((b.xa + b.xb) / 2, b.baseline, b.label, { size: fs, anchor: "middle", fill: ink.brackets }));
  }

  // Legend: at the right beside the plot, or across the top when that would crowd it.
  const legendMark = (it, x, yMid) => {
    if (bars) {
      const fill = black ? GREYS[it.slot % GREYS.length] : hue(it.lv.color);
      return el("rect", { x, y: yMid - fs * 0.38, width: fs * 0.76, height: fs * 0.76, fill, "fill-opacity": black ? null : 0.28,
        stroke: black ? "#000" : hue(it.lv.color), "stroke-width": lw });
    }
    const col = { slot: it.slot, cell: { color: it.lv.color } };
    return symbol(shapeOf(col), x + fs * 0.38, yMid, size, pointFill(col), black && it.slot >= SHAPES.length ? "#000" : null, lw * 0.75);
  };
  if (legendRight) {
    legendItems.forEach((it, i) => {
      const lx = W - pad - legendW, ly = y0 + fs * 0.6 + i * fs * 1.35;
      body.push(legendMark(it, lx, ly), text(lx + fs * 1.25, ly + fs * 0.35, it.lv.name, { size: fs, fill: ink.text }));
    });
  } else {
    legendRows.forEach((row, r) => {
      let lx = pad;
      const ly = pad + head.height + fs * 0.6 + r * fs * 1.35;
      for (const it of row) {
        body.push(legendMark(it, lx, ly), text(lx + fs * 1.25, ly + fs * 0.35, it.lv.name, { size: fs, fill: ink.text }));
        lx += it.w + fs;
      }
    });
  }
  return { defs: [], body, layout: { plot: [x0, y0, x1, y1], scale, brackets: brackets.map((b) => ({ label: b.label, top: b.top, yb: b.yb })), labels: g.mode.mode,
    heading: head.lines } };
}

// ---------------------------------------------------------------- XY figures

function xyFigure(spec, o, W, H) {
  const fs = o.font, lw = o.line, pad = Math.max(2, fs * 0.3), tick = fs * 0.4;
  const black = o.palette === "black";
  const ink = o.ink, head = headingOf(spec, o, W);
  const panels = spec.panels;
  const P = Math.max(1, panels.length);
  const C = Math.min(P, o.columns || (P <= 3 ? P : P === 4 ? 2 : 3)), R = Math.ceil(P / C);
  const colorOf = (s) => (black ? "#000000" : o.color(s.color));
  const dashOf = (s) => (black ? DASHES[(s.slot || 0) % DASHES.length] : null);

  // Legend across the top, wrapping.
  const items = spec.legend.length > 1 ? spec.legend.map((it) => ({ ...it, w: fs * 2.2 + textWidth(it.label, fs) })) : [];
  const rows = [];
  let row = [], width = 0;
  for (const it of items) {
    if (row.length && width + fs * 1.2 + it.w > W - 2 * pad) { rows.push(row); row = []; width = 0; }
    width += (row.length ? fs * 1.2 : 0) + it.w;
    row.push(it);
  }
  if (row.length) rows.push(row);
  const legendH = rows.length * fs * 1.35;

  const titleH = panels.some((p) => p.title) ? fs * 1.4 : 0;
  const marksH = panels.some((p) => p.spans?.length) ? fs * 1.15 : 0;
  const gridTop = pad + head.height + legendH, gridBottom = H - pad - fs * 1.35;
  const cellH = (gridBottom - gridTop) / R;
  const rowGap = R > 1 ? fs * 0.6 : 0;
  const plotH = Math.max(fs * 3, cellH - titleH - marksH - fs * 0.5 - (tick + fs * 1.35) - rowGap);
  const ys = niceScale(spec.yDomain[0], spec.yDomain[1], Math.max(3, Math.min(8, Math.round(plotH / (fs * 2.6)))));
  const yLabels = ys.ticks.map((t) => tickLabel(t, ys.step));
  const tickW = Math.max(0, ...yLabels.map((l) => textWidth(l, fs)));
  const yLines = wrap(spec.yLabel || "", Math.max(fs * 4, gridBottom - gridTop), fs, o.bold);
  const gridLeft = pad + yLines.length * fs * 1.15 + fs * 0.25;
  const cellW = (W - pad - gridLeft) / C;
  const inset = tickW + fs * 0.3 + tick;
  const rightPad = fs * 1.1;
  const plotW = Math.max(fs * 4, cellW - inset - rightPad);
  const [xa, xb] = spec.xDomain || [0, 100];
  const xs = niceScale(xa, xb, Math.max(2, Math.min(6, Math.round(plotW / (fs * 4)))), true);

  const defs = [], body = [];
  const plots = [];
  panels.forEach((p, i) => {
    const r = Math.floor(i / C), c = i % C;
    const cx0 = gridLeft + c * cellW, cy0 = gridTop + r * cellH;
    const x0 = cx0 + inset, x1 = x0 + plotW;
    const y0 = cy0 + titleH + marksH + fs * 0.5, y1 = y0 + plotH;
    plots.push([x0, y0, x1, y1]);
    const x = (v) => x0 + ((v - xa) / (xb - xa || 1)) * plotW;
    const y = (v) => y1 - ((v - ys.min) / (ys.max - ys.min || 1)) * plotH;
    const clampX = (v) => Math.min(x1, Math.max(x0, x(v)));
    if (p.title) body.push(text((x0 + x1) / 2, cy0 + fs * 0.95, p.title, { size: fs, bold: o.bold, anchor: "middle", fill: ink.text }));

    // Landmark bands behind everything, in their own colours, labelled above the plot.
    const marks = [];
    (p.spans || []).forEach((sp, k) => {
      const tone = o.color(sp.color || `var(--lm-${sp.tone || 0})`);
      const { stops, spread, means } = landmarkStops(sp);
      if (stops) {
        const lo = stops[0][0], hi = stops[stops.length - 1][0];
        const left = clampX(lo), right = clampX(hi);
        if (right > left) {
          let fill = tone;
          if (spread) {
            const id = `lm${i}-${k}`;
            defs.push(el("linearGradient", { id, gradientUnits: "userSpaceOnUse", x1: x(lo), x2: x(hi), y1: 0, y2: 0 },
              stops.map(([at, strength]) => el("stop", { offset: ((at - lo) / (hi - lo)).toFixed(4), "stop-color": tone, "stop-opacity": strength })).join("")));
            fill = `url(#${id})`;
          }
          body.push(el("rect", { x: left, y: y0, width: right - left, height: plotH, fill, opacity: BAND_OPACITY }));
        }
      }
      for (const mean of means) {
        if (Math.abs(clampX(mean) - x(mean)) > 0.5) continue;
        body.push(line(x(mean), y0, x(mean), y1, tone, lw));
      }
      if (sp.label) marks.push({ text: sp.label, cx: (clampX(sp.x0) + clampX(sp.x1 ?? sp.x0)) / 2, w: textWidth(sp.label, fs * 0.85) });
    });
    for (const lb of spreadLabels(marks, cx0 + 2, cx0 + cellW - 2, fs * 0.5)) {
      body.push(text(lb.left, y0 - fs * 0.3, lb.text, { size: fs * 0.85, fill: spec.ink?.text ? ink.text : "#333333" }));
    }

    for (const rule of spec.rules || []) {
      if (rule.y == null || rule.y < ys.min || rule.y > ys.max) continue;
      body.push(line(x0, y(rule.y), x1, y(rule.y), ink.rule, lw * 0.75, { "stroke-dasharray": dashArray([3, 2], lw) }));
      if (rule.label) body.push(text(x1, y(rule.y) - fs * 0.3, rule.label, { size: fs * 0.8, anchor: "end", fill: ink.rule }));
    }

    const path = (points) => {
      let d = "", pen = false;
      for (const [px, py] of points) {
        if (py == null || !Number.isFinite(py)) { pen = false; continue; }
        d += `${pen ? "L" : "M"}${r2(x(px))},${r2(y(py))}`;
        pen = true;
      }
      return d;
    };
    for (const s of p.series.filter((q) => q.band?.length)) {
      const runs = [[]];
      for (const b of s.band) {
        if (b[1] != null && b[2] != null) runs[runs.length - 1].push(b);
        else if (runs[runs.length - 1].length) runs.push([]);
      }
      for (const run of runs.filter((q) => q.length > 1)) {
        const d = run.map((b, k) => `${k ? "L" : "M"}${r2(x(b[0]))},${r2(y(b[2]))}`).join("")
          + run.slice().reverse().map((b) => `L${r2(x(b[0]))},${r2(y(b[1]))}`).join("") + "Z";
        body.push(el("path", { d, fill: black ? "#000000" : o.color(s.bandColor || s.color), "fill-opacity": black ? 0.1 : 0.2, stroke: "none" }));
      }
    }
    const draw = (s, width, extra) => {
      const d = path(s.points);
      if (d) body.push(el("path", { d, fill: "none", "stroke-width": width, "stroke-linejoin": "round", "stroke-linecap": "round", ...extra }));
    };
    for (const s of p.series.filter((q) => q.role === "specimen")) draw(s, lw * 0.5, { stroke: black ? "#000000" : colorOf(s), "stroke-opacity": black ? 0.25 : 0.4 });
    for (const s of p.series.filter((q) => q.role === "dentin")) {
      draw(s, lw, { stroke: black ? "#7f7f7f" : colorOf(s), "stroke-dasharray": dashArray(black ? dashOf(s) || [3, 2] : [3, 2], lw), "stroke-linecap": "butt" });
    }
    for (const s of p.series.filter((q) => q.role === "mean")) draw(s, lw * 1.5, { stroke: colorOf(s), "stroke-dasharray": dashArray(dashOf(s), lw) });

    // Axes, ticks out.
    body.push(line(x0, y(ys.max), x0, y1, ink.axes, lw, { "stroke-linecap": "square" }));
    body.push(line(x0, y1, x1, y1, ink.axes, lw, { "stroke-linecap": "square" }));
    ys.ticks.forEach((t, k) => {
      body.push(line(x0 - tick, y(t), x0, y(t), ink.axes, lw));
      body.push(text(x0 - tick - fs * 0.3, y(t) + fs * 0.35, yLabels[k], { size: fs, anchor: "end", fill: ink.text }));
    });
    for (const t of xs.ticks) {
      body.push(line(x(t), y1, x(t), y1 + tick, ink.axes, lw));
      body.push(text(x(t), y1 + tick + fs * 1.0, tickLabel(t, xs.step), { size: fs, anchor: "middle", fill: ink.text }));
    }
  });

  // One x title under the grid, one y title beside it.
  const left = gridLeft + inset, right = gridLeft + C * cellW - rightPad;
  body.push(...drawHeading(head, o, W, left, right));
  if (spec.xLabel) body.push(text((left + right) / 2, H - pad - fs * 0.25, spec.xLabel, { size: fs, bold: o.bold, anchor: "middle", fill: ink.axisTitle }));
  const top = plots.length ? plots[0][1] : gridTop, bottom = plots.length ? plots[plots.length - 1][3] : gridBottom;
  yLines.forEach((l, i) => body.push(text(pad + fs * 0.85 + i * fs * 1.15, (top + bottom) / 2, l, { size: fs, bold: o.bold, anchor: "middle", rotate: -90, fill: ink.axisTitle })));

  // The legend: a stretch of each line and its name.
  rows.forEach((items, r) => {
    let lx = gridLeft + inset;
    const ly = pad + head.height + fs * 0.6 + r * fs * 1.35;
    for (const it of items) {
      const stroke = it.dentin ? (black ? "#7f7f7f" : o.color(it.color)) : black ? "#000000" : o.color(it.color);
      const dash = it.dentin ? [3, 2] : black ? DASHES[(it.slot || 0) % DASHES.length] : null;
      body.push(line(lx, ly, lx + fs * 1.8, ly, stroke, it.dentin ? lw : lw * 1.5, { "stroke-dasharray": dashArray(dash, lw) }));
      body.push(text(lx + fs * 2.2, ly + fs * 0.35, it.label, { size: fs, fill: ink.text }));
      lx += it.w + fs * 1.2;
    }
  });
  return { defs, body, layout: { plots, scale: ys, xscale: xs, heading: head.lines } };
}
