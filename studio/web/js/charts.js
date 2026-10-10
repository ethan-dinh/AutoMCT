// Small SVG charts. The marks are thin and the chrome recessive; colour only
// ever marks identity (a group, a series), and every value is also reachable
// through a tooltip, a direct label or the chart's table view.

import { h, icon, replace, showTip, hideTip, sig, cssColor } from "./lib.js";

const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
  return el;
};

export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (count / span) * step0;
  const step = step0 * (err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

export function extent(values, pad = 0.05) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (v != null && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!Number.isFinite(lo)) return [0, 1];
  const span = hi - lo || Math.abs(hi) || 1;
  return [lo - span * pad, hi + span * pad];
}

const linear = (d0, d1, r0, r1) => {
  const k = (r1 - r0) / ((d1 - d0) || 1);
  const f = (v) => r0 + (v - d0) * k;
  f.invert = (p) => d0 + (p - r0) / k;
  return f;
};

// A chart that re-renders to its container's width.
function responsive(render, minHeight) {
  const wrap = h("div", { class: "chart", style: { minHeight: `${minHeight}px` } });
  let last = 0;
  const ro = new ResizeObserver((entries) => {
    const w = Math.floor(entries[0].contentRect.width);
    if (w > 0 && Math.abs(w - last) > 1) {
      last = w;
      replace(wrap, render(w));
    }
  });
  ro.observe(wrap);
  return wrap;
}

function tickStep(ticks) {
  return ticks.length > 1 ? ticks[1] - ticks[0] : null;
}

// Room for the longest y tick label, so a rotated axis title never overlaps it.
function leftMargin(labels, min = 44) {
  const longest = Math.max(0, ...labels.map((l) => String(l).length));
  return Math.max(min, Math.round(24 + longest * 6.6));
}

function axes(svg, { x, y, width, height, m, xTicks, yTicks, xLabel, yLabel, fx, fy }) {
  const ys = tickStep(yTicks), xs = tickStep(xTicks);
  for (const t of yTicks) {
    svg.appendChild(s("line", { class: "grid", x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }));
    const label = s("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" });
    label.textContent = fy(t, ys);
    svg.appendChild(label);
  }
  svg.appendChild(s("line", { class: "axis", x1: m.l, x2: width - m.r, y1: height - m.b, y2: height - m.b }));
  for (const t of xTicks) {
    const label = s("text", { x: x(t), y: height - m.b + 16, "text-anchor": "middle" });
    label.textContent = fx(t, xs);
    svg.appendChild(label);
  }
  if (xLabel) {
    const t = s("text", { class: "axis-title", x: m.l + (width - m.l - m.r) / 2, y: height - 4, "text-anchor": "middle" });
    t.textContent = xLabel;
    svg.appendChild(t);
  }
  if (yLabel) {
    const t = s("text", { class: "axis-title", x: -(m.t + (height - m.t - m.b) / 2), y: 13, transform: "rotate(-90)", "text-anchor": "middle" });
    t.textContent = yLabel;
    svg.appendChild(t);
  }
}

function tipRows(head, rows) {
  return h("div", null,
    head ? h("div", { class: "t-head" }, head) : null,
    rows.map((r) => h("div", { class: "t-row" },
      r.dash ? h("i", { class: "key dash", style: { borderTopColor: r.color } })
        : h("i", { class: `key ${r.dot ? "dot" : ""}`, style: { background: r.color } }),
      h("b", null, r.value),
      h("span", null, r.label || ""))));
}

// ---------------------------------------------------------------- line chart

/**
 * series: [{id, label, color, points: [[x, y], ...], width, opacity, dash, band: [[x, lo, hi]], tip: bool}]
 * spans:  [{x0, x1, label, tone, color, sd0, sd1, detail}]  shaded vertical ranges (landmarks), each in
 *         its own colour (`color`, or `landmarkTone`'s); sd0/sd1, each end's standard deviation across
 *         specimens, draw a strip about it (see `landmarkBand`), and `detail` joins the crosshair's
 *         tooltip on it. A series' `bandColor`, when given, fills its band instead of its `color`.
 * rules:  [{y, label}] horizontal reference lines; [{x, label}] vertical ones
 */
// One row of labels above the plot, each centred on its mark where there is
// room. Labels that would overlap are pushed apart, and none runs past lo..hi.
// Widths are estimated (11px text), which errs on the side of more space.
export function spreadLabels(labels, lo, hi, gap = 8) {
  const sorted = [...labels].sort((a, b) => a.cx - b.cx);
  let edge = lo;
  for (const lb of sorted) { lb.left = Math.max(lb.cx - lb.w / 2, edge); edge = lb.left + lb.w + gap; }
  edge = hi;
  for (const lb of sorted.reverse()) { lb.left = Math.min(lb.left, edge - lb.w); edge = lb.left - gap; }
  return labels;
}

// ---------------------------------------------------------------- landmark bands

export const BAND_OPACITY = 0.25;
// How strong the strip one SD either side of a mean end is, against the band's full colour.
const SPREAD = 0.45;

// Gradient ids must be unique in the page, and a chart redraws on every resize.
let gradientCount = 0;

/**
 * How a landmark is coloured along the axis: `stops`, [position, strength]
 * in order (null when nothing is filled), whether any end has a spread to
 * draw, and the mean ends a line marks. The page's charts and the exported
 * figures (figure.js) both draw a band from these.
 *
 * A line marks each mean end, in a lighter strip one standard deviation
 * either side of it (sd0, sd1: how far that end moves between specimens), as
 * the curves' ribbons are one SD either side of their means; the band is in
 * full colour between the strips. So a strip's width is the spread of the end
 * it straddles, and where two molars meet, both strips show. An event
 * (eruption) is a line in its own strip. No spread, no strip.
 */
export function landmarkStops(sp) {
  const isSpan = sp.x1 != null && sp.x1 > sp.x0;
  const sd0 = sp.sd0 > 0 ? sp.sd0 : 0;
  const sd1 = isSpan && sp.sd1 > 0 ? sp.sd1 : 0;
  let stops;
  if (isSpan) {
    const inner0 = sp.x0 + sd0, inner1 = sp.x1 - sd1;
    const meet = (inner0 + inner1) / 2;     // strips wider than the band meet in its middle
    const a = Math.min(inner0, meet), b = Math.max(inner1, meet);
    stops = [[sp.x0 - sd0, sd0 ? SPREAD : 1], [a, sd0 ? SPREAD : 1], [a, 1], [b, 1], [b, sd1 ? SPREAD : 1], [sp.x1 + sd1, sd1 ? SPREAD : 1]];
  } else {
    stops = sd0 ? [[sp.x0 - sd0, SPREAD], [sp.x0 + sd0, SPREAD]] : null;
  }
  return { stops, spread: Boolean(sd0 || sd1), means: isSpan ? [sp.x0, sp.x1] : [sp.x0] };
}

/** A landmark's colour: the one picked for it, or its --lm-N tone. */
const spanColor = (sp) => sp.color || `var(--lm-${sp.tone || 0})`;

/** One landmark drawn across the plot, from `top` to `bottom`, in its colour (see landmarkStops). */
function landmarkBand(svg, sp, x, top, bottom, clampX) {
  const tone = spanColor(sp);
  const { stops, spread, means } = landmarkStops(sp);
  if (stops) {
    const lo = stops[0][0], hi = stops[stops.length - 1][0];
    const left = clampX(lo), right = clampX(hi);
    if (right > left) {
      let fill = tone;
      if (spread) {
        let defs = svg.querySelector("defs");
        if (!defs) { defs = s("defs"); svg.insertBefore(defs, svg.firstChild); }
        const id = `landmark-spread-${++gradientCount}`;
        const g = s("linearGradient", { id, gradientUnits: "userSpaceOnUse", x1: x(lo), x2: x(hi), y1: 0, y2: 0 });
        for (const [at, strength] of stops) {
          g.appendChild(s("stop", { offset: ((at - lo) / (hi - lo)).toFixed(4), style: `stop-color: ${tone}; stop-opacity: ${strength}` }));
        }
        defs.appendChild(g);
        fill = `url(#${id})`;
      }
      svg.appendChild(s("rect", { x: left, y: top, width: right - left, height: bottom - top, fill, opacity: BAND_OPACITY }));
    }
  }
  for (const mean of means) {
    const at = x(mean);
    if (Math.abs(clampX(mean) - at) > 0.5) continue;           // off the chart
    svg.appendChild(s("line", { x1: at, x2: at, y1: top, y2: bottom, stroke: tone, "stroke-width": 1.5 }));
  }
}

/** The colour a landmark is drawn in (--lm-N): M1 green, M2 violet, M3 magenta, eruption yellow, anything else grey (0). */
export function landmarkTone(name) {
  return { M1: 1, M2: 2, M3: 3, eruption: 4 }[name] || 0;
}

export function lineChart({ series, spans = [], rules = [], xLabel, yLabel, xDomain, yDomain,
                            fx = (v) => sig(v, 3), fy = (v) => sig(v, 3), height = 300, tipHead }) {
  return responsive((width) => {
    const all = series.flatMap((sr) => sr.points);
    const xd = xDomain || extent(all.map((p) => p[0]), 0);
    const yd = yDomain || extent([...all.map((p) => p[1]), ...series.flatMap((sr) => (sr.band || []).flatMap((b) => [b[1], b[2]]))]);
    const yTicks = niceTicks(yd[0], yd[1], 5);
    const m = { l: leftMargin(yTicks.map((t) => fy(t, tickStep(yTicks))), 54), r: 18, t: 16, b: 42 };
    const x = linear(xd[0], xd[1], m.l, width - m.r);
    const y = linear(yd[0], yd[1], height - m.b, m.t);
    const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img" });
    axes(svg, { x, y, width, height, m, xTicks: niceTicks(xd[0], xd[1], Math.max(3, Math.floor(width / 90))),
      yTicks, xLabel, yLabel, fx, fy });

    // Each landmark in its own colour, see-through, so that where two overlap
    // both show, with how much its ends vary drawn about them.
    const labels = [];
    const clampX = (v) => Math.min(width - m.r, Math.max(m.l, x(v)));
    for (const sp of spans) {
      landmarkBand(svg, sp, x, m.t, height - m.b, clampX);
      const x0 = clampX(sp.x0), x1 = clampX(sp.x1 ?? sp.x0);
      if (sp.label) labels.push({ text: sp.label, cx: (x0 + x1) / 2, w: sp.label.length * 6.2 });
    }
    for (const lb of spreadLabels(labels, 2, width - 2)) {
      const t = s("text", { class: "band-label", x: lb.left, y: m.t - 4, "text-anchor": "start" });
      t.textContent = lb.text;
      svg.appendChild(t);
    }
    for (const r of rules) {
      if (r.y != null && r.y >= yd[0] && r.y <= yd[1]) {
        svg.appendChild(s("line", { class: "rule", x1: m.l, x2: width - m.r, y1: y(r.y), y2: y(r.y) }));
        if (r.label) {
          const t = s("text", { class: "label rule-label", x: width - m.r, y: y(r.y) - 5, "text-anchor": "end" });
          t.textContent = r.label;
          svg.appendChild(t);
        }
      }
    }

    const drawn = [];
    for (const sr of series) {
      if (sr.band && sr.band.length) {
        // One shape per unbroken run: a band never bridges a stretch where it has no spread.
        const runs = [[]];
        for (const b of sr.band) {
          if (b[1] != null && b[2] != null) runs[runs.length - 1].push(b);
          else if (runs[runs.length - 1].length) runs.push([]);
        }
        for (const good of runs.filter((r) => r.length > 1)) {
          const d = good.map((b, i) => `${i ? "L" : "M"}${x(b[0]).toFixed(1)},${y(b[2]).toFixed(1)}`).join("") +
            good.slice().reverse().map((b) => `L${x(b[0]).toFixed(1)},${y(b[1]).toFixed(1)}`).join("") + "Z";
          svg.appendChild(s("path", { d, fill: sr.bandColor || sr.color, opacity: 0.12 }));
        }
      }
      let d = "", pen = false;
      for (const [px, py] of sr.points) {
        if (py == null || !Number.isFinite(py)) { pen = false; continue; }
        d += `${pen ? "L" : "M"}${x(px).toFixed(1)},${y(py).toFixed(1)}`;
        pen = true;
      }
      svg.appendChild(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": sr.width ?? 2,
        "stroke-linejoin": "round", "stroke-linecap": sr.dash ? "butt" : "round", "stroke-dasharray": sr.dash,
        opacity: sr.opacity ?? 1 }));
      if (sr.markers) {
        for (const [px, py] of sr.points) {
          if (py == null) continue;
          svg.appendChild(s("circle", { cx: x(px), cy: y(py), r: 3, fill: sr.color, stroke: "var(--surface)", "stroke-width": 1.5 }));
        }
      }
      if (sr.tip !== false) drawn.push(sr);
    }

    // Crosshair: the pointer picks an x; every series reports its value there.
    const cross = s("g", { style: "display:none" });
    const vline = s("line", { class: "crosshair", y1: m.t, y2: height - m.b });
    cross.appendChild(vline);
    const dots = s("g");
    cross.appendChild(dots);
    svg.appendChild(cross);
    const hit = s("rect", { class: "hit", x: m.l, y: m.t, width: Math.max(0, width - m.l - m.r), height: Math.max(0, height - m.t - m.b) });
    const leave = () => { cross.style.display = "none"; hideTip(); };
    hit.addEventListener("pointerleave", leave);
    hit.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * width;
      const xv = x.invert(px);
      const rows = [];
      replace(dots);
      for (const sr of drawn) {
        const p = nearest(sr.points, xv);
        if (!p || p[1] == null) continue;
        if (Math.abs(x(p[0]) - px) > 40) continue;
        rows.push({ color: sr.color, value: fy(p[1]), label: sr.label, y: p[1], dash: !!sr.dash });
        dots.appendChild(s("circle", { cx: x(p[0]), cy: y(p[1]), r: 4, fill: sr.color, stroke: "var(--surface)", "stroke-width": 2 }));
      }
      if (!rows.length) { leave(); return; }
      rows.sort((a, b) => b.y - a.y);
      // The landmark the pointer is on, with its spread, after the series.
      const near = 4 / Math.max(x(1) - x(0), 1e-9);
      const marks = spans.filter((sp) => sp.detail
        && xv >= sp.x0 - (sp.sd0 || 0) - near
        && xv <= (sp.x1 ?? sp.x0) + ((sp.x1 != null ? sp.sd1 : sp.sd0) || 0) + near)
        .map((sp) => ({ color: spanColor(sp), value: sp.label, label: sp.detail }));
      vline.setAttribute("x1", px); vline.setAttribute("x2", px);
      cross.style.display = "";
      showTip(e.clientX, e.clientY, tipRows(tipHead ? tipHead(xv) : `${xLabel || "x"} ${fx(xv)}`, [...rows.slice(0, 14), ...marks]));
    });
    svg.appendChild(hit);
    return svg;
  }, height);
}

function nearest(points, xv) {
  let lo = 0, hi = points.length - 1;
  if (hi < 0) return null;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] <= xv) lo = mid; else hi = mid;
  }
  return Math.abs(points[lo][0] - xv) <= Math.abs(points[hi][0] - xv) ? points[lo] : points[hi];
}

// ---------------------------------------------------------------- strip plot

// Text cut to fit a width, at the 12px label size's estimated 6.6px a character.
function fit(text, width) {
  const room = Math.floor(width / 6.6);
  return text.length <= room ? text : room > 2 ? `${text.slice(0, room - 1)}…` : "";
}

// The mean bar and SD whisker: a page's own colours (--chart-mean, --chart-error) where it sets them.
const MEAN_INK = "var(--chart-mean, var(--ink))";
const ERROR_INK = "var(--chart-error, var(--ink-2))";

/**
 * One column per group: each specimen a dot, the group mean a short bar and
 * +/- 1 SD a whisker. groups: [{name, color, values: [{label, value}], cluster}]
 *
 * Consecutive groups with the same `cluster` sit together under its label,
 * with a gap between clusters: the columns of a comparison side by side, the
 * comparison repeated once per cluster.
 */
export function stripPlot({ groups, yLabel, fy = (v) => sig(v, 3), height = 260, yDomain, zero = false }) {
  const clustered = groups.some((g) => g.cluster);
  return responsive((width) => {
    const all = groups.flatMap((g) => g.values.map((v) => v.value));
    let yd = yDomain || extent(all, 0.12);
    if (zero && yd[0] > 0) yd = [0, yd[1]];
    const ticks = niceTicks(yd[0], yd[1], 5), step = tickStep(ticks);
    const m = { l: leftMargin(ticks.map((t) => fy(t, step)), 58), r: 12, t: 12, b: clustered ? 76 : 58 };
    // Column centres, in units of one column, with 0.6 of a column between clusters.
    const GAP = 0.6;
    const at = [];
    const clusters = [];
    let units = 0;
    groups.forEach((g, gi) => {
      if (gi && clustered && g.cluster !== groups[gi - 1].cluster) units += GAP;
      if (!gi || g.cluster !== groups[gi - 1].cluster) clusters.push({ label: g.cluster || "", first: gi, last: gi });
      clusters[clusters.length - 1].last = gi;
      at.push(units + 0.5);
      units += 1;
    });
    const band = (width - m.l - m.r) / Math.max(units, 1);
    // A cluster label too wide for its columns breaks after its first detail ("Female," / "P22").
    const room = (c) => (at[c.last] - at[c.first] + 1 + GAP) * band;
    const lines = (c) => (c.label.length * 6.6 > room(c) && c.label.includes(", ")
      ? [c.label.slice(0, c.label.indexOf(", ") + 1), c.label.slice(c.label.indexOf(", ") + 2)] : [c.label]);
    const twoLines = clustered && clusters.some((c) => lines(c).length > 1);
    if (twoLines) m.b += 15;
    const fullHeight = height + (clustered ? 18 : 0) + (twoLines ? 15 : 0);
    const y = linear(yd[0], yd[1], fullHeight - m.b, m.t);
    const svg = s("svg", { viewBox: `0 0 ${width} ${fullHeight}`, height: fullHeight, role: "img" });
    for (const t of ticks) {
      svg.appendChild(s("line", { class: "grid", x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }));
      const label = s("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" });
      label.textContent = fy(t, step);
      svg.appendChild(label);
    }
    svg.appendChild(s("line", { class: "axis", x1: m.l, x2: width - m.r, y1: fullHeight - m.b, y2: fullHeight - m.b }));
    if (yLabel) {
      const t = s("text", { class: "axis-title", x: -(m.t + (fullHeight - m.t - m.b) / 2), y: 13, transform: "rotate(-90)", "text-anchor": "middle" });
      t.textContent = yLabel;
      svg.appendChild(t);
    }
    if (clustered) {
      clusters.forEach((c, ci) => {
        const cx = m.l + band * (at[c.first] + at[c.last]) / 2;
        const label = s("text", { class: "label cluster", x: cx, y: fullHeight - m.b + 64, "text-anchor": "middle" });
        lines(c).forEach((line, li) => {
          const span = s("tspan", { x: cx, dy: li ? 15 : 0 });
          span.textContent = fit(line, room(c));
          label.appendChild(span);
        });
        const title = s("title");
        title.textContent = c.label;
        label.appendChild(title);
        svg.appendChild(label);
        if (ci + 1 < clusters.length) {
          const xs = m.l + band * (at[c.last] + 0.5 + GAP / 2);
          svg.appendChild(s("line", { class: "grid", x1: xs, x2: xs, y1: m.t, y2: fullHeight - m.b + (twoLines ? 85 : 70) }));
        }
      });
    }
    groups.forEach((g, gi) => {
      const cx = m.l + band * at[gi];
      svg.appendChild(s("rect", { x: cx - 4.5, y: fullHeight - m.b + 8, width: 9, height: 9, rx: 2, fill: g.color }));
      const name = s("text", { class: "label", x: cx, y: fullHeight - m.b + 31, "text-anchor": "middle" });
      name.textContent = fit(g.name, band - 4);
      svg.appendChild(name);
      const n = s("text", { x: cx, y: fullHeight - m.b + 46, "text-anchor": "middle" });
      n.textContent = band > 52 ? `n = ${g.values.length}` : `${g.values.length}`;
      svg.appendChild(n);
      const vals = g.values.map((v) => v.value).filter((v) => v != null);
      if (vals.length) {
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const sd = vals.length > 1 ? Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (vals.length - 1)) : 0;
        const reach = Math.min(22, band * 0.42);
        if (sd > 0) {
          svg.appendChild(s("line", { x1: cx + reach * 0.8, x2: cx + reach * 0.8, y1: y(mean - sd), y2: y(mean + sd), stroke: ERROR_INK, "stroke-width": 1 }));
          for (const v of [mean - sd, mean + sd]) svg.appendChild(s("line", { x1: cx + reach * 0.8 - 4, x2: cx + reach * 0.8 + 4, y1: y(v), y2: y(v), stroke: ERROR_INK, "stroke-width": 1 }));
        }
        svg.appendChild(s("line", { x1: cx - reach * 0.75, x2: cx + reach, y1: y(mean), y2: y(mean), stroke: MEAN_INK, "stroke-width": 2, "stroke-linecap": "round" }));
      }
      const tipHead = g.cluster ? `${g.name}, ${g.cluster}` : g.name;
      g.values.forEach((v) => {
        if (v.value == null) return;
        const jitter = (hash(v.label) - 0.5) * Math.min(band * 0.35, 24);
        const px = cx - 4 + jitter, py = y(v.value);
        svg.appendChild(s("circle", { cx: px, cy: py, r: 4.5, fill: g.color, stroke: "var(--surface)", "stroke-width": 2 }));
        const hitDot = s("circle", { class: "hit", cx: px, cy: py, r: 12, tabindex: 0, "aria-label": `${v.label}: ${fy(v.value)}` });
        const show = (e) => {
          const r = hitDot.getBoundingClientRect();
          showTip(e.clientX ?? r.right, e.clientY ?? r.top, tipRows(tipHead, [{ color: g.color, dot: true, value: fy(v.value), label: v.label }]));
        };
        hitDot.addEventListener("pointermove", show);
        hitDot.addEventListener("focus", show);
        hitDot.addEventListener("pointerleave", hideTip);
        hitDot.addEventListener("blur", hideTip);
        if (v.onClick) { hitDot.style.cursor = "pointer"; hitDot.addEventListener("click", v.onClick); }
        svg.appendChild(hitDot);
      });
    });
    return svg;
  }, height + (clustered ? 18 : 0));
}

function hash(text) {
  let x = 2166136261;
  for (let i = 0; i < text.length; i++) { x ^= text.charCodeAt(i); x = Math.imul(x, 16777619); }
  return ((x >>> 0) % 1000) / 1000;
}

// ---------------------------------------------------------------- bars

/** Grouped columns. categories: ["M1", ...]; series: [{label, color, values: [...]}] */
export function barChart({ categories, series, yLabel, fy = (v) => sig(v, 3), height = 220, yDomain }) {
  return responsive((width) => {
    const all = series.flatMap((sr) => sr.values);
    const yd = yDomain || [0, extent(all, 0.1)[1]];
    const ticks = niceTicks(yd[0], yd[1], 4), step = tickStep(ticks);
    const m = { l: leftMargin(ticks.map((t) => fy(t, step)), 54), r: 12, t: 14, b: 34 };
    const y = linear(yd[0], yd[1], height - m.b, m.t);
    const band = (width - m.l - m.r) / Math.max(categories.length, 1);
    const barW = Math.min(24, (band * 0.7) / Math.max(series.length, 1) - 2);
    const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img" });
    for (const t of ticks) {
      svg.appendChild(s("line", { class: "grid", x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }));
      const label = s("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" });
      label.textContent = fy(t, step);
      svg.appendChild(label);
    }
    svg.appendChild(s("line", { class: "axis", x1: m.l, x2: width - m.r, y1: y(yd[0]), y2: y(yd[0]) }));
    if (yLabel) {
      const t = s("text", { class: "axis-title", x: -(m.t + (height - m.t - m.b) / 2), y: 13, transform: "rotate(-90)", "text-anchor": "middle" });
      t.textContent = yLabel;
      svg.appendChild(t);
    }
    categories.forEach((cat, ci) => {
      const cx = m.l + band * (ci + 0.5);
      const label = s("text", { class: "label", x: cx, y: height - m.b + 18, "text-anchor": "middle" });
      label.textContent = cat;
      svg.appendChild(label);
      const groupW = series.length * barW + (series.length - 1) * 2;
      series.forEach((sr, si) => {
        const v = sr.values[ci];
        if (v == null) return;
        const x0 = cx - groupW / 2 + si * (barW + 2);
        const top = y(v), base = y(yd[0]);
        const r = Math.min(4, (base - top) / 2, barW / 2);
        const d = `M${x0},${base}V${top + r}Q${x0},${top} ${x0 + r},${top}H${x0 + barW - r}Q${x0 + barW},${top} ${x0 + barW},${top + r}V${base}Z`;
        svg.appendChild(s("path", { d, fill: sr.color }));
        const valueText = s("text", { class: "label", x: x0 + barW / 2, y: top - 5, "text-anchor": "middle" });
        valueText.textContent = fy(v);
        svg.appendChild(valueText);
        const hitRect = s("rect", { class: "hit", x: x0 - 4, y: m.t, width: barW + 8, height: base - m.t });
        hitRect.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY,
          tipRows(cat, [{ color: sr.color, value: fy(v), label: sr.label }])));
        hitRect.addEventListener("pointerleave", hideTip);
        svg.appendChild(hitRect);
      });
    });
    return svg;
  }, height);
}

// ---------------------------------------------------------------- histogram

/** Log-scaled intensity histogram with labelled markers (e.g. the enamel threshold). */
export function histogramChart({ edges, counts, markers = [], xLabel = "Intensity", height = 200, color = "var(--s1)" }) {
  const pts = counts.map((c, i) => [(edges[i] + edges[i + 1]) / 2, c > 0 ? Math.log10(c) : null]);
  const ymax = Math.max(...pts.map((p) => p[1] ?? 0), 1);
  return lineChart({
    series: [{ id: "h", label: "voxels (log10)", color, points: pts, width: 2, band: pts.map((p) => [p[0], 0, p[1] ?? 0]) }],
    rules: [], spans: markers.map((mk) => ({ x0: mk.x, label: mk.label })),
    xLabel, yLabel: "Voxels (log₁₀)", yDomain: [0, ymax * 1.05], height,
    fy: (v) => sig(v, 2), tipHead: (xv) => `${xLabel} ${sig(xv, 4)}`,
  });
}

// ---------------------------------------------------------------- legend, table, card

export function legend(items) {
  return h("div", { class: "legend" }, items.map((it) =>
    h("span", { class: "li" }, it.kind === "dash"
      ? h("i", { class: "dash", style: { borderTopColor: it.color } })
      : h("i", { class: it.kind || "line", style: { background: it.color } }), it.label)));
}

export function dataTable(columns, rows) {
  return h("div", { class: "table-wrap" }, h("table", { class: "data" },
    h("thead", null, h("tr", null, columns.map((c) => h("th", { class: c.numeric ? "n" : "" }, c.label)))),
    h("tbody", null, rows.map((r) => h("tr", null, columns.map((c) => {
      const v = r[c.key];
      return h("td", { class: c.numeric ? "n" : "" }, c.format ? c.format(v, r) : (v ?? "–"));
    }))))));
}

/**
 * A card's title, which `rename` ({fallback, set}), when given, lets the
 * reader change: a pencil beside it, or a double-click, turns it into a text
 * box. Enter or leaving it keeps the text, Escape drops it, and an empty box
 * goes back to `fallback`. `set` gets the new title, or null for the default.
 */
function cardTitle(title, rename) {
  const heading = h("h3", null, title);
  if (!rename) return heading;
  const wrap = h("span", { class: "cc-title" });
  const show = (text) => {
    const custom = text !== rename.fallback;
    replace(wrap, h("h3", { ondblclick: edit, title: custom ? `Renamed from “${rename.fallback}”. Double-click to rename.` : "Double-click to rename" }, text),
      h("button", { type: "button", class: "btn ghost icon small cc-rename", "aria-label": `Rename “${text}”`, title: "Rename this chart", onclick: edit }, icon("edit")));
  };
  function edit() {
    const shown = wrap.querySelector("h3").textContent;
    const input = h("input", { class: "cc-title-input", value: shown, placeholder: rename.fallback, maxlength: 200, "aria-label": "Chart title" });
    let done = false;
    const finish = (keep) => {
      if (done) return;
      done = true;
      const text = input.value.trim();
      const next = keep ? text || rename.fallback : shown;
      show(next);
      if (keep && next !== shown) rename.set(next === rename.fallback ? null : next);
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
    });
    input.addEventListener("blur", () => finish(true));
    replace(wrap, input);
    input.focus();
    input.select();
  }
  show(title);
  return wrap;
}

/**
 * A titled chart with a chart/table toggle. `aside` sits beside the chart on a
 * wide screen (a card's statistics, say) and under it on a narrow one; `extra`
 * always goes underneath. `onExport`, when given, adds an Export button, and
 * `rename` makes the title editable (see cardTitle).
 */
export function chartCard({ title, sub, chart, table, legendItems, extra, controls, aside, onExport, rename }) {
  const body = h("div", { class: "cc-body" });
  const split = aside ? h("div", { class: "cc-split" }, body, h("div", { class: "cc-aside" }, aside)) : null;
  let showingTable = false;
  const toggle = table ? h("button", { class: "btn small ghost", onclick: () => {
    showingTable = !showingTable;
    toggle.textContent = showingTable ? "Show chart" : "Show table";
    replace(body, showingTable ? table() : chart);
    // A table wants the card's width: the statistics step aside while it shows.
    split?.classList.toggle("showing-table", showingTable);
  } }, "Show table") : null;
  replace(body, chart);
  const exportButton = onExport ? h("button", { class: "btn small ghost cc-export", title: "Save as a Prism-style figure, or its data for Prism",
    onclick: onExport }, icon("download"), "Export") : null;
  return h("div", { class: "chart-card" },
    h("div", { class: "cc-head" }, cardTitle(title, rename), sub ? h("span", { class: "sub" }, sub) : null, exportButton, toggle),
    controls ? h("div", { class: "cc-controls" }, controls) : null,
    legendItems && legendItems.length > 1 ? legend(legendItems) : null,
    split || body,
    extra || null);
}

export { cssColor };
