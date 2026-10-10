// One heatmap scale over several specimens: the compare page's arithmetic,
// kept free of the DOM so the tests can run it under Node.
//
// A specimen's preview holds its scan's stored values. On the mg HA/cm³
// scale each specimen's values are converted with its own HA calibration
// (tools/scanco_calibration.py: density = (value + offset) / divisor * slope
// + intercept), and a window set on that scale is converted back into each
// specimen's stored values before its renderer sees it. The conversion is
// linear, so where a density falls in the window -- and so its colour -- is
// the same in every specimen, whatever setting each scan was made with.

import { toMgHA } from "./lib.js";
import { histogramPercentile, tissueHistogram } from "./volume.js";
import { OUTSIDE, stepPalette } from "./colormaps.js";

/** A stored value on the shared scale: in mg HA/cm³ with a calibration, as stored without one. */
export const toScale = (value, cal) => (cal ? toMgHA(value, cal) : value);

/** A value on the shared scale, back in one specimen's stored values. */
export const fromScale = (value, cal) =>
  (cal ? (value - cal.intercept) / cal.slope * cal.divisor - (cal.offset || 0) : value);

/**
 * The window that shows every specimen: from the lowest of their `low`
 * percentiles to the highest of their `high` ones, each read from its own
 * histogram (without the air, for a scan that is not masked) and put on the
 * shared scale.
 *
 *   specs  [{hist: {edges, counts}, masked, cal}]
 *   cal    each specimen's calibration on the mg HA/cm³ scale, null on the raw one
 */
export function sharedWindow(specs, low = 0.01, high = 0.998) {
  let lo = Infinity, hi = -Infinity;
  for (const { hist, masked, cal } of specs) {
    const tissue = tissueHistogram(hist, masked);
    lo = Math.min(lo, toScale(histogramPercentile(tissue, low), cal));
    hi = Math.max(hi, toScale(histogramPercentile(tissue, high), cal));
  }
  return Number.isFinite(lo) && hi > lo ? { lo, hi } : null;
}

/**
 * Where a value on the shared scale sits on the colour ramp, 0..1, as the 3-D
 * renderer's shader puts it: its place in the window -- or, with steps
 * coloured, its step's middle -- bent by the gamma. The legend and the
 * histogram's strip draw from it, so they match the panels exactly.
 *
 *   steps  {origin, width, fill} on the shared scale, or null
 */
export function rampPosition(value, { lo, hi, gamma = 1, steps = null }) {
  let v = value;
  if (steps && steps.width > 0 && steps.fill) v = steps.origin + (Math.floor((v - steps.origin) / steps.width) + 0.5) * steps.width;
  const t = Math.min(1, Math.max(0, (v - lo) / ((hi - lo) || 1)));
  return Math.pow(t, 1 / (gamma || 1));
}

/**
 * Steps of `width` for the random scale (colormaps.js), counted from the
 * first that reaches into [lo, hi]: it takes the palette's first colour, and
 * every step in the window one of its own, up to MOST_COLOURS of them.
 */
export function randomSteps(lo, hi, width, seed = 0) {
  const first = Math.floor(lo / width + 1e-9), last = Math.ceil(hi / width - 1e-9) - 1;
  return { origin: first * width, width, fill: true, lines: false, palette: stepPalette(last - first + 1, seed) };
}

/**
 * A value's colour on the random scale, as the shader gives it: step k from
 * the steps' origin takes palette[k mod its length]; outside the window, a grey.
 */
export function stepColour(value, { lo, hi, steps }) {
  if (value < lo) return OUTSIDE.below;
  if (value > hi) return OUTSIDE.above;
  const n = steps.palette.length, k = Math.floor((value - steps.origin) / steps.width);
  return steps.palette[((k % n) + n) % n];
}

/** A round step width (1, 2, 2.5 or 5 x 10^k) that cuts [lo, hi] into about `count` steps. */
export function stepWidth(lo, hi, count = 6) {
  const raw = Math.abs(hi - lo) / count;
  if (!(raw > 0)) return 1;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const f = raw / mag;
  return (f < 1.5 ? 1 : f < 2.25 ? 2 : f < 3.5 ? 2.5 : f < 7.5 ? 5 : 10) * mag;
}

/** The round step widths from about `most` steps across [lo, hi] down to about `fewest`. */
export function stepChoices(lo, hi, most = 200, fewest = 3) {
  const out = [];
  for (const width of [1, 2, 2.5, 5].flatMap((f) => [-2, -1, 0, 1, 2, 3, 4, 5].map((k) => f * 10 ** k)).sort((a, b) => a - b)) {
    const n = Math.abs(hi - lo) / width;
    if (n <= most && n >= fewest) out.push(width);
  }
  return out;
}

/**
 * Steps set on the shared scale, in one specimen's stored values: the same
 * edges -- 1,950 and 2,000 mg HA/cm³, say -- wherever its calibration puts them.
 */
export function stepsFor(steps, cal) {
  if (!steps || !(steps.width > 0)) return null;
  const origin = fromScale(steps.origin, cal);
  return { ...steps, origin, width: fromScale(steps.origin + steps.width, cal) - origin };
}

/**
 * How many columns lay out `n` equal panels on a width x height stage: the
 * layout that leaves each panel the most room, landscape panels winning ties,
 * since a jaw or a molar row lies longer than it stands tall.
 */
export function gridColumns(n, width, height) {
  let best = 1, room = -Infinity;
  for (let cols = 1; cols <= Math.max(1, n); cols++) {
    const rows = Math.ceil(n / cols);
    const r = Math.min(width / cols, (height / rows) * 1.3);
    if (r > room + 0.5) { room = r; best = cols; }
  }
  return best;
}
