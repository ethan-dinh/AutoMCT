// Drawing a cleaned incisor profile: the R report's running median, and where
// a group's curve may start. The server does the rest of the report's cleanup
// (trajectory.py) and marks each slice kept or not; the pages smooth the kept
// slices here, at whatever width the reader picks. No DOM here: the tests run
// this module under Node against the server's Python.

// The statuses that carry a measured ratio (trajectory.MEASURABLE).
export const MEASURABLE = ["resolved_hi", "resolved_lo", "coincident"];

/** The typical spacing between consecutive slices: 1, or N when every Nth slice was measured. */
export function sliceStep(zs) {
  const sorted = [...zs].sort((a, b) => a - b);
  const gaps = [];
  for (let i = 1; i < sorted.length; i++) { const g = sorted[i] - sorted[i - 1]; if (g > 0) gaps.push(g); }
  if (!gaps.length) return 1;
  gaps.sort((a, b) => a - b);
  const mid = gaps.length >> 1;
  return gaps.length % 2 ? gaps[mid] : (gaps[mid - 1] + gaps[mid]) / 2;
}

/** The running median's width in points: `slices` slices, fewer points at a wider step. Always odd. */
export function medianWindow(step, slices = 21) {
  const k = Math.max(1, Math.round(slices / step));
  return k % 2 ? k : k + 1;
}

function median(values) {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const mid = v.length >> 1;
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

/**
 * The report's roll_med_gapped(): a centred k-point running median over
 * `values`, in slice order, with `zs` their slice indices. Null at either end,
 * where the window would be partial, and wherever the window's slices span
 * more than 1.5 steps per point: a window straddling missing slices is
 * dropped rather than bridged.
 */
export function gappedMedian(values, zs, k, step = 1) {
  const n = values.length, out = new Array(n).fill(null);
  if (n < k || k < 1) return out;
  const half = k >> 1;
  for (let i = half; i < n - (k - 1 - half); i++) {
    const lo = i - half, hi = lo + k;
    let zmin = Infinity, zmax = -Infinity;
    for (let j = lo; j < hi; j++) { zmin = Math.min(zmin, zs[j]); zmax = Math.max(zmax, zs[j]); }
    if (zmax - zmin > (k - 1) * 1.5 * step) continue;
    out[i] = median(values.slice(lo, hi));
  }
  return out;
}

/**
 * One specimen's curve: `points` are [% along, value, slice index] for the
 * slices it uses. The running median runs along the scan over `slices`
 * slices (1: none); the curve comes back as [% along, value or null] in
 * order along the tooth, a null wherever the median is empty.
 */
export function smoothProfile(points, slices = 21) {
  const bySlice = points.filter((p) => p[2] != null && Number.isFinite(p[1])).sort((a, b) => a[2] - b[2]);
  if (!bySlice.length) return [];
  const zs = bySlice.map((p) => p[2]);
  const step = sliceStep(zs);
  const k = slices > 1 ? medianWindow(step, slices) : 1;
  const values = bySlice.map((p) => p[1]);
  const med = k > 1 ? gappedMedian(values, zs, k, step) : values;
  return bySlice.map((p, i) => [p[0], med[i]]).sort((a, b) => a[0] - b[0]);
}

/**
 * Where a group's curve starts: the lowest tenth of the tooth in which its
 * specimens measure, on average, half their slices (the report's section 5).
 * `coverages` holds each specimen's % measured per tenth, null where it has
 * no slice. Null when no tenth reaches half: the curve is drawn throughout.
 */
export function coverageFloor(coverages, floor = 50) {
  for (let b = 0; b < 10; b++) {
    const vals = coverages.map((c) => c?.[b]).filter((v) => v != null);
    if (vals.length && vals.reduce((s, v) => s + v, 0) / vals.length >= floor) return b * 10;
  }
  return null;
}

/** A curve's value at x, interpolated between its two neighbours; null outside it or where either is empty. */
export function valueAt(pts, x) {
  if (!pts.length || x < pts[0][0] || x > pts[pts.length - 1][0]) return null;
  let lo = 0, hi = pts.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pts[mid][0] <= x) lo = mid; else hi = mid; }
  const a = pts[lo], b = pts[hi];
  if (a[0] === x) return a[1];
  if (b[0] === x) return b[1];
  if (a[1] == null || b[1] == null) return null;
  const f = b[0] === a[0] ? 0 : (x - a[0]) / (b[0] - a[0]);
  return a[1] + f * (b[1] - a[1]);
}
