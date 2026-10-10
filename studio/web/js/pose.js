// Where a specimen sits in its preview, and how to line one up with another.
//
// The 3-D views turn a specimen about its tissue's centre, not its box's: a
// scan cropped with more air on one side would otherwise swing round a point
// beside it, and specimens side by side would sit off-centre in their panels
// by however differently they were cropped.
//
// Lining up: the segmentation's reorientation only permutes and flips axes, so
// each scan keeps whatever tilt it had in the holder. One specimen is fitted
// to another by the turn and shift that bring their surfaces closest -- the
// mean distance from each one's surface to the other's, each distance capped
// so a part one has and the other lacks does not pull the fit -- tried from
// the specimen as scanned and from every way of laying its principal axes on
// the other's, then refined in smaller and smaller steps. Rigid: nothing is
// scaled, so sizes stay comparable.
//
// Positions are in the renderer's world frame, in mm: x along array axis 0,
// y against axis 1, z along axis 2, the origin at the preview box's centre.
// Matrices are 3 x 3, row-major.

export const identity = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];

export const matMul = (a, b) => Array.from({ length: 9 }, (_, k) => {
  const i = Math.floor(k / 3), j = k % 3;
  return a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
});

export const transpose = (m) => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];

export const apply = (m, v) => [m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
  m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];

const det = (m) => m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);

// How far a rotation turns, in radians.
export const angleOf = (m) => Math.acos(Math.min(1, Math.max(-1, (m[0] + m[4] + m[8] - 1) / 2)));

// A right-handed turn by `angle` about the unit axis `a` (Rodrigues).
export function axisAngle([x, y, z], angle) {
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [c + x * x * t, x * y * t - z * s, x * z * t + y * s,
    y * x * t + z * s, c + y * y * t, y * z * t - x * s,
    z * x * t - y * s, z * y * t + x * s, c + z * z * t];
}

// Gram-Schmidt on the columns: keeps a matrix a rotation as small turns pile up.
export function orthonormal(m) {
  const unit = (v) => { const l = Math.hypot(...v) || 1; return v.map((c) => c / l); };
  const a = unit([m[0], m[3], m[6]]);
  const raw = [m[1], m[4], m[7]];
  const d = a[0] * raw[0] + a[1] * raw[1] + a[2] * raw[2];
  const b = unit(raw.map((c, i) => c - d * a[i]));
  const n = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  return [a[0], b[0], n[0], a[1], b[1], n[1], a[2], b[2], n[2]];
}

// A symmetric matrix's eigenvectors (the columns of `vectors`) and values,
// largest first, by Jacobi rotations.
export function eigenSym(m) {
  const A = [...m];
  let V = identity();
  for (let sweep = 0; sweep < 64; sweep++) {
    const off = A[1] * A[1] + A[2] * A[2] + A[5] * A[5];
    const scale = A[0] * A[0] + A[4] * A[4] + A[8] * A[8];
    if (off <= 1e-24 * scale || off === 0) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = A[p * 3 + q];
      if (apq === 0) continue;
      const theta = (A[q * 3 + q] - A[p * 3 + p]) / (2 * apq);
      const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const kp = A[k * 3 + p], kq = A[k * 3 + q];
        A[k * 3 + p] = c * kp - s * kq;
        A[k * 3 + q] = s * kp + c * kq;
      }
      for (let k = 0; k < 3; k++) {
        const pk = A[p * 3 + k], qk = A[q * 3 + k];
        A[p * 3 + k] = c * pk - s * qk;
        A[q * 3 + k] = s * pk + c * qk;
      }
      for (let k = 0; k < 3; k++) {
        const kp = V[k * 3 + p], kq = V[k * 3 + q];
        V[k * 3 + p] = c * kp - s * kq;
        V[k * 3 + q] = s * kp + c * kq;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => A[j * 4] - A[i * 4]);
  const vectors = Array.from({ length: 9 }, (_, k) => V[Math.floor(k / 3) * 3 + order[k % 3]]);
  // A right-handed set, so turning one set onto another is a rotation.
  if (det(vectors) < 0) for (let k = 0; k < 3; k++) vectors[k * 3 + 2] *= -1;
  return { vectors, values: order.map((i) => A[i * 4]) };
}

// The value between a histogram's two classes that separates them best (Otsu).
export function otsu(hist) {
  const { counts, edges } = hist;
  const mid = (i) => (edges[i] + edges[i + 1]) / 2;
  let total = 0, sum = 0;
  for (let i = 0; i < counts.length; i++) { total += counts[i]; sum += counts[i] * mid(i); }
  let wB = 0, sumB = 0, best = -1, cut = edges[0];
  for (let i = 0; i < counts.length - 1; i++) {
    wB += counts[i];
    sumB += counts[i] * mid(i);
    const wF = total - wB;
    if (!wB) continue;
    if (!wF) break;
    const between = wB * wF * (sumB / wB - (sum - sumB) / wF) ** 2;
    if (between > best) { best = between; cut = edges[i + 1]; }
  }
  return cut;
}

// Cells across the distance grid's widest reach, at most: fine enough to tell
// a molar from its neighbour, coarse enough to be quick.
const GRID_CELLS = 40;

// A specimen's tissue, from a loaded preview (a Volume, or {meta, data}):
//   centre   where its tissue's mass sits, in world mm;
//   points   up to maxPoints of its surface voxels, relative to the centre (x, y, z, ...);
//   radius   how far its tissue reaches from the centre;
//   axes     its principal axes, as the columns of a rotation, longest first;
//   grid     the distance to its surface, in mm, on a grid about the centre: what fitting measures by.
// Tissue is every voxel of a masked preview, or above the Otsu threshold of an
// unmasked one's histogram; its surface, the tissue voxels beside any that are
// not. Null when there is none.
export function tissueCloud(vol, { maxPoints = 3000, budget = 2_000_000 } = {}) {
  const [Z, Y, X] = vol.meta.shape;
  const [sz, sy, sx] = vol.meta.spacing_preview_mm;
  const data = vol.data, q = vol.meta.quant;
  let floor = 1;
  if (!vol.meta.masked && vol.meta.histogram) {
    const raw = otsu(vol.meta.histogram);
    floor = Math.min(65535, Math.max(1, Math.ceil(q.offset + ((raw - q.lo) / ((q.hi - q.lo) || 1)) * q.scale)));
  }
  // Every f-th voxel along each axis, from the middle of each block of f.
  const f = Math.max(1, Math.round(Math.cbrt((Z * Y * X) / budget)));
  const o = (f - 1) >> 1;
  const half = [(Z * sz) / 2, (Y * sy) / 2, (X * sx) / 2];
  const YX = Y * X;
  // fn(x, y, z in the world, whether it is on the surface) for each tissue voxel visited.
  const visit = (fn) => {
    for (let z = o; z < Z; z += f) {
      const wx = (z + 0.5) * sz - half[0];
      for (let y = o; y < Y; y += f) {
        const wy = half[1] - (y + 0.5) * sy;
        for (let x = o, i = z * YX + y * X + o; x < X; x += f, i += f) {
          if (data[i] < floor) continue;
          const edge = x < f || x + f >= X || y < f || y + f >= Y || z < f || z + f >= Z ||
            data[i - f] < floor || data[i + f] < floor || data[i - f * X] < floor || data[i + f * X] < floor ||
            data[i - f * YX] < floor || data[i + f * YX] < floor;
          fn(wx, wy, (x + 0.5) * sx - half[2], edge);
        }
      }
    }
  };
  let n = 0;
  const s = [0, 0, 0], ss = [0, 0, 0, 0, 0, 0];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const surface = [];
  visit((a, b, c, edge) => {
    n++;
    s[0] += a; s[1] += b; s[2] += c;
    ss[0] += a * a; ss[1] += a * b; ss[2] += a * c; ss[3] += b * b; ss[4] += b * c; ss[5] += c * c;
    if (a < lo[0]) lo[0] = a; if (a > hi[0]) hi[0] = a;
    if (b < lo[1]) lo[1] = b; if (b > hi[1]) hi[1] = b;
    if (c < lo[2]) lo[2] = c; if (c > hi[2]) hi[2] = c;
    if (edge) surface.push(a, b, c);
  });
  if (!n) return null;
  const centre = s.map((v) => v / n);
  const [cx, cy, cz] = centre;
  const cov = [ss[0] / n - cx * cx, ss[1] / n - cx * cy, ss[2] / n - cx * cz,
    0, ss[3] / n - cy * cy, ss[4] / n - cy * cz, 0, 0, ss[5] / n - cz * cz];
  cov[3] = cov[1]; cov[6] = cov[2]; cov[7] = cov[5];
  const { vectors: axes, values: spread } = eigenSym(cov);

  const corner = [0, 1, 2].map((i) => Math.max(hi[i] - centre[i], centre[i] - lo[i]));
  const radius = Math.hypot(...corner) + f * Math.max(sz, sy, sx);
  const step = Math.max(Math.max(...corner) / GRID_CELLS, f * Math.min(sz, sy, sx));
  // Three cells of margin, and the outermost voxels' centres on cell centres.
  const origin = lo.map((v, i) => v - centre[i] - 3.5 * step);
  const dims = [0, 1, 2].map((i) => Math.ceil((hi[i] - lo[i]) / step) + 7);
  const seeds = new Uint8Array(dims[0] * dims[1] * dims[2]);
  const m = surface.length / 3, every = Math.max(1, Math.ceil(m / maxPoints));
  const points = new Float32Array(3 * Math.ceil(m / every));
  let far = 0;
  for (let k = 0, w = 0; k < m; k++) {
    const a = surface[3 * k] - cx, b = surface[3 * k + 1] - cy, c = surface[3 * k + 2] - cz;
    far = Math.max(far, a * a + b * b + c * c);
    seeds[(Math.floor((a - origin[0]) / step) * dims[1] + Math.floor((b - origin[1]) / step)) * dims[2] + Math.floor((c - origin[2]) / step)] = 1;
    if (k % every === 0) { points[w++] = a; points[w++] = b; points[w++] = c; }
  }
  return {
    centre, points, count: n, axes, spread,
    radius: Math.min(radius, Math.sqrt(far) + f * Math.max(sz, sy, sx)),
    grid: { origin, dims, step, dist: distanceField(seeds, dims, step) },
  };
}

// The Euclidean distance from every cell to the nearest seed, in mm (the
// separable exact transform of Felzenszwalb and Huttenlocher).
export function distanceField(seeds, dims, step) {
  const [a, b, c] = dims;
  const FAR = 4 * (a * a + b * b + c * c) + 1;  // beyond any true squared distance, and exact in a double
  const d = new Float64Array(seeds.length);
  for (let i = 0; i < d.length; i++) d[i] = seeds[i] ? 0 : FAR;
  const len = Math.max(a, b, c);
  const f = new Float64Array(len), out = new Float64Array(len), v = new Int32Array(len), z = new Float64Array(len + 1);
  const line = (start, stride, n) => {
    for (let i = 0; i < n; i++) f[i] = d[start + i * stride];
    let k = 0;
    v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
    for (let q = 1; q < n; q++) {
      // z[0] is -Infinity, so k never passes below 0.
      let s = (f[q] + q * q - f[v[k]] - v[k] * v[k]) / (2 * (q - v[k]));
      while (s <= z[k]) {
        k--;
        s = (f[q] + q * q - f[v[k]] - v[k] * v[k]) / (2 * (q - v[k]));
      }
      k++;
      v[k] = q; z[k] = s; z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < n; q++) {
      while (z[k + 1] < q) k++;
      out[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
    }
    for (let i = 0; i < n; i++) d[start + i * stride] = out[i];
  };
  for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) line((i * b + j) * c, 1, c);
  for (let i = 0; i < a; i++) for (let l = 0; l < c; l++) line(i * b * c + l, c, b);
  for (let j = 0; j < b; j++) for (let l = 0; l < c; l++) line(j * c + l, b * c, a);
  const mm = new Float32Array(d.length);
  for (let i = 0; i < d.length; i++) mm[i] = Math.sqrt(d[i]) * step;
  return mm;
}

// The mean distance from each point -- turned by R and moved by t, or with
// inverse, moved back by t and turned back by R -- to the grid's surface, in
// mm, read between cell centres and capped at `cap`.
function misfit(grid, pts, R, t, inverse, cap) {
  const { origin, dims, step, dist } = grid;
  const M = inverse ? transpose(R) : R;
  const inv = 1 / step, s1 = dims[2], s0 = dims[1] * dims[2];
  let sum = 0;
  for (let k = 0; k < pts.length; k += 3) {
    let a = pts[k], b = pts[k + 1], c = pts[k + 2];
    if (inverse) { a -= t[0]; b -= t[1]; c -= t[2]; }
    let x = M[0] * a + M[1] * b + M[2] * c, y = M[3] * a + M[4] * b + M[5] * c, z = M[6] * a + M[7] * b + M[8] * c;
    if (!inverse) { x += t[0]; y += t[1]; z += t[2]; }
    const fx = (x - origin[0]) * inv - 0.5, fy = (y - origin[1]) * inv - 0.5, fz = (z - origin[2]) * inv - 0.5;
    const i = Math.floor(fx), j = Math.floor(fy), l = Math.floor(fz);
    if (i < 0 || j < 0 || l < 0 || i >= dims[0] - 1 || j >= dims[1] - 1 || l >= dims[2] - 1) { sum += cap; continue; }
    const u = fx - i, v = fy - j, w = fz - l, o = i * s0 + j * s1 + l;
    const e00 = dist[o] + (dist[o + 1] - dist[o]) * w, e01 = dist[o + s1] + (dist[o + s1 + 1] - dist[o + s1]) * w;
    const e10 = dist[o + s0] + (dist[o + s0 + 1] - dist[o + s0]) * w, e11 = dist[o + s0 + s1] + (dist[o + s0 + s1 + 1] - dist[o + s0 + s1]) * w;
    const e0 = e00 + (e01 - e00) * v, e1 = e10 + (e11 - e10) * v;
    const e = e0 + (e1 - e0) * u;
    sum += e < cap ? e : cap;
  }
  return pts.length ? (sum * 3) / pts.length : cap;
}

// Every rotation that lays one set of principal axes along another, whichever
// way each points: the 24 signed orderings that keep a right hand.
function axisTurns(to, from) {
  const out = [];
  const back = transpose(from);
  for (const perm of [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]) {
    for (let signs = 0; signs < 8; signs++) {
      const S = Array(9).fill(0);
      perm.forEach((j, i) => { S[i * 3 + j] = signs & (1 << i) ? -1 : 1; });
      if (det(S) < 0) continue;
      out.push(orthonormal(matMul(matMul(to, S), back)));
    }
  }
  return out;
}

const UNIT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
// Principal-axis turns refined beside the specimen as scanned.
const STARTS = 6;
const DEG = Math.PI / 180;

// The rigid move that lays `mov` on `ref`: a point p of mov, relative to its
// centre, sits at rot x p + shift relative to ref's. {rot, shift, misfit (the
// mean distance left between their surfaces, mm), turned (radians), moved
// (mm), from ("scan" or "axes")}, or null without two clouds.
export function register(ref, mov) {
  if (!ref || !mov || !ref.points.length || !mov.points.length) return null;
  // A distance counts up to a quarter of the larger specimen's reach: past that
  // the surfaces are not the same part, and the fit should not bend to join them.
  const cap = 0.25 * Math.max(ref.radius, mov.radius);
  const cost = (R, t) => (misfit(ref.grid, mov.points, R, t, false, cap) + misfit(mov.grid, ref.points, R, t, true, cap)) / 2;
  const cell = Math.max(ref.grid.step, mov.grid.step);
  // Smaller and smaller turns about each axis and moves along it, while any helps.
  const refine = (start) => {
    let best = start;
    for (const [deg, cells] of [[12, 2], [6, 1], [3, 0.5], [1.5, 0.25], [0.75, 0.125], [0.35, 0.06]]) {
      for (let round = 0; round < 16; round++) {
        let next = null;
        for (const axis of UNIT) {
          for (const sign of [1, -1]) {
            const R = orthonormal(matMul(axisAngle(axis, sign * deg * DEG), best.R));
            const cR = cost(R, best.t);
            if (cR < (next || best).c - 1e-12) next = { R, t: best.t, c: cR };
            const t = best.t.map((v, i) => v + sign * cells * cell * axis[i]);
            const cT = cost(best.R, t);
            if (cT < (next || best).c - 1e-12) next = { R: best.R, t, c: cT };
          }
        }
        if (!next) break;
        best = { ...best, ...next };
      }
    }
    return best;
  };
  // Where to start: as scanned, and the principal-axis turns that fit best
  // before refining -- several, since a nearly symmetric specimen (one with its
  // ramus cropped off, say) can fit a flipped start nearly as well at first.
  const starts = [identity(), ...axisTurns(ref.axes, mov.axes)]
    .map((R, i) => ({ R, t: [0, 0, 0], c: cost(R, [0, 0, 0]), from: i ? "axes" : "scan" }));
  const fits = [starts[0], ...starts.slice(1).sort((x, y) => x.c - y.c).slice(0, STARTS)].map(refine);
  // The closest fit, or of fits within 1% of it, the smallest turn: a symmetric
  // shape is not flipped for nothing. (A hemimandible cropped of its ramus
  // still fits its flip 3% worse.)
  const least = Math.min(...fits.map((x) => x.c));
  const best = fits.filter((x) => x.c <= least * 1.01 + 1e-12).reduce((x, y) => (angleOf(y.R) < angleOf(x.R) ? y : x));
  return { rot: best.R, shift: best.t, misfit: best.c, turned: angleOf(best.R), moved: Math.hypot(...best.t), from: best.from };
}
