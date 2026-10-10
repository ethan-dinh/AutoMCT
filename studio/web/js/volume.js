// A preview volume in the browser: slices, projections and overlays, all
// computed locally from the quantized array so scrolling and re-windowing
// never wait on the server.
//
// Planes: 0 = axial (a Y x X image at one Z), 1 = coronal (Z x X at one Y),
// 2 = sagittal (Z x Y at one X). Axis 0 is the specimen's long axis after the
// pipeline's reorientation.

import { api, enc, fetchBinary } from "./api.js";

export class Volume {
  constructor(meta, data, overlays = {}, unwrap = null) {
    this.meta = meta;
    this.data = data;
    this.overlays = overlays;
    this.unwrap = unwrap;
    [this.Z, this.Y, this.X] = meta.shape;
    this.q = meta.quant;
    this.step = (this.q.hi - this.q.lo) / this.q.scale;
    this.cache = new Map();
  }

  value(q) {
    if (this.meta.masked && q === 0) return null;
    return this.q.lo + (q - this.q.offset) * this.step;
  }

  dims(plane) {
    // [width, height] of the image for a plane.
    if (plane === 0) return [this.X, this.Y];
    if (plane === 1) return [this.X, this.Z];
    return [this.Y, this.Z];
  }

  depth(plane) {
    return [this.Z, this.Y, this.X][plane];
  }

  // Physical size of one image pixel (mm) along [x, y] for a plane.
  pixelMm(plane) {
    const [sz, sy, sx] = this.meta.spacing_preview_mm;
    if (plane === 0) return [sx, sy];
    if (plane === 1) return [sx, sz];
    return [sy, sz];
  }

  slice(plane, i, src = this.data, Type = Uint16Array) {
    const { Z, Y, X } = this;
    if (plane === 0) return src.subarray(i * Y * X, (i + 1) * Y * X);
    const [w, hgt] = this.dims(plane);
    const out = new Type(w * hgt);
    if (plane === 1) {
      for (let z = 0; z < Z; z++) {
        const base = z * Y * X + i * X;
        out.set(src.subarray(base, base + X), z * X);
      }
    } else {
      for (let z = 0; z < Z; z++) {
        const base = z * Y * X + i;
        const row = z * Y;
        for (let y = 0; y < Y; y++) out[row + y] = src[base + y * X];
      }
    }
    return out;
  }

  // Maximum or mean over [a, b) along a plane's axis. Means skip background
  // when the volume is masked, so a thin tooth is not averaged into the air.
  project(plane, mode, a = 0, b = this.depth(plane), src = this.data, key = "base") {
    const cacheKey = `${key}:${plane}:${mode}:${a}:${b}`;
    if (this.cache.has(cacheKey)) return this.cache.get(cacheKey);
    const { Z, Y, X } = this;
    const YX = Y * X;
    const [w, hgt] = this.dims(plane);
    const isMax = mode === "max";
    const acc = isMax ? new (src.constructor)(w * hgt) : new Float64Array(w * hgt);
    const cnt = isMax ? null : new Uint32Array(w * hgt);
    const skipZero = this.meta.masked || src !== this.data;
    // One loop nest per plane, so the inner loop is a straight run of memory.
    const add = (o, v) => {
      if (isMax) { if (v > acc[o]) acc[o] = v; }
      else if (!skipZero || v) { acc[o] += v; cnt[o]++; }
    };
    if (plane === 0) {
      for (let z = a; z < b; z++) {
        const base = z * YX;
        if (isMax) {
          for (let k = 0; k < YX; k++) { const v = src[base + k]; if (v > acc[k]) acc[k] = v; }
        } else {
          for (let k = 0; k < YX; k++) add(k, src[base + k]);
        }
      }
    } else if (plane === 1) {
      for (let z = 0; z < Z; z++) {
        const out0 = z * X;
        for (let y = a; y < b; y++) {
          const base = z * YX + y * X;
          for (let x = 0; x < X; x++) add(out0 + x, src[base + x]);
        }
      }
    } else {
      for (let z = 0; z < Z; z++) {
        for (let y = 0; y < Y; y++) {
          const base = z * YX + y * X, o = z * Y + y;
          for (let x = a; x < b; x++) add(o, src[base + x]);
        }
      }
    }
    let out = acc;
    if (!isMax) {
      out = new (src.constructor)(w * hgt);
      for (let k = 0; k < out.length; k++) out[k] = cnt[k] ? Math.round(acc[k] / cnt[k]) : 0;
    }
    if (this.cache.size > 24) this.cache.clear();
    this.cache.set(cacheKey, out);
    return out;
  }

  image(plane, index, proj, slab) {
    if (proj === "slice") return this.slice(plane, index);
    if (proj === "slab") {
      const half = Math.max(1, Math.floor(slab / 2));
      return this.project(plane, "max", Math.max(0, index - half), Math.min(this.depth(plane), index + half + 1));
    }
    return this.project(plane, proj);
  }

  overlayImage(name, plane, index, proj, slab) {
    const src = this.overlays[name];
    if (!src) return null;
    if (proj === "slice") return this.slice(plane, index, src, Uint8Array);
    if (proj === "slab") {
      const half = Math.max(1, Math.floor(slab / 2));
      return this.project(plane, "max", Math.max(0, index - half), Math.min(this.depth(plane), index + half + 1), src, name);
    }
    return this.project(plane, "max", 0, this.depth(plane), src, name);
  }

  // Preview voxel -> full-resolution voxel index in the source file's frame.
  toFull(z, y, x) {
    const f = this.meta.factors, c = this.meta.crop;
    return [
      Math.min(c[0][1] - 1, Math.round(c[0][0] + (z + 0.5) * f[0] - 0.5)),
      Math.min(c[1][1] - 1, Math.round(c[1][0] + (y + 0.5) * f[1] - 0.5)),
      Math.min(c[2][1] - 1, Math.round(c[2][0] + (x + 0.5) * f[2] - 0.5)),
    ];
  }

  // Full-resolution z (axis 0) -> preview index.
  fromFullZ(zFull) {
    return Math.max(0, Math.min(this.Z - 1, Math.floor((zFull - this.meta.crop[0][0]) / this.meta.factors[0])));
  }

  // The full-resolution z range one preview slice covers, as its midpoint.
  fullZ(z) {
    return this.meta.crop[0][0] + (z + 0.5) * this.meta.factors[0] - 0.5;
  }
}

// ---------------------------------------------------------------- loading

const LRU = [];

export async function loadVolume(sample, view) {
  const hit = LRU.find((e) => e.sample === sample && e.view === view);
  const base = `/api/samples/${enc(sample)}/views/${view}`;
  const meta = await api.get(`${base}/meta`);
  if (hit && hit.meta.base.mtime === meta.base.mtime && hit.meta.shape.join() === meta.shape.join() &&
      hit.meta.overlays.length === meta.overlays.length) {
    // The same preview, with what the server says of it now: a calibration can arrive after it was built.
    hit.volume.meta.current = meta.current;
    hit.volume.meta.calibration = meta.calibration;
    return hit.volume;
  }
  const [data, ...overlayArrays] = await Promise.all([
    fetchBinary(`${base}/file/base.u16.gz`, Uint16Array),
    ...meta.overlays.map((o) => fetchBinary(`${base}/file/${o.file}`, Uint8Array)),
  ]);
  const overlays = Object.fromEntries(meta.overlays.map((o, i) => [o.name, overlayArrays[i]]));
  const unwrap = meta.unwrap ? await fetchBinary(`${base}/file/${meta.unwrap.file}`, Uint16Array) : null;
  const volume = new Volume(meta, data, overlays, unwrap);
  LRU.unshift({ sample, view, meta, volume });
  LRU.length = Math.min(LRU.length, 3);
  return volume;
}

// ---------------------------------------------------------------- histogram helpers

// The histogram of the tissue alone. For an unmasked scan the air peak
// dominates; everything up to it is dropped, as view_napari.py's auto-contrast does.
export function tissueHistogram(hist, masked) {
  if (masked) return hist;
  let peak = 0;
  hist.counts.forEach((c, i) => { if (c > hist.counts[peak]) peak = i; });
  if (peak > hist.counts.length * 0.6) return hist;
  return { edges: hist.edges, counts: hist.counts.map((c, i) => (i <= peak ? 0 : c)) };
}

export function histogramPercentile(hist, p) {
  const counts = hist.counts, edges = hist.edges;
  const total = counts.reduce((a, b) => a + b, 0);
  if (!total) return edges[0];
  let acc = 0;
  const target = total * p;
  for (let i = 0; i < counts.length; i++) {
    acc += counts[i];
    if (acc >= target) return edges[i] + (edges[i + 1] - edges[i]) * 0.5;
  }
  return edges[edges.length - 1];
}

// Linear interpolation of y at x over sorted xs (null outside the data).
export function interp(xs, ys, x) {
  if (!xs.length) return null;
  let lo = 0, hi = xs.length - 1;
  if (x < xs[lo] || x > xs[hi]) return x < xs[lo] ? ys[lo] : ys[hi];
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const span = xs[hi] - xs[lo];
  if (!span) return ys[lo];
  const f = (x - xs[lo]) / span;
  return ys[lo] + f * (ys[hi] - ys[lo]);
}
