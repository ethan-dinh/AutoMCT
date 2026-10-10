// Heatmap colour scales. Every sequential map here rises monotonically in
// lightness, so brighter always means denser. "Ratio" is the diverging map for
// enamel/dentin ratios: neutral grey at 1.0 (same density as dentin), blue
// below it (hypomineralized), red above (mineralized enamel). "Rainbow" is for
// cutting a scale into steps, where each step's own hue tells it from its
// neighbours: blue (low) to red (high), Turbo's order, without its near-black
// ends so every step shows on the dark stage. Its lightness does not follow
// density, so read it by its colour bar. "Random" (below the maps) is no scale
// at all: each step a colour unrelated to its neighbours'.

export const COLORMAPS = {
  mineral: {
    label: "Mineral",
    stops: ["#050b14", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#f5faff"],
  },
  gray: { label: "Greyscale", stops: ["#000000", "#ffffff"] },
  magma: {
    label: "Heat",
    stops: ["#000004", "#140e36", "#3b0f70", "#641a80", "#8c2981", "#b73779", "#de4968", "#f7705c", "#fe9f6d", "#fecf92", "#fcfdbf"],
  },
  viridis: {
    label: "Viridis",
    stops: ["#440154", "#482475", "#414487", "#355f8d", "#2a788e", "#21918c", "#22a884", "#44bf70", "#7ad151", "#bddf26", "#fde725"],
  },
  cividis: {
    label: "Cividis",
    stops: ["#00204d", "#123570", "#414d6b", "#5c5d6e", "#7c7b78", "#9a9277", "#bcaf6f", "#dccb5f", "#fee825"],
  },
  ratio: {
    label: "Ratio",
    diverging: true,
    stops: ["#b7d3f6", "#5598e7", "#1c5cab", "#20303a", "#383835", "#5a2323", "#b33a3a", "#e34948", "#f6b3ae"],
  },
  rainbow: {
    label: "Rainbow",
    stops: ["#4143a7", "#476ee6", "#4196ff", "#27bee9", "#18ddc2", "#32f298", "#6dfe62", "#a4fc3c",
      "#cbed34", "#ecd13a", "#fdae35", "#fc8423", "#ed5510", "#d43305", "#af1801"],
  },
};

export const SEQUENTIAL = ["mineral", "gray", "magma", "viridis", "cividis"];

function hexToRgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

const tables = new Map();

// 256 RGB entries, interpolated between the stops.
export function table(name) {
  if (tables.has(name)) return tables.get(name);
  const stops = (COLORMAPS[name] || COLORMAPS.gray).stops.map(hexToRgb);
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const k = Math.min(Math.floor(t), stops.length - 2);
    const f = t - k;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = Math.round(stops[k][c] * (1 - f) + stops[k + 1][c] * f);
  }
  tables.set(name, out);
  return out;
}

// Packed little-endian RGBA, the layout an ImageData's Uint32Array view expects.
export function pack(r, g, b, a = 255) {
  return ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
}

// quantized value (0..65535) -> packed colour, for a window [lo, hi] in true units.
export function buildLut(name, quant, lo, hi, gamma = 1, masked = false) {
  const tab = table(name);
  const lut = new Uint32Array(65536);
  const step = (quant.hi - quant.lo) / quant.scale;
  const span = hi - lo || 1;
  const invGamma = 1 / (gamma || 1);
  for (let q = 0; q < 65536; q++) {
    if (masked && q === 0) { lut[q] = 0; continue; }
    const v = quant.lo + (q - quant.offset) * step;
    let t = (v - lo) / span;
    t = t <= 0 ? 0 : t >= 1 ? 1 : Math.pow(t, invGamma);
    const i = Math.round(t * 255) * 3;
    lut[q] = pack(tab[i], tab[i + 1], tab[i + 2]);
  }
  return lut;
}

export function colorAt(name, t) {
  const tab = table(name);
  const i = Math.round(Math.max(0, Math.min(1, t)) * 255) * 3;
  return [tab[i], tab[i + 1], tab[i + 2]];
}

export function gradientCanvas(name, width = 120, height = 10) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const tab = table(name);
  for (let x = 0; x < width; x++) {
    const i = Math.round((x / (width - 1)) * 255) * 3;
    ctx.fillStyle = `rgb(${tab[i]},${tab[i + 1]},${tab[i + 2]})`;
    ctx.fillRect(x, 0, 1, height);
  }
  return canvas;
}

// ------------------------------------------------------------ random steps
// No scale at all: each step takes a colour unrelated to its neighbours', so
// the pattern the steps make -- where each density lies -- shows, where a
// gradient would blend it. They are the most distinct colours the screen has,
// picked greedily (Glasbey's way): each as far, in OKLab, from every one
// picked before it as any can be, lightness counting half, since the light
// shades a colour's lightness and not its hue. None is dark, pale or grey:
// those are the stage's, the light's and OUTSIDE's.
export const RANDOM = "random";
// More steps than this take the colours round again.
export const MOST_COLOURS = 32;
// On the random scale, what lies below the window, and above it.
export const OUTSIDE = { below: [96, 104, 106], above: [222, 227, 226] };

const linear = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
function oklab(rgb) {
  const [r, g, b] = rgb.map(linear);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
}
// How far apart two OKLab colours look, lightness counting half.
const gap = (p, q) => Math.hypot(0.5 * (p[0] - q[0]), p[1] - q[1], p[2] - q[2]);
/** How unlike two [r, g, b] colours look: the distance the palette is picked by. */
export const unlike = (p, q) => gap(oklab(p), oklab(q));

let distinct = null;
// The MOST_COLOURS colours, most distinct first: [{rgb, lab}].
function distinctColours() {
  if (distinct) return distinct;
  const pool = [];
  for (let r = 0; r < 256; r += 17) for (let g = 0; g < 256; g += 17) for (let b = 0; b < 256; b += 17) {
    const lab = oklab([r, g, b]);
    if (lab[0] >= 0.52 && lab[0] <= 0.92 && Math.hypot(lab[1], lab[2]) >= 0.09) pool.push({ rgb: [r, g, b], lab });
  }
  // Far from black, white, the stage and the two greys to begin with.
  const taken = [[0, 0, 0], [255, 255, 255], [14, 18, 19], OUTSIDE.below, OUTSIDE.above].map(oklab);
  const nearest = pool.map((c) => Math.min(...taken.map((t) => gap(c.lab, t))));
  distinct = [];
  while (distinct.length < MOST_COLOURS) {
    let i = 0;
    for (let j = 1; j < pool.length; j++) if (nearest[j] > nearest[i]) i = j;
    distinct.push(pool[i]);
    for (let j = 0; j < pool.length; j++) nearest[j] = Math.min(nearest[j], gap(pool[j].lab, pool[i].lab));
  }
  return distinct;
}

// A repeatable random number stream (mulberry32): one seed, one order.
function stream(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const palettes = new Map();

/**
 * Colours for `n` steps, [r, g, b] each: the n most distinct (at most
 * MOST_COLOURS). Seed 0 lays them out in the order they were picked; any
 * other deals them in a random order of its own, each still among the most
 * unlike the two before it, so neighbouring steps never look alike.
 */
export function stepPalette(n, seed = 0) {
  n = Math.max(1, Math.min(MOST_COLOURS, Math.floor(n) || 1));
  const key = `${n}:${seed}`;
  if (palettes.has(key)) return palettes.get(key);
  const set = distinctColours().slice(0, n);
  let order = set;
  if (seed) {
    const rand = stream(seed), left = [...set];
    order = left.splice(Math.floor(rand() * left.length), 1);
    while (left.length) {
      const score = (c) => Math.min(...order.slice(-2).map((o) => gap(o.lab, c.lab)));
      left.sort((a, b) => score(b) - score(a));
      order.push(left.splice(Math.floor(rand() * Math.min(left.length, Math.max(2, Math.ceil(n / 4)))), 1)[0]);
    }
  }
  const out = order.map((c) => c.rgb);
  palettes.set(key, out);
  return out;
}

/** A swatch of colours side by side, for the random scale's button. */
export function stepsCanvas(colours, width = 120, height = 10) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  colours.forEach(([r, g, b], i) => {
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    const x = Math.round((i * width) / colours.length);
    ctx.fillRect(x, 0, Math.round(((i + 1) * width) / colours.length) - x, height);
  });
  return canvas;
}
