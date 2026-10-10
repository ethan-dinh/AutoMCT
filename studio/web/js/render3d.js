// A preview volume in 3-D on the lightbox stage, turned with the mouse.
//
// Two ways to look: a surface, where each ray stops at the first voxel as
// bright as the chosen level, shaded, tinted by the structure there and
// coloured by the density just beneath it; or an X-ray, the brightest value
// along each ray. A cut-away plane, square to the view, slices off the near
// part to show the inside. It draws from the preview the slices use, held as
// 16-bit floats, so nothing new is built or downloaded.
//
// For fine differences the surface's colour can be averaged over a little
// tissue straight in from it, cut into steps -- each one colour of the scale,
// or on the random scale one of its own -- and shaded less, so the light does
// not move colours off the scale.
//
// World frame: x along array axis 0 (the specimen's long axis), y against
// axis 1 (so an image row's "up" stays up), z along axis 2. Two reflections
// -- (z, y, x) array order and the flipped y -- make it a true, unmirrored
// picture of the scan.
//
// The camera turns freely, all the way round in any direction (a trackball:
// dragging turns about the screen's own axes, Ctrl- or ⌘-drag rolls), about
// the tissue's centre rather than the box's (pose.js).
//
// Several of these can turn together (the compare page): each reports the
// user's moves through onMove, and takes another's camera with setCamera. A
// specimen can also be turned and moved on its own, to line it up with the
// others; that is the specimen's pose, not the camera's, so the views stay
// linked while one specimen sits differently in its own.

import { table, OUTSIDE, MOST_COLOURS } from "./colormaps.js";
import { identity, matMul, transpose, apply, axisAngle, orthonormal, tissueCloud } from "./pose.js";

const FOV = 30 * Math.PI / 180;
const MAX_LABELS = 8;
// Named sides as [yaw, pitch]: turned about the vertical, then tipped towards the top.
const VIEWS = { side: [0, 0.15], top: [0, Math.PI / 2], other: [Math.PI, 0.15], bottom: [0, -Math.PI / 2], end: [Math.PI / 2, 0.15] };
// Radians per pixel dragged.
const TURN = 0.008;

const VERT = `#version 300 es
layout(location = 0) in vec2 aPos;
out vec2 vNdc;
void main() { vNdc = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vNdc;
uniform sampler3D uVol;
uniform sampler3D uLab;
uniform sampler2D uCmap;
uniform vec3 uEye, uRight, uUp, uForward;
uniform float uTanHalf, uAspect;
uniform vec3 uExtent, uTexel;
uniform float uStep, uLevel, uLo, uHi, uInvGamma, uCut, uTint, uDeep;
// Below this a texel is empty, not dim: a masked volume's background (0). Unmasked, nothing is.
uniform float uFloor;
// How much the light shapes the surface (0: its colour alone), and the radius the
// density beneath it is averaged over (0: one point).
uniform float uShade, uAvg;
// Steps: edges at uStepOrigin + k * uStepWidth (none while the width is 0).
// uStepFill gives each step one colour, its middle's; uLines draws its edges.
// With a palette (the random scale), step k takes colour k mod uColours of it
// instead, and whatever lies outside the window uBelow or uAbove.
uniform float uStepOrigin, uStepWidth, uStepFill, uLines, uColours;
uniform vec3 uPalette[${MOST_COLOURS}], uBelow, uAbove;
uniform int uStyle, uSteps;
uniform vec3 uLabelColor[${MAX_LABELS}];
uniform float uLabelOn[${MAX_LABELS}];
uniform vec3 uStage;
out vec4 outColor;

vec3 toTex(vec3 w) { return vec3(w.z / uExtent.z + 0.5, 0.5 - w.y / uExtent.y, w.x / uExtent.x + 0.5); }
float at(vec3 w) { return texture(uVol, toTex(w)).r; }
vec3 cmap(float v) {
  if (uColours > 0.0 && uStepWidth > 0.0) {
    if (v < max(uLo, uFloor)) return uBelow;
    if (v > uHi) return uAbove;
    float k = floor((v - uStepOrigin) / uStepWidth);
    // k mod uColours, exactly: (k + 0.5) / uColours is never within rounding of a whole number.
    return uPalette[int(k - uColours * floor((k + 0.5) / uColours))];
  }
  if (v >= uFloor && uStepWidth > 0.0 && uStepFill > 0.5) {
    v = uStepOrigin + (floor((v - uStepOrigin) / uStepWidth) + 0.5) * uStepWidth;
  }
  float t = v < uFloor ? 0.0 : clamp((v - uLo) / max(uHi - uLo, 1e-6), 0.0, 1.0);
  t = pow(t, uInvGamma);
  return texture(uCmap, vec2((t * 255.0 + 0.5) / 256.0, 0.5)).rgb;
}

// The density beneath a surface point, straight in: at the point, or the mean
// over a small cylinder there -- radius uAvg, reaching 2 x uAvg deeper -- of
// the voxels at least as dense as the surface, so a thin wall is not averaged
// with the air or background around it. Averaging is what lets differences
// finer than one voxel's noise show.
float beneath(vec3 p, vec3 inward) {
  if (uAvg <= 0.0) return at(p);
  vec3 a = normalize(cross(inward, abs(inward.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 b = cross(inward, a);
  float floorV = max(uFloor, uLevel), sum = 0.0, n = 0.0;
  for (int i = 0; i < 3; i++) {
    vec3 q = p + inward * (uAvg * float(i));
    for (int j = 0; j < 5; j++) {
      vec3 o = j == 1 ? a : j == 2 ? -a : j == 3 ? b : j == 4 ? -b : vec3(0.0);
      float v = at(q + o * uAvg);
      if (v >= floorV) { sum += v; n += 1.0; }
    }
  }
  return n > 0.0 ? sum / n : at(p);
}

void main() {
  vec3 rd = normalize(uForward + vNdc.x * uTanHalf * uAspect * uRight + vNdc.y * uTanHalf * uUp);
  vec3 ro = uEye;
  vec3 halfSize = 0.5 * uExtent;
  vec3 inv = 1.0 / rd;
  vec3 t0 = (-halfSize - ro) * inv, t1 = (halfSize - ro) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1);
  float tEnter = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
  float tExit = min(min(tmax.x, tmax.y), tmax.z);
  float tCut = uCut / max(dot(rd, uForward), 1e-4);
  bool cut = tCut > tEnter;
  float t = max(tEnter, tCut);
  bool inBox = tExit > t;

  if (uStyle == 1) {
    float m = 0.0;
    // A loop bound set at draw time, not a constant: compilers may try to unroll a constant one.
    for (int i = 0; i < uSteps; i++) {
      if (!inBox || t > tExit) break;
      m = max(m, at(ro + rd * t));
      t += uStep;
    }
    outColor = vec4(!inBox || m <= uLo || m < uFloor ? uStage : cmap(m), 1.0);
    return;
  }

  // No early return from here on: the step edges' width below takes every pixel's neighbours.
  float prev = t;
  bool hit = false, face = false;
  for (int i = 0; i < uSteps; i++) {
    if (!inBox || t > tExit) break;
    if (at(ro + rd * t) >= uLevel) { hit = true; face = cut && i == 0; break; }
    prev = t;
    t += uStep;
  }
  vec3 col = uStage;
  float band = 0.0, lineDens = 0.0;
  if (hit) {
    if (!face) {
      float a = prev, b = t;
      for (int k = 0; k < 6; k++) { float m = 0.5 * (a + b); if (at(ro + rd * m) >= uLevel) b = m; else a = m; }
      t = b;
    }
    vec3 p = ro + rd * t;
    vec3 n;
    if (face) {
      n = -uForward;
    } else {
      vec3 c = toTex(p);
      float gs = texture(uVol, c + vec3(uTexel.x, 0.0, 0.0)).r - texture(uVol, c - vec3(uTexel.x, 0.0, 0.0)).r;
      float gt = texture(uVol, c + vec3(0.0, uTexel.y, 0.0)).r - texture(uVol, c - vec3(0.0, uTexel.y, 0.0)).r;
      float gr = texture(uVol, c + vec3(0.0, 0.0, uTexel.z)).r - texture(uVol, c - vec3(0.0, 0.0, uTexel.z)).r;
      vec3 g = vec3(gr / (uTexel.z * uExtent.x), -gt / (uTexel.y * uExtent.y), gs / (uTexel.x * uExtent.z));
      n = -g / max(length(g), 1e-9);
      if (dot(n, rd) > 0.0) n = -n;
    }
    // The density a little beneath the surface, straight in from it, so it is
    // read at one depth whatever angle the surface is seen from: enamel reads
    // brighter than dentin.
    vec3 inward = face ? uForward : -n;
    vec3 inside = face ? p : p + inward * uDeep;
    float dens = beneath(inside, inward);
    vec3 base = cmap(dens);
    if (uStepWidth > 0.0) band = (dens - uStepOrigin) / uStepWidth;
    lineDens = dens;
    // The structure beneath: read as deep as the density, then deeper, so the
    // tint does not flicker along the voxel steps of the structure's edge.
    int label = int(texture(uLab, toTex(inside)).r * 255.0 + 0.5);
    if (label == 0 && !face) label = int(texture(uLab, toTex(inside + inward * 0.75 * uDeep)).r * 255.0 + 0.5);
    for (int k = 0; k < ${MAX_LABELS}; k++) {
      if (k + 1 == label && uLabelOn[k] > 0.5) base = mix(base, uLabelColor[k], uTint);
    }
    vec3 v = -rd;
    vec3 key = normalize(0.55 * uUp - 0.45 * uRight - 0.7 * uForward);
    float diffuse = 0.60 * max(dot(n, v), 0.0) + 0.25 * max(dot(n, key), 0.0);
    float spec = 0.08 * pow(max(dot(reflect(-key, n), v), 0.0), 28.0);
    // Light shows the shape but moves the colour off the scale; uShade says how far.
    col = face ? base : mix(base, base * (0.32 + diffuse) + spec, uShade);
  }
  // Each step's edge as a line about a pixel wide however fast the density
  // changes -- inside the window only, and only where the density is smooth
  // from pixel to pixel: where noise flips a pixel across an edge and back,
  // a line would be speckle, not an edge.
  float fw = fwidth(band);
  if (hit && uLines > 0.5 && uStepWidth > 0.0 && lineDens >= uLo && lineDens <= uHi) {
    float f = fract(band);
    float line = (1.0 - smoothstep(0.6 * fw, 1.6 * fw, min(f, 1.0 - f))) * (1.0 - smoothstep(0.1, 0.3, fw));
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col, lum > 0.42 ? vec3(0.04, 0.06, 0.07) : vec3(0.9, 0.93, 0.92), 0.65 * line);
  }
  outColor = vec4(col, 1.0);
}`;

// Smoothing the scan: a Gaussian along one axis of the volume, one layer of it
// per draw, the three axes in turn. The first pass reads the scan and gives
// each value a weight -- none for a masked volume's background (0) -- the
// passes carry the weighted sum and the weights, and the last divides, so each
// tissue voxel is a mean of tissue alone and a surface keeps its place and its
// density. The background stays empty.
const TAPS = 9;
const BLUR = `#version 300 es
precision highp float;
precision highp sampler3D;
uniform sampler3D uSrc, uScan;
uniform int uLayer, uAxis, uRadius;
uniform float uMasked, uW[${TAPS}];
out vec4 outColor;
void main() {
  ivec3 p = ivec3(ivec2(gl_FragCoord.xy), uLayer), size = textureSize(uSrc, 0);
  ivec3 d = uAxis == 0 ? ivec3(1, 0, 0) : uAxis == 1 ? ivec3(0, 1, 0) : ivec3(0, 0, 1);
  vec2 sum = vec2(0.0);
  for (int k = 1 - ${TAPS}; k < ${TAPS}; k++) {
    ivec3 q = p + d * k;
    if (abs(k) > uRadius || any(lessThan(q, ivec3(0))) || any(greaterThanEqual(q, size))) continue;
    vec2 s = texelFetch(uSrc, q, 0).rg;
    if (uAxis == 0) s = vec2(s.r, uMasked > 0.5 && s.r <= 0.0 ? 0.0 : 1.0);
    sum += uW[abs(k)] * s;
  }
  if (uAxis < 2) { outColor = vec4(sum, 0.0, 1.0); return; }
  bool empty = sum.y <= 1e-6 || (uMasked > 0.5 && texelFetch(uScan, p, 0).r <= 0.0);
  outColor = vec4(empty ? 0.0 : sum.x / sum.y, 0.0, 0.0, 1.0);
}`;

const hexRgb = (hex) => { const v = parseInt(hex.slice(1), 16); return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255]; };
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// A number's IEEE half-float bits, rounded to nearest: the 3-D texture holds
// 16-bit floats, fine enough that a narrow window shows no steps of its own.
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
export function toHalf(value) {
  f32[0] = value;
  const x = u32[0], sign = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112;
  let m = x & 0x7fffff;
  if (e <= 0) {
    if (e < -10) return sign;
    m = (m | 0x800000) >> (1 - e);
    return sign | ((m + 0x1000) >> 13);
  }
  if (e >= 31) return sign | 0x7c00;
  return sign | ((e << 10) + ((m + 0x1000) >> 13));  // a rounding carry moves into the exponent, as it should
}

// The camera's turn for a named side: its columns are the view's right, up and
// back (towards the eye), in the world frame.
function orientation(name) {
  const [yaw, pitch] = VIEWS[name] || VIEWS.side;
  return matMul(axisAngle([0, 1, 0], yaw), axisAngle([1, 0, 0], -pitch));
}

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const copyPose = (p) => ({ rot: [...p.rot], shift: [...p.shift] });

export class Render3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true, alpha: false });
    this.ok = !!this.gl;
    this.error = this.ok ? null : "3-D needs WebGL 2, which this browser does not offer.";
    // rot: the camera's turn (columns right, up, back); it orbits `target` at `dist`.
    this.cam = { rot: orientation("side"), dist: 1, target: [0, 0, 0] };
    // How far the tissue reaches from its centre (what the view fits), and the box's half-diagonal (what a ray crosses).
    this.radius = 1;
    this.boxRadius = 1;
    // The tissue's centre in the box's frame (mm from the box's centre), and the tissue itself (pose.js).
    this.pivot = [0, 0, 0];
    this.cloud = null;
    // deep: how far beneath the surface its colour is read, in mm (null: two voxels);
    // smooth: the radius, in mm, the density there is averaged over (0: one point);
    // shade: how much light shapes the surface, 0 (the colour alone) to 1;
    // steps: {origin, width, fill, lines, palette} in the volume's units -- edges at
    // origin + k x width, each step one colour (fill) and its edges drawn (lines);
    // with a palette ([r, g, b] 0..255 each: the random scale), step k takes its
    // colour k mod its length -- or null;
    // blur: the scan smoothed in 3-D before anything is read from it, a Gaussian's sigma in mm (0: as scanned).
    this.opts = { style: "surface", level: 0, lo: 0, hi: 1, gamma: 1, cmap: "gray", overlays: [], cut: 0, tint: 0.7,
      deep: null, smooth: 0, shade: 1, steps: null, blur: 0 };
    this.volumeKey = null;
    this.moving = false;
    this.onChange = null;
    // (how) after the user moves the view: "camera" (turned, moved or zoomed),
    // "reset" (double-clicked back to the first view) or "align" (turned or moved the specimen alone).
    this.onMove = null;
    // Alt-drag turns or moves the specimen alone, where a page lines several up
    // (off in the viewer); with `alone`, a plain drag does.
    this.alignable = false;
    this.alone = false;
    // The specimen's pose: world = rot x (volume - pivot) + shift. `home` is
    // the pose resetModel returns to: where the page lined it up.
    this.pose = { rot: identity(), shift: [0, 0, 0] };
    this.home = copyPose(this.pose);
    if (!this.ok) return;
    try {
      this.program = this.#program();
    } catch (err) {
      this.ok = false;
      this.error = `3-D could not start: ${err.message}`;
      return;
    }
    this.#controls();
  }

  #program() {
    const gl = this.gl;
    const make = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, make(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, make(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    this.loc = {};
    for (const name of ["uVol", "uLab", "uCmap", "uEye", "uRight", "uUp", "uForward", "uTanHalf", "uAspect", "uExtent",
      "uTexel", "uStep", "uLevel", "uLo", "uHi", "uInvGamma", "uCut", "uTint", "uDeep", "uStyle", "uSteps", "uLabelColor",
      "uLabelOn", "uStage", "uFloor", "uShade", "uAvg", "uStepOrigin", "uStepWidth", "uStepFill", "uLines",
      "uColours", "uPalette", "uBelow", "uAbove"])
      this.loc[name] = gl.getUniformLocation(p, name);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return p;
  }

  // ------------------------------------------------------------ data
  // The quantized preview (uint16) as 8-bit intensities in its full range, and
  // one label per voxel: which structure overlay, if any, covers it.
  setVolume(vol) {
    if (!this.ok) return;
    const key = `${vol.meta.base?.mtime}|${vol.meta.shape.join("x")}|${vol.meta.overlays.map((o) => o.name).join(",")}`;
    if (key === this.volumeKey) return;
    const gl = this.gl;
    const limit = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
    let f = 1;
    while (Math.max(vol.Z, vol.Y, vol.X) / f > limit) f *= 2;
    const nz = Math.ceil(vol.Z / f), ny = Math.ceil(vol.Y / f), nx = Math.ceil(vol.X / f);
    const q = vol.meta.quant, masked = !!vol.meta.masked;
    // Each stored value's place in the texture, as norm() puts it: 0..1, with 0
    // kept for a masked volume's background.
    const lut = new Uint16Array(65536);
    for (let v = 0; v < 65536; v++) {
      const t = clamp((v - q.offset) / q.scale, 0, 1);
      lut[v] = toHalf(masked ? (v === 0 ? 0 : (1 + 254 * t) / 255) : t);
    }
    const intensity = new Uint16Array(nz * ny * nx);
    const labels = new Uint8Array(nz * ny * nx);
    const overlays = vol.meta.overlays.slice(0, MAX_LABELS).map((o) => vol.overlays[o.name]);
    let i = 0;
    for (let z = 0; z < vol.Z; z += f) {
      for (let y = 0; y < vol.Y; y += f) {
        let src = (z * vol.Y + y) * vol.X;
        for (let x = 0; x < vol.X; x += f, i++, src += f) {
          intensity[i] = lut[vol.data[src]];
          for (let k = 0; k < overlays.length; k++) if (overlays[k] && overlays[k][src] >= 128) { labels[i] = k + 1; break; }
        }
      }
    }
    const upload = (unit, data, filter, [format, type]) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, format, nx, ny, nz, 0, gl.RED, type, data);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, filter);
      for (const wrap of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, wrap, gl.CLAMP_TO_EDGE);
      return tex;
    };
    if (this.textures) for (const tex of this.textures) gl.deleteTexture(tex);
    if (this.blurred) { gl.deleteTexture(this.blurred.tex); this.blurred = null; }
    this.textures = [upload(0, intensity, gl.LINEAR, [gl.R16F, gl.HALF_FLOAT]), upload(1, labels, gl.NEAREST, [gl.R8, gl.UNSIGNED_BYTE])];
    const [sz, sy, sx] = vol.meta.spacing_preview_mm;
    // The texture's size and the mm between its texels, along its x, y and z.
    this.size = [nx, ny, nz];
    this.spacing = [sx * f, sy * f, sz * f];
    this.extent = [vol.Z * sz, vol.Y * sy, vol.X * sx];
    this.texel = [1 / nx, 1 / ny, 1 / nz];
    this.voxel = Math.min(sz, sy, sx) * f;
    this.boxRadius = Math.hypot(...this.extent) / 2;
    // Turned about the tissue's centre, and fitted to the tissue, not to the air around it.
    this.cloud = tissueCloud(vol);
    this.pivot = this.cloud ? [...this.cloud.centre] : [0, 0, 0];
    this.radius = this.cloud ? Math.min(this.cloud.radius, 2 * this.boxRadius) : this.boxRadius;
    this.quant = { q, masked };
    this.volumeKey = key;
    this.pose = { rot: identity(), shift: [0, 0, 0] };
    this.home = copyPose(this.pose);
    this.setView("side");
  }

  // A value in the volume's own units -> the 0..1 of the 8-bit texture. Not
  // clamped: a window reaching past the preview's own values (one scale over
  // several specimens) keeps its true width, so each value keeps its colour.
  norm(value) {
    const { q, masked } = this.quant;
    const t = (value - q.lo) / ((q.hi - q.lo) || 1);
    return masked ? (1 + 254 * t) / 255 : t;
  }

  // Steps in the volume's units -> the texture's: where they start and how wide each is.
  stepUniforms(steps) {
    if (!steps || !(steps.width > 0)) return { origin: 0, width: 0, fill: false, lines: false };
    const origin = this.norm(steps.origin), width = this.norm(steps.origin + steps.width) - origin;
    const palette = steps.palette?.length ? steps.palette.slice(0, MOST_COLOURS) : null;
    return width > 1e-7 ? { origin, width, fill: !!steps.fill, lines: !!steps.lines, palette } : { origin: 0, width: 0, fill: false, lines: false };
  }

  setOptions(opts) {
    Object.assign(this.opts, opts);
    if (this.ok && opts.cmap) this.#cmapTexture(opts.cmap);
  }

  #cmapTexture(name) {
    if (this.cmapName === name) return;
    const gl = this.gl;
    const rgb = table(name);
    gl.activeTexture(gl.TEXTURE2);
    if (!this.cmapTex) this.cmapTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.cmapTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, rgb);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.cmapName = name;
  }

  // The scan's texture as the view reads it: smoothed by `sigma` mm, made once
  // and kept, or as scanned (also where smoothing cannot be drawn: it needs
  // float render targets).
  #volumeTexture(sigma) {
    if (!(sigma > 0)) return this.textures[0];
    if (this.blurred?.sigma === sigma) return this.blurred.tex;
    const tex = this.#blur(sigma);
    if (this.blurred) this.gl.deleteTexture(this.blurred.tex);
    this.blurred = tex ? { sigma, tex } : null;
    return tex || this.textures[0];
  }

  #blur(sigma) {
    const gl = this.gl;
    if (!gl.getExtension("EXT_color_buffer_float")) return null;
    if (!this.blurProgram) {
      const make = (type, src) => {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
        return sh;
      };
      const prog = gl.createProgram();
      gl.attachShader(prog, make(gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, make(gl.FRAGMENT_SHADER, BLUR));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
      const loc = {};
      for (const name of ["uSrc", "uScan", "uLayer", "uAxis", "uRadius", "uMasked", "uW"]) loc[name] = gl.getUniformLocation(prog, name);
      this.blurProgram = { prog, loc };
    }
    const { prog, loc } = this.blurProgram, [nx, ny, nz] = this.size;
    const target = (format) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.texStorage3D(gl.TEXTURE_3D, 1, format, nx, ny, nz);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      for (const wrap of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, wrap, gl.CLAMP_TO_EDGE);
      return tex;
    };
    gl.activeTexture(gl.TEXTURE3);
    const a = target(gl.RG16F), b = target(gl.RG16F), out = target(gl.R16F);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.useProgram(prog);
    gl.viewport(0, 0, nx, ny);
    gl.bindVertexArray(this.vao);
    gl.uniform1i(loc.uSrc, 3);
    gl.uniform1i(loc.uScan, 4);
    gl.uniform1f(loc.uMasked, this.quant.masked ? 1 : 0);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_3D, this.textures[0]);
    let ok = true;
    [[this.textures[0], a], [a, b], [b, out]].forEach(([src, dst], axis) => {
      if (!ok) return;
      // Sigma in this axis's texels; weights out to 2.5 sigma, as far as the taps reach.
      const s = sigma / this.spacing[axis], radius = Math.min(TAPS - 1, Math.ceil(2.5 * s));
      const w = new Float32Array(TAPS);
      for (let k = 0; k <= radius; k++) w[k] = Math.exp(-(k * k) / (2 * s * s));
      const total = w.reduce((sum, v, k) => sum + (k ? 2 * v : v), 0);
      gl.uniform1fv(loc.uW, w.map((v) => v / total));
      gl.uniform1i(loc.uAxis, axis);
      gl.uniform1i(loc.uRadius, radius);
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_3D, src);
      for (let z = 0; z < nz; z++) {
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, dst, 0, z);
        if (z === 0 && gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) { ok = false; return; }
        gl.uniform1i(loc.uLayer, z);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
    gl.deleteFramebuffer(fb);
    gl.deleteTexture(a);
    gl.deleteTexture(b);
    if (!ok) { gl.deleteTexture(out); return null; }
    return out;
  }

  // ------------------------------------------------------------ camera
  // The distance from which the whole specimen fits the canvas, however narrow,
  // wherever its pose has moved it.
  fitDistance() {
    const aspect = this.canvas.clientWidth / Math.max(this.canvas.clientHeight, 1) || 1;
    const half = Math.min(FOV / 2, Math.atan(Math.tan(FOV / 2) * aspect));
    return (this.radius + Math.hypot(...this.pose.shift)) / Math.sin(half) * 1.02;
  }

  // A named side, from `dist` (by default, the distance that fits this volume).
  setView(name, dist = this.fitDistance()) {
    this.cam.rot = orientation(name);
    this.cam.target = [0, 0, 0];
    this.cam.dist = dist;
    this.render();
  }

  // Take another view's camera, as a linked view turns with it; drawn fast while it moves.
  setCamera(cam) {
    this.cam = { rot: orthonormal(cam.rot), dist: cam.dist, target: [...cam.target] };
    this.#moved();
  }

  // Turn the camera about its own axes, as far as it goes: dragging right
  // turns the near side right, dragging down tips it down, and on over the
  // top without stopping.
  rotate(dx, dy) {
    this.cam.rot = orthonormal(matMul(this.cam.rot, matMul(axisAngle([0, 1, 0], -dx * TURN), axisAngle([1, 0, 0], -dy * TURN))));
  }

  // Roll the picture by `angle` radians, anticlockwise on screen.
  roll(angle) {
    this.cam.rot = orthonormal(matMul(this.cam.rot, axisAngle([0, 0, 1], -angle)));
  }

  pan(dx, dy) {
    const { right, up } = this.#basis();
    const s = 2 * this.cam.dist * Math.tan(FOV / 2) / Math.max(this.canvas.clientHeight, 1);
    for (let i = 0; i < 3; i++) this.cam.target[i] += (-dx * right[i] + dy * up[i]) * s;
  }

  // ------------------------------------------------------------ the specimen's own pose
  // Turned about its own centre, by the view's axes: exactly as the camera's
  // turn by the same drag looks, so dragging right turns its near side right.
  turnModel(dx, dy) {
    const { right, up } = this.#basis();
    this.#turnPose(matMul(axisAngle(right, dy * TURN), axisAngle(up, dx * TURN)));
  }

  rollModel(angle) {
    const { forward } = this.#basis();
    this.#turnPose(axisAngle(forward.map((v) => -v), angle));
  }

  #turnPose(turn) {
    this.pose.rot = orthonormal(matMul(turn, this.pose.rot));
  }

  // Moved across the view, following the pointer at the specimen's own depth.
  moveModel(dx, dy) {
    const { right, up } = this.#basis();
    const s = 2 * this.#depth() * Math.tan(FOV / 2) / Math.max(this.canvas.clientHeight, 1);
    for (let i = 0; i < 3; i++) this.pose.shift[i] += (dx * right[i] - dy * up[i]) * s;
  }

  // How far in front of the eye the specimen's centre sits.
  #depth() {
    const { eye, forward } = this.#basis();
    const d = dot(sub(this.pose.shift, eye), forward);
    return d > this.radius * 0.05 ? d : this.cam.dist;
  }

  // Where the page lines the specimen up (home: resetModel comes back to it).
  setPose(pose, home = true) {
    this.pose = copyPose(pose);
    if (home) this.home = copyPose(pose);
    this.render();
  }

  resetModel() {
    this.pose = copyPose(this.home);
    this.render();
  }

  // The camera in the volume's own frame: a specimen in a pose looks the same
  // as one in none, seen by a camera moved back by that pose.
  #local(b) {
    const { rot, shift } = this.pose;
    const back = transpose(rot), c = this.pivot;
    return { eye: add(apply(back, sub(b.eye, shift)), c), forward: apply(back, b.forward),
      right: apply(back, b.right), up: apply(back, b.up) };
  }

  #basis() {
    const { rot, dist, target } = this.cam;
    const right = [rot[0], rot[3], rot[6]], up = [rot[1], rot[4], rot[7]], back = [rot[2], rot[5], rot[8]];
    return { eye: target.map((t, i) => t + dist * back[i]), forward: back.map((v) => -v), right, up };
  }

  // The view in the world ({eye, forward, right, up}) and in the volume's own frame (local).
  camera() {
    const world = this.#basis();
    return { ...world, local: this.#local(world) };
  }

  zoom(factor) { this.cam.dist = clamp(this.cam.dist * factor, this.radius * 0.05, this.radius * 30); }

  // While the view moves, draw at half resolution with a coarser step; settle to full after.
  #moved() {
    this.moving = true;
    clearTimeout(this.settle);
    this.settle = setTimeout(() => { this.moving = false; this.render(); }, 160);
    this.render();
  }

  #controls() {
    const c = this.canvas, pointers = new Map();
    let mode = null, pinch = null;
    const pinchState = () => {
      const [a, b] = [...pointers.values()];
      return { d: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, m: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
    };
    // The pointer's angle about the canvas's centre, anticlockwise: what a roll follows.
    const angleAt = ([x, y]) => {
      const r = c.getBoundingClientRect();
      return Math.atan2(r.top + r.height / 2 - y, x - r.left - r.width / 2);
    };
    c.addEventListener("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      // Shift or the right button moves; Ctrl or ⌘ rolls; Alt (or `alone`) does either to the specimen alone.
      const how = e.button === 1 || e.button === 2 || e.shiftKey ? "pan" : e.ctrlKey || e.metaKey ? "roll" : "rotate";
      mode = { how, alone: this.alignable && (e.altKey || this.alone) };
      pinch = pointers.size === 2 ? pinchState() : null;
      c.classList.add("dragging");
    });
    c.addEventListener("pointermove", (e) => {
      if (!pointers.has(e.pointerId)) return;
      const prev = pointers.get(e.pointerId), now = [e.clientX, e.clientY];
      pointers.set(e.pointerId, now);
      const dx = now[0] - prev[0], dy = now[1] - prev[1];
      let how = "camera";
      if (pointers.size === 2 && pinch) {
        const next = pinchState();
        this.zoom(pinch.d / next.d);
        this.pan(next.m[0] - pinch.m[0], next.m[1] - pinch.m[1]);
        pinch = next;
      } else if (mode) {
        let turn = 0;
        if (mode.how === "roll") {
          turn = angleAt(now) - angleAt(prev);
          if (turn > Math.PI) turn -= 2 * Math.PI;
          if (turn < -Math.PI) turn += 2 * Math.PI;
        }
        if (mode.alone) {
          how = "align";
          if (mode.how === "pan") this.moveModel(dx, dy);
          else if (mode.how === "roll") this.rollModel(turn);
          else this.turnModel(dx, dy);
        } else if (mode.how === "pan") this.pan(dx, dy);
        else if (mode.how === "roll") this.roll(turn);
        else this.rotate(dx, dy);
      }
      this.#moved();
      if (this.onMove) this.onMove(how);
    });
    const release = (e) => {
      pointers.delete(e.pointerId);
      pinch = pointers.size === 2 ? pinchState() : null;
      if (!pointers.size) { mode = null; c.classList.remove("dragging"); }
    };
    c.addEventListener("pointerup", release);
    c.addEventListener("pointercancel", release);
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("dblclick", () => { this.setView("side"); if (this.onMove) this.onMove("reset"); });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoom(Math.exp((e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY) * 0.0015));
      this.#moved();
      if (this.onMove) this.onMove("camera");
    }, { passive: false });
  }

  // Arrow keys: turn by a step.
  nudge(dx, dy) { this.rotate(dx, dy); this.#moved(); if (this.onMove) this.onMove("camera"); }

  // ------------------------------------------------------------ drawing
  // One draw per frame, of the latest state. A pending frame is kept, not
  // re-queued: a caller asking every frame would otherwise put it off forever.
  render() {
    if (!this.ok || !this.textures || this.frame) return;
    this.frame = requestAnimationFrame(() => { this.frame = 0; this.#draw(); });
  }

  // Draw now: for a caller already drawing once per frame (the viewer does).
  drawNow() {
    if (!this.ok || !this.textures) return;
    this.#draw();
  }

  #draw() {
    const gl = this.gl, c = this.canvas, o = this.opts;
    const scan = this.#volumeTexture(o.blur || 0);
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * (this.moving ? 0.5 : 1);
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.program);
    const L = this.loc, b = this.#local(this.#basis());
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, scan);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D, this.textures[1]);
    this.#cmapTexture(o.cmap);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.cmapTex);
    gl.uniform1i(L.uVol, 0); gl.uniform1i(L.uLab, 1); gl.uniform1i(L.uCmap, 2);
    gl.uniform3fv(L.uEye, b.eye); gl.uniform3fv(L.uRight, b.right); gl.uniform3fv(L.uUp, b.up); gl.uniform3fv(L.uForward, b.forward);
    gl.uniform1f(L.uTanHalf, Math.tan(FOV / 2)); gl.uniform1f(L.uAspect, w / h);
    gl.uniform3fv(L.uExtent, this.extent); gl.uniform3fv(L.uTexel, this.texel);
    const step = this.voxel * (this.moving ? 1.6 : 0.7);
    gl.uniform1f(L.uStep, step);
    gl.uniform1i(L.uSteps, Math.ceil(2 * this.boxRadius / step) + 2);
    gl.uniform1f(L.uDeep, o.deep ?? this.voxel * 2);
    gl.uniform1f(L.uLevel, Math.max(this.norm(o.level), 1.5 / 255));
    gl.uniform1f(L.uLo, this.norm(o.lo)); gl.uniform1f(L.uHi, this.norm(o.hi));
    gl.uniform1f(L.uInvGamma, 1 / (o.gamma || 1));
    gl.uniform1f(L.uFloor, this.quant.masked ? 0.5 / 255 : -1);
    gl.uniform1f(L.uShade, clamp(o.shade ?? 1, 0, 1));
    gl.uniform1f(L.uAvg, Math.max(0, o.smooth || 0));
    const s = this.stepUniforms(o.steps);
    gl.uniform1f(L.uStepOrigin, s.origin); gl.uniform1f(L.uStepWidth, s.width);
    gl.uniform1f(L.uStepFill, s.fill ? 1 : 0); gl.uniform1f(L.uLines, s.lines ? 1 : 0);
    gl.uniform1f(L.uColours, s.palette ? s.palette.length : 0);
    if (s.palette) {
      const rgb = (c) => c.map((v) => v / 255);
      gl.uniform3fv(L.uPalette, new Float32Array(s.palette.flatMap(rgb)));
      gl.uniform3fv(L.uBelow, rgb(OUTSIDE.below)); gl.uniform3fv(L.uAbove, rgb(OUTSIDE.above));
    }
    // The cut, square to the view through the specimen: 0 keeps everything; 1 slices all its tissue away.
    gl.uniform1f(L.uCut, dot(sub(this.pivot, b.eye), b.forward) - this.radius + clamp(o.cut, 0, 1) * 2 * this.radius);
    gl.uniform1f(L.uTint, o.tint);
    gl.uniform1i(L.uStyle, o.style === "xray" ? 1 : 0);
    const colors = new Float32Array(MAX_LABELS * 3), on = new Float32Array(MAX_LABELS);
    o.overlays.slice(0, MAX_LABELS).forEach((ov, k) => { colors.set(hexRgb(ov.color), k * 3); on[k] = ov.on ? 1 : 0; });
    gl.uniform3fv(L.uLabelColor, colors); gl.uniform1fv(L.uLabelOn, on);
    gl.uniform3fv(L.uStage, hexRgb("#0e1213"));
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    if (this.onChange) this.onChange();
  }

  save(filename) {
    if (!this.ok) return;
    this.moving = false;
    this.#draw();
    this.canvas.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    });
  }

  dispose() {
    clearTimeout(this.settle);
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    const ext = this.gl?.getExtension("WEBGL_lose_context");
    if (ext) ext.loseContext();
  }
}

export const VIEW_NAMES = Object.keys(VIEWS);
