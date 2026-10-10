// Files the browser writes itself: a PNG that records its DPI, a TIFF, and a
// ZIP of several files. No DOM here except saveBlob, so the tests can run the
// encoders under Node and read what they write back with Pillow and zipfile.

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes, start = 0, end = bytes.length) {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export const utf8 = (text) => new TextEncoder().encode(text);

// ---------------------------------------------------------------- PNG

/**
 * The PNG with a pHYs chunk saying it is `dpi` pixels to the inch, so Word,
 * Illustrator and journal checkers read its print size. The canvas writes
 * none; any it did write is replaced.
 */
export function pngWithDpi(png, dpi) {
  const src = png instanceof Uint8Array ? png : new Uint8Array(png);
  const view = new DataView(src.buffer, src.byteOffset, src.byteLength);
  const perMetre = Math.round(dpi / 0.0254);
  const chunk = new Uint8Array(21);
  const cv = new DataView(chunk.buffer);
  cv.setUint32(0, 9);
  chunk.set(utf8("pHYs"), 4);
  cv.setUint32(8, perMetre);
  cv.setUint32(12, perMetre);
  chunk[16] = 1;                                   // unit: the metre
  cv.setUint32(17, crc32(chunk, 4, 17));
  // Copy chunk by chunk, dropping any pHYs and putting ours after IHDR.
  const parts = [src.subarray(0, 8)];
  let at = 8;
  while (at < src.length) {
    const length = view.getUint32(at);
    const type = String.fromCharCode(...src.subarray(at + 4, at + 8));
    const next = at + 12 + length;
    if (type !== "pHYs") parts.push(src.subarray(at, next));
    if (type === "IHDR") parts.push(chunk);
    at = next;
  }
  return concat(parts);
}

// ---------------------------------------------------------------- TIFF

// One row, PackBits-compressed (TIFF compression 32773): runs of a byte as
// (1 - n, byte), anything else as (n - 1, n bytes).
function packRow(src, start, end, out, o) {
  let i = start;
  while (i < end) {
    let run = 1;
    while (i + run < end && run < 128 && src[i + run] === src[i]) run++;
    if (run >= 3) {
      out[o++] = 257 - run;
      out[o++] = src[i];
      i += run;
      continue;
    }
    // A literal stretch, up to the next run of three.
    let j = i;
    while (j < end && j - i < 128 && !(j + 2 < end && src[j] === src[j + 1] && src[j] === src[j + 2])) j++;
    if (j === i) j = Math.min(i + 1, end);
    out[o++] = j - i - 1;
    for (let k = i; k < j; k++) out[o++] = src[k];
    i = j;
  }
  return o;
}

/**
 * An RGB (or RGBA, `alpha`) TIFF of canvas pixels, PackBits-compressed, that
 * records `dpi` as its resolution. `rgba` is getImageData's data: four bytes
 * a pixel, not premultiplied.
 */
export function encodeTiff(rgba, width, height, dpi, { alpha = false } = {}) {
  const spp = alpha ? 4 : 3;
  const rowBytes = width * spp;
  const rowsPerStrip = Math.max(1, Math.min(height, Math.floor(65536 / Math.max(rowBytes, 1))));
  const strips = Math.ceil(height / rowsPerStrip);
  // Compress every strip first: the header records where each one lands.
  const row = new Uint8Array(rowBytes);
  const packed = new Uint8Array(rowBytes + Math.ceil(rowBytes / 128) + 1);
  const stripData = [];
  for (let s = 0; s < strips; s++) {
    const chunks = [];
    for (let y = s * rowsPerStrip; y < Math.min(height, (s + 1) * rowsPerStrip); y++) {
      for (let x = 0, p = y * width * 4, q = 0; x < width; x++, p += 4) {
        row[q++] = rgba[p]; row[q++] = rgba[p + 1]; row[q++] = rgba[p + 2];
        if (alpha) row[q++] = rgba[p + 3];
      }
      chunks.push(packed.slice(0, packRow(row, 0, rowBytes, packed, 0)));
    }
    stripData.push(concat(chunks));
  }

  const software = utf8("MicroCT Studio\0");
  const entries = [];   // [tag, type, count, value or bytes]
  const SHORT = 3, LONG = 4, RATIONAL = 5, ASCII = 2;
  const shorts = (...v) => { const b = new Uint8Array(v.length * 2); const d = new DataView(b.buffer); v.forEach((x, i) => d.setUint16(i * 2, x, true)); return b; };
  const longs = (v) => { const b = new Uint8Array(v.length * 4); const d = new DataView(b.buffer); v.forEach((x, i) => d.setUint32(i * 4, x, true)); return b; };
  const rational = (num, den) => longs([num, den]);
  // DPI as a fraction to 1/100 of a dot.
  const res = rational(Math.round(dpi * 100), 100);
  entries.push([256, LONG, 1, longs([width])], [257, LONG, 1, longs([height])],
    [258, SHORT, spp, shorts(...Array(spp).fill(8))], [259, SHORT, 1, shorts(32773)],
    [262, SHORT, 1, shorts(2)], [273, LONG, strips, null], [277, SHORT, 1, shorts(spp)],
    [278, LONG, 1, longs([rowsPerStrip])], [279, LONG, strips, longs(stripData.map((d) => d.length))],
    [282, RATIONAL, 1, res], [283, RATIONAL, 1, res], [284, SHORT, 1, shorts(1)],
    [296, SHORT, 1, shorts(2)], [305, ASCII, software.length, software]);
  if (alpha) entries.push([338, SHORT, 1, shorts(2)]);   // unassociated alpha

  const ifdSize = 2 + entries.length * 12 + 4;
  // Values longer than four bytes follow the directory, word-aligned; then the strips.
  let extra = 8 + ifdSize;
  const offsets = new Map();
  const stripOffsetsAt = () => entries.find((e) => e[0] === 273);
  for (const e of entries) {
    const size = e[0] === 273 ? strips * 4 : e[3].length;
    if (size > 4) { offsets.set(e[0], extra); extra += size + (size % 2); }
  }
  const stripOffsets = [];
  let at = extra;
  for (const d of stripData) { stripOffsets.push(at); at += d.length + (d.length % 2); }
  stripOffsetsAt()[3] = longs(stripOffsets);

  const out = new Uint8Array(at);
  const dv = new DataView(out.buffer);
  out.set([0x49, 0x49, 42, 0], 0);
  dv.setUint32(4, 8, true);
  dv.setUint16(8, entries.length, true);
  entries.forEach(([tag, type, count, bytes], i) => {
    const p = 10 + i * 12;
    dv.setUint16(p, tag, true);
    dv.setUint16(p + 2, type, true);
    dv.setUint32(p + 4, count, true);
    if (offsets.has(tag)) {
      dv.setUint32(p + 8, offsets.get(tag), true);
      out.set(bytes, offsets.get(tag));
    } else {
      out.set(bytes, p + 8);
    }
  });
  dv.setUint32(10 + entries.length * 12, 0, true);
  stripData.forEach((d, i) => out.set(d, stripOffsets[i]));
  return out;
}

// ---------------------------------------------------------------- ZIP

/**
 * A ZIP of [{name, data}] (data: text or bytes), stored without compression:
 * the figures are PNG or TIFF already compressed, and the tables are small.
 * Names are UTF-8 and may hold folders ("figures/density.png").
 */
export function zip(files, when = new Date()) {
  const time = (when.getHours() << 11) | (when.getMinutes() << 5) | (when.getSeconds() >> 1);
  const date = ((when.getFullYear() - 1980) << 9) | ((when.getMonth() + 1) << 5) | when.getDate();
  const locals = [], central = [];
  let offset = 0;
  for (const f of files) {
    const name = utf8(f.name);
    const data = typeof f.data === "string" ? utf8(f.data) : f.data;
    const crc = crc32(data);
    const head = new Uint8Array(30 + name.length);
    const hv = new DataView(head.buffer);
    hv.setUint32(0, 0x04034b50, true);
    hv.setUint16(4, 20, true);
    hv.setUint16(6, 0x0800, true);               // names in UTF-8
    hv.setUint16(8, 0, true);                    // stored
    hv.setUint16(10, time, true);
    hv.setUint16(12, date, true);
    hv.setUint32(14, crc, true);
    hv.setUint32(18, data.length, true);
    hv.setUint32(22, data.length, true);
    hv.setUint16(26, name.length, true);
    head.set(name, 30);
    locals.push(head, data);
    const dir = new Uint8Array(46 + name.length);
    const dv = new DataView(dir.buffer);
    dv.setUint32(0, 0x02014b50, true);
    dv.setUint16(4, 20, true);
    dv.setUint16(6, 20, true);
    dv.setUint16(8, 0x0800, true);
    dv.setUint16(10, 0, true);
    dv.setUint16(12, time, true);
    dv.setUint16(14, date, true);
    dv.setUint32(16, crc, true);
    dv.setUint32(20, data.length, true);
    dv.setUint32(24, data.length, true);
    dv.setUint16(28, name.length, true);
    dv.setUint32(42, offset, true);
    dir.set(name, 46);
    central.push(dir);
    offset += head.length + data.length;
  }
  const dirSize = central.reduce((a, b) => a + b.length, 0);
  const end = new Uint8Array(22);
  const ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, dirSize, true);
  ev.setUint32(16, offset, true);
  return concat([...locals, ...central, end]);
}

// ---------------------------------------------------------------- helpers

export function concat(parts) {
  const out = new Uint8Array(parts.reduce((a, b) => a + b.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

/** A file name from a title: "Enamel density, mean (mg HA/cm³)" -> "enamel_density_mean_mg_ha_cm3". */
export function slug(text) {
  return String(text).replace(/[µμ]/g, "u").normalize("NFKD")
    .replace(/[^\w]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase().slice(0, 80) || "figure";
}

export function saveBlob(data, filename, type = "application/octet-stream") {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}
