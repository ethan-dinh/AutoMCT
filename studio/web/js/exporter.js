// Exporting the results page: a chart as a Prism-style figure at a chosen
// size and DPI, its data as a Prism table, and everything at once. The page
// hands over entries, {key, spec: () => spec, edit}, whose specs figure.js
// draws and prism.js lays out as tables; this module is the dialogs and the
// files they save. A spec asked for again after an edit carries it.
//
// `edit`, when given, lets the dialog change what the page keeps for the
// chart: edit.title(text | null), edit.label(defaultLabel, text | null) and
// edit.colors(onChange), the page's colour editor. A spec's `defaults` holds
// the heading and axis titles it had before any were renamed.

import { h, modal, replace, toast, errorToast, debounce } from "./lib.js";
import { renderFigure, defaultSize, FIGURE_DEFAULTS } from "./figure.js";
import { tableOf, pzfx, csv, tsv, longCsv } from "./prism.js";
import { pngWithDpi, encodeTiff, zip, saveBlob, slug, utf8 } from "./files.js";

const KEY = "studio.figure-export";
const FORMATS = [["png", "PNG"], ["tiff", "TIFF"], ["svg", "SVG"]];
const DPIS = [150, 300, 600, 1200];
const WIDTHS = [[3.5, "1 column"], [5.5, "1.5 columns"], [7.2, "2 columns"]];
// The largest raster a browser canvas reliably holds.
const MAX_SIDE = 16000, MAX_AREA = 2.5e8;

// The options every figure shares, remembered in this browser; sizes are remembered per chart.
function loadSettings() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch { /* private window */ }
  return { ...FIGURE_DEFAULTS, dpi: 300, format: "png", sizes: {}, ...saved };
}
function saveSettings(s) {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* private window */ }
}
const sizeOf = (settings, entry, spec) => settings.sizes[entry.key] || defaultSize(spec);

/**
 * The page's colours as they print: the light theme's, whatever theme the
 * page shows in, since a figure goes on white. Read once per export, with the
 * theme switched only for the reading, so nothing repaints.
 */
function printColors() {
  const root = document.documentElement;
  const before = root.getAttribute("data-theme");
  root.setAttribute("data-theme", "light");
  const style = getComputedStyle(root);
  const names = [...Array(8).keys()].map((i) => `--s${i + 1}`).concat([0, 1, 2, 3, 4].map((i) => `--lm-${i}`), ["--ink", "--ink-2", "--ink-3"]);
  const known = new Map(names.map((n) => [n, style.getPropertyValue(n).trim()]));
  if (before == null) root.removeAttribute("data-theme"); else root.setAttribute("data-theme", before);
  return (c) => {
    const m = /^var\((--[\w-]+)\)$/.exec(c || "");
    return m ? known.get(m[1]) || "#000000" : c;
  };
}

function pixels(fig, dpi) {
  return [Math.round((fig.width / 72) * dpi), Math.round((fig.height / 72) * dpi)];
}

async function rasterize(fig, dpi, transparent) {
  const [w, hgt] = pixels(fig, dpi);
  if (w > MAX_SIDE || hgt > MAX_SIDE || w * hgt > MAX_AREA) {
    throw new Error(`${w} × ${hgt} pixels is more than the browser can draw. Lower the DPI or the size.`);
  }
  const url = URL.createObjectURL(new Blob([fig.svg(w, hgt)], { type: "image/svg+xml" }));
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = hgt;
    const ctx = canvas.getContext("2d");
    if (!transparent) { ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, w, hgt); }
    ctx.drawImage(img, 0, 0, w, hgt);
    return canvas;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** One figure as a file: {name, data}, named after its title and DPI. */
async function figureFile(spec, options, format, dpi) {
  const fig = renderFigure(spec, options);
  const base = slug(spec.title);
  if (format === "svg") return { name: `${base}.svg`, data: utf8(fig.svg()) };
  const canvas = await rasterize(fig, dpi, options.transparent);
  if (format === "tiff") {
    const pixelsData = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    return { name: `${base}_${dpi}dpi.tif`, data: encodeTiff(pixelsData, canvas.width, canvas.height, dpi, { alpha: options.transparent }) };
  }
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  return { name: `${base}_${dpi}dpi.png`, data: pngWithDpi(new Uint8Array(await blob.arrayBuffer()), dpi) };
}

const MIME = { svg: "image/svg+xml", png: "image/png", tiff: "image/tiff" };

// A labelled setting. Not a <label>: a click on the words would press the first of its buttons.
const group = (label, control, help) => h("div", { class: "field" }, h("span", { class: "lbl" }, label), control,
  help ? h("span", { class: "help" }, help) : null);

// A row of buttons, one pressed.
function seg(label, options, current, pick) {
  const box = h("div", { class: "seg", role: "group", "aria-label": label });
  const draw = (value) => replace(box, options.map(([v, text]) => h("button", {
    type: "button", "aria-pressed": value === v ? "true" : "false", onclick: () => { draw(v); pick(v); },
  }, text)));
  draw(current);
  return box;
}

const number = (value, { min, max, step, onchange, label }) => h("input", {
  type: "number", value, min, max, step, "aria-label": label, class: "num",
  oninput: (e) => { const v = Number(e.target.value); if (Number.isFinite(v) && v >= min && v <= max) onchange(v); },
});

function saveTable(table, format) {
  if (format === "pzfx") saveBlob(pzfx([table], { notes: table.notes || [] }), `${slug(table.title)}.pzfx`, "application/xml");
  else saveBlob(csv(table), `${slug(table.title)}.csv`, "text/csv");
}

// navigator.clipboard needs a secure page; served over http to an iPad
// (python -m studio --tailscale) it is missing, and the old copy command still works.
async function writeClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const area = h("textarea", { readonly: true, style: { position: "fixed", opacity: "0", top: "0", left: "0" } });
  area.value = text;
  document.body.append(area);
  area.select();
  area.setSelectionRange(0, text.length);
  const ok = document.execCommand("copy");
  area.remove();
  if (!ok) throw new Error("This browser would not copy; save the table as CSV instead.");
}

async function copyTable(table) {
  try {
    await writeClipboard(tsv(table));
    toast(table.type === "xy" ? "Copied. In Prism, paste into an XY table from its X title cell."
      : "Copied. In Prism, paste into a Column table from its first title cell.");
  } catch (err) { errorToast(err); }
}

/**
 * One chart as a figure: a preview drawn exactly as it will be saved, beside
 * its size, resolution, format and style, and its data for Prism.
 * `notes`, a function, gives the lines a Prism file's info sheet carries.
 */
export function exportFigure(entry, { notes = () => [] } = {}) {
  let spec = entry.spec();
  if (!spec) return;
  const settings = loadSettings();
  const size = { ...sizeOf(settings, entry, spec) };
  const color = printColors();
  const column = spec.kind !== "xy";
  const preview = h("img", { class: "fig-preview", alt: `Preview of ${spec.title}` });
  const readout = h("div", { class: "fig-readout muted" });
  let url = null;

  const options = () => ({ ...settings, ...size, color });
  const redraw = debounce(() => {
    const fig = renderFigure(spec, options());
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(new Blob([fig.svg()], { type: "image/svg+xml" }));
    preview.src = url;
    preview.style.aspectRatio = `${fig.width} / ${fig.height}`;
    const [w, hgt] = pixels(fig, settings.dpi);
    const mm = (inches) => Math.round(inches * 25.4);
    readout.textContent = `${size.width} × ${size.height} in (${mm(size.width)} × ${mm(size.height)} mm)`
      + (settings.format === "svg" ? ", vector" : `, ${w.toLocaleString()} × ${hgt.toLocaleString()} px at ${settings.dpi} dpi`);
  }, 60);
  const set = (key, value) => { settings[key] = value; saveSettings(settings); redraw(); refresh(); };
  const setSize = (key, value) => {
    size[key] = value;
    settings.sizes[entry.key] = { ...size };
    saveSettings(settings);
    redraw();
  };

  const widthInput = number(size.width, { min: 1, max: 14, step: 0.1, label: "Width in inches", onchange: (v) => setSize("width", v) });
  const heightInput = number(size.height, { min: 1, max: 14, step: 0.1, label: "Height in inches", onchange: (v) => setSize("height", v) });
  const dpiInput = number(settings.dpi, { min: 72, max: 2400, step: 1, label: "Dots per inch", onchange: (v) => set("dpi", Math.round(v)) });
  const check = (key, label) => h("label", { class: "check" },
    h("input", { type: "checkbox", checked: settings[key], onchange: (e) => set(key, e.target.checked) }), label);

  const rasterOnly = h("div", { class: "fig-raster" });
  const refresh = () => {
    rasterOnly.hidden = settings.format === "svg";
    nsCheck.hidden = !column || settings.significance === "off";
  };
  const nsCheck = check("ns", "Mark “ns”");
  // Titles and colours are the page's: changed here, they change there, and the preview follows.
  const edited = () => { spec = entry.spec() || spec; redraw(); };
  const edit = entry.edit;
  const defaults = spec.defaults || {};
  const titleField = (label, value, fallback, change) => h("input", {
    type: "text", value: value || "", placeholder: fallback || "", maxlength: 200, "aria-label": label,
    oninput: (e) => { const text = e.target.value.trim(); change(text && text !== fallback ? text : null); edited(); },
  });
  const titles = edit ? group("Titles", h("div", { class: "fig-titles" },
    h("span", { class: "muted" }, "Title"), titleField("Figure title", spec.heading, defaults.heading, edit.title),
    h("span", { class: "muted" }, "Y axis"), titleField("Y axis title", spec.yLabel, defaults.yLabel, (t) => edit.label(defaults.yLabel, t)),
    column ? null : [h("span", { class: "muted" }, "X axis"), titleField("X axis title", spec.xLabel, defaults.xLabel, (t) => edit.label(defaults.xLabel, t))],
    h("span"), check("title", "Title above the plot")),
  "The title is the chart's on the page too and names the files; the axis titles are the figure's own. Empty a box for the original.") : null;
  const colors = edit?.colors ? h("details", { class: "fig-colors" },
    h("summary", null, "Pick colours: groups, landmarks, lines, axes, text, background"), edit.colors(edited)) : null;
  const form = h("div", { class: "fig-form" },
    titles,
    group("Size", h("div", { class: "row wrap" }, widthInput, h("span", { class: "muted" }, "×"), heightInput, h("span", { class: "muted" }, "in"),
      h("span", { class: "row presets" }, WIDTHS.map(([w, label]) => h("button", {
        type: "button", class: "btn small ghost", title: `${w} in wide, a journal's ${label.toLowerCase()}`,
        onclick: () => { widthInput.value = w; setSize("width", w); },
      }, label)),
      h("button", { type: "button", class: "btn small ghost", title: "The size this chart starts at",
        onclick: () => { const d = defaultSize(spec); widthInput.value = d.width; heightInput.value = d.height; setSize("width", d.width); setSize("height", d.height); } }, "Reset")))),
    group("Format", seg("Format", FORMATS, settings.format, (v) => set("format", v)),
      "PNG or TIFF at the resolution below; SVG stays vector, for Illustrator or Inkscape."),
    h("div", { class: "field-wrap" }, rasterOnly),
    group("Text", h("div", { class: "row" }, number(settings.font, { min: 5, max: 24, step: 0.5, label: "Font size in points", onchange: (v) => set("font", v) }),
      h("span", { class: "muted" }, "pt Arial"), check("bold", "Bold titles and names"))),
    group("Colour", h("div", null, seg("Colour", [["group", "The page's colours"], ["black", "Black, a symbol per group"]], settings.palette, (v) => set("palette", v)),
      colors)),
    column ? group("Plot", h("div", { class: "row wrap" },
      seg("Plot", [["scatter", "Scatter, mean line"], ["bars", "Bars and points"]], settings.style, (v) => set("style", v)),
      seg("Error bars", [["sd", "± SD"], ["sem", "± SEM"]], settings.error, (v) => set("error", v)),
      check("zero", "Y axis from 0"))) : null,
    column ? group("Significance", h("div", { class: "row wrap" },
      seg("Significance", [["stars", "Stars"], ["p", "P values"], ["off", "None"]], settings.significance, (v) => set("significance", v)), nsCheck),
    "Welch t-tests between the groups within each cluster, as the page runs them: * p < 0.05, ** < 0.01, *** < 0.001, **** < 0.0001. Uncorrected for multiple tests.") : null,
    check("transparent", "Transparent background"));
  replace(rasterOnly, group("Resolution", h("div", { class: "row wrap" }, dpiInput, h("span", { class: "muted" }, "dpi"),
    seg("Resolution", DPIS.map((d) => [d, String(d)]), settings.dpi, (v) => { dpiInput.value = v; set("dpi", v); })),
  "300 dpi for photographs and colour, 600 for charts with fine lines, 1200 for line art alone."));
  refresh();

  const table = () => ({ ...tableOf(spec), notes: notes() });
  const saving = { busy: false };
  const saveFigure = async () => {
    if (saving.busy) return false;
    saving.busy = true;
    try {
      const file = await figureFile(spec, options(), settings.format, settings.dpi);
      saveBlob(file.data, file.name, MIME[settings.format]);
    } catch (err) { errorToast(err); }
    saving.busy = false;
    return false;
  };

  modal({
    title: `Export ${spec.title}`,
    wide: true,
    onClose: () => { if (url) URL.revokeObjectURL(url); },
    body: h("div", { class: "fig-dialog" },
      h("div", { class: "fig-stage" }, h("div", { class: "fig-paper" }, preview), readout),
      form),
    actions: [
      { label: "Copy table", onClick: () => { copyTable(table()); return false; } },
      { label: "CSV", onClick: () => { saveTable(table(), "csv"); return false; } },
      { label: "Prism file (.pzfx)", onClick: () => { saveTable(table(), "pzfx"); return false; } },
      { label: "Save figure", primary: true, onClick: saveFigure },
    ],
  });
  redraw();
}

/**
 * Everything on the page at once: every chart shown as a figure, with the
 * options and sizes last used for each, and every measurement (under every
 * switch) as Prism tables, in one project and as CSVs. `figures` and `data`
 * are page entries; `notes` gives the info sheet's lines.
 */
export function exportAll({ figures, data, notes = () => [], name = "microct_results" }) {
  const settings = loadSettings();
  // The tables with something in them, as the page stands while the dialog is open.
  const tables = data.map((e) => e.spec()).filter(Boolean).map(tableOf).filter(tableHasValues);
  const want = { figures: true, pzfx: true, csv: true };
  const status = h("div", { class: "muted fig-status", "aria-live": "polite" });
  const check = (key, label, help) => h("label", { class: "check block" },
    h("input", { type: "checkbox", checked: want[key], onchange: (e) => { want[key] = e.target.checked; } }),
    h("span", null, h("b", null, label), help ? h("span", { class: "muted" }, ` ${help}`) : null));
  const dpiInput = number(settings.dpi, { min: 72, max: 2400, step: 1, label: "Dots per inch", onchange: (v) => { settings.dpi = Math.round(v); saveSettings(settings); } });

  const body = h("div", { class: "fig-all" },
    check("figures", `Figures (${figures.length})`, "each chart as it is shown now, at the size and style last set in its own export"),
    h("div", { class: "row wrap indent" },
      seg("Format", FORMATS, settings.format, (v) => { settings.format = v; saveSettings(settings); }),
      dpiInput, h("span", { class: "muted" }, "dpi")),
    check("pzfx", `Prism project (.pzfx, ${tables.length} table${tables.length === 1 ? "" : "s"})`, "every measurement under every switch, a data table each; open it in Prism to graph and test"),
    check("csv", "CSV tables", "the same tables, one file each, plus every value with its specimen in all_values_long.csv"),
    h("p", { class: "muted" }, "The tables hold each specimen's own value, never a summary, grouped as the page groups them now; hidden and flagged specimens are left out."),
    status);

  const run = async () => {
    if (!want.figures && !want.pzfx && !want.csv) { toast("Pick something to export.", "warning"); return false; }
    const files = [];
    const color = printColors();
    try {
      if (want.figures) {
        let i = 0;
        for (const e of figures) {
          i += 1;
          status.textContent = `Drawing figure ${i} of ${figures.length}…`;
          const spec = e.spec();
          if (!spec) continue;
          const file = await figureFile(spec, { ...settings, ...sizeOf(settings, e, spec), color }, settings.format, settings.dpi);
          files.push({ name: `figures/${String(i).padStart(2, "0")}_${file.name}`, data: file.data });
        }
      }
      const lines = notes();
      if (want.pzfx) files.push({ name: `prism/${name}.pzfx`, data: pzfx(tables, { notes: lines }) });
      if (want.csv) {
        tables.forEach((t, i) => files.push({ name: `csv/${String(i + 1).padStart(2, "0")}_${slug(t.title)}.csv`, data: csv(t) }));
        files.push({ name: "csv/all_values_long.csv", data: longCsv(tables) });
      }
      if (files.length === 1 && want.pzfx) saveBlob(files[0].data, `${name}.pzfx`, "application/xml");
      else {
        files.push({ name: "README.txt", data: readme(lines, want) });
        saveBlob(zip(files), `${name}_prism.zip`, "application/zip");
      }
      status.textContent = `Saved ${files.length} file${files.length === 1 ? "" : "s"}.`;
    } catch (err) {
      status.textContent = "";
      errorToast(err);
    }
    return false;
  };

  modal({ title: "Export for Prism", body, actions: [{ label: "Export", primary: true, onClick: run }] });
}

function tableHasValues(t) {
  if (t.type === "column") return t.columns.some((c) => c.values.length);
  if (t.type === "grouped") return t.datasets.some((d) => d.cells.some((c) => c.length));
  return t.datasets.some((d) => d.subcolumns.some((s) => s.values.some((v) => v != null)));
}

function readme(lines, want) {
  const out = [...lines, ""];
  if (want.figures) out.push("figures/  Each chart as the Results page showed it, drawn in Prism's style.");
  if (want.pzfx) {
    out.push("prism/    One Prism project: a data table per measurement. Column tables hold a group per column;",
      "          Grouped tables a stratum per row, a group per data set and a specimen per subcolumn;",
      "          XY tables the position along the incisor as X and a specimen per subcolumn.",
      "          The project's info sheet lists which specimen sits where in each table.");
  }
  if (want.csv) {
    out.push("csv/      The same tables as CSV, a group per column (a group per stratum for split tables),",
      "          to import into Prism or paste from the first title cell. all_values_long.csv has every",
      "          value with its specimen and group.");
  }
  return out.join("\n") + "\n";
}
