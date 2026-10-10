// The specimens a folder scan found, as a sheet with every detail a column.
//
// Rows arrive filled in with what the scans' headers, names and folders say.
// A guess (from a folder's name or the name typed at the scanner) is tinted
// until someone checks it. Finishing the rest is quick: the top row sets a
// column for every selected specimen, a cell's corner drags its value up or
// down, cells paste straight from a spreadsheet (matched to specimens by scan
// name when a column holds them), and Enter moves down a row. The server
// plans the names as the details change, so each row shows the name it will
// be imported under.

import { h, icon, replace, toast, errorToast, debounce, modal, basename } from "../lib.js";
import { api } from "../api.js";
import { META_COLUMNS, META_LABELS, provenance, fmtVoxel, readMetaValue } from "../components.js";

const COL = Object.fromEntries(META_COLUMNS.map((c) => [c.key, c]));
// The details the name is made of: changing one changes the name.
const NAME_FIELDS = new Set(["age", "line", "genotype", "sex", "replicate"]);
// The table's columns in order, for pasting a block into it.
const ORDER = ["name", ...META_COLUMNS.map((c) => c.key)];
const PIN = 36, NAME_W = 250;
const GROUPS = [["name", "In the name"], ["animal", ""], ["scan", "Scan"], ["notes", ""]];
// Column headers where the form's label is too long for a narrow column.
const SHORT_LABELS = { line: "Line", replicate: "Rep.", voxel_um: "Voxel" };

// Spreadsheet headers, squeezed to letters, and the column each fills.
const HEADERS = {
  name: ["specimen", "name", "sample", "samplename", "specimenname", "id", "sampleid", "specimenid", "scan", "scanid",
    "scanname", "file", "filename", "measurement", "measurementno"],
  age: ["age", "ageday", "agedays", "postnatalday", "pnd"],
  line: ["line", "mouseline", "strain", "basemouse", "background"],
  genotype: ["genotype", "gene", "geno"],
  replicate: ["rep", "replicate", "repl", "n"],
  sex: ["sex", "gender"],
  side: ["side", "hemimandible", "hemi"],
  study: ["study", "project", "experiment", "cohort"],
  scan_date: ["scandate", "date", "datescanned", "scanned", "dateofscan", "acquisitiondate"],
  scanner: ["scanner", "instrument", "machine", "device"],
  voxel_um: ["voxel", "voxelsize", "voxelum", "resolution", "pixelsize"],
  notes: ["notes", "note", "comment", "comments", "remarks"],
};
const HEADER_OF = new Map(Object.entries(HEADERS).flatMap(([key, words]) => words.map((w) => [w, key])));
const headerKey = (text) => HEADER_OF.get(String(text).toLowerCase().replace(/\(.*?\)/g, "").replace(/[^a-z]/g, "")) || null;

// A name as a spreadsheet might spell it: no case, no extension, no separators.
const squeeze = (s) => String(s ?? "").toLowerCase().replace(/\.(isq|rsq|nrrd|tiff?|dcm|bmp)$/, "").replace(/[\s_.-]+/g, "");
// A scan's number, so 4019 in a spreadsheet finds C0004019.ISQ.
const digitsOf = (s) => { const d = String(s ?? "").replace(/\D/g, "").replace(/^0+/, ""); return d.length >= 2 ? d : ""; };
const looksNumbered = (s) => /^[A-Za-z#]{0,2}\s*\d{2,}$/.test(String(s ?? "").trim());

const plural = (n, word = "specimen") => `${n} ${word}${n === 1 ? "" : "s"}`;
const blank = (v) => v === "" || v == null;
// Enough to name a specimen in the lab's convention (the server's composes_name).
const composes = (m) => !blank(String(m.age ?? "").trim()) && !blank(String(m.genotype ?? "").trim());

let counter = 0;

export function foundSheet({ rows, suggestions = {}, sourceLine, commit }) {
  const uid = ++counter;
  const els = new Map();       // row key -> its elements
  const history = [];          // snapshots to undo fills, pastes and whole-column sets
  let planSeq = 0, anchor = null, undoNote = null;

  const live = () => rows.filter((r) => r.include && !r.exists);
  const shown = (key, v) => COL[key]?.options?.find(([o]) => o === v)?.[1] ?? v;
  const clean = (meta) => Object.fromEntries(Object.entries(meta).filter(([, v]) => !blank(v) && String(v).trim() !== "")
    .map(([k, v]) => [k, typeof v === "string" ? v.trim() : v]));

  // ------------------------------------------------ suggestions
  const lists = Object.fromEntries(META_COLUMNS.filter((c) => c.type === "text")
    .map((c) => [c.key, h("datalist", { id: `sheet${uid}-${c.key}` })]));
  function refreshLists() {
    for (const [key, dl] of Object.entries(lists)) {
      const values = new Set([...(suggestions[key] || []), ...rows.map((r) => r.meta[key]).filter((v) => !blank(v)).map(String)]);
      replace(dl, [...values].sort((a, b) => a.localeCompare(b)).map((v) => h("option", { value: v })));
    }
  }

  // ------------------------------------------------ cells
  function control(col, bulk = false) {
    let el;
    if (col.type === "select") {
      const options = bulk ? [["", "All…"], ...col.options, ["-", "Not recorded"]] : [["", ""], ...col.options];
      el = h("select", null, options.map(([v, l]) => h("option", { value: v }, l)));
    } else if (col.type === "date") {
      el = h("input", { type: "date" });
    } else if (col.type === "int" || col.type === "number") {
      el = h("input", { type: "number", min: col.type === "int" ? 1 : 0, step: col.type === "int" ? 1 : "any" });
    } else {
      el = h("input", { type: "text", list: lists[col.key]?.id, autocomplete: "off", spellcheck: "false" });
    }
    el.setAttribute("aria-label", bulk ? `${META_LABELS[col.key]} for every selected specimen` : META_LABELS[col.key]);
    return el;
  }

  function paint(r, key) {
    const { cells, inputs } = els.get(r.key);
    const td = cells[key], info = r.sources[key];
    let title = provenance(info);
    td.classList.toggle("guess", Boolean(info?.guess) && !r.exists);
    if (key === "voxel_um") {
      const file = r.recorded?.voxel_um;
      const need = Boolean(r.recorded?.needs_voxel) && blank(r.meta.voxel_um) && !r.exists;
      td.classList.toggle("need", need);
      inputs.voxel_um.placeholder = file == null ? (need ? "needed" : "") : Array.isArray(file) ? `${Math.min(...file)}–${Math.max(...file)}` : String(file);
      if (need) title = "This scan records no voxel size. Set it, or its measurements come out wrong.";
      else if (file != null && blank(r.meta.voxel_um)) title = `The file records ${fmtVoxel(file)}. Leave blank to use it.`;
    }
    if (key === "replicate") {
      // Only a name made from the details takes a number.
      const auto = r.planned?.from === "details" && blank(r.meta.replicate);
      inputs.replicate.placeholder = auto ? String(r.planned.replicate) : "";
      if (auto) title = `Numbered automatically: ${r.planned.replicate} is the next free number.`;
    }
    td.title = title;
  }

  function drawName(r) {
    const e = els.get(r.key), p = r.planned;
    const value = r.typed || (r.exists ? r.existing || r.name : p?.name || r.name);
    if (document.activeElement !== e.name && e.name.value !== value) e.name.value = value;
    e.name.classList.toggle("bad", Boolean(p?.error) && !r.exists);
    const line = sourceLine(r);
    replace(e.sub,
      p?.error && !r.exists ? h("span", { class: "err" }, p.error) : h("span", { title: line.title || null }, line.text),
      r.typed && !r.exists ? h("button", { type: "button", class: "linkish", onclick: () => {
        r.typed = ""; e.name.value = ""; replan();
      } }, "Name it from the details") : null);
  }

  function refreshRow(r) {
    const e = els.get(r.key);
    if (!e) return;
    e.tr.classList.toggle("off", !r.include && !r.exists);
    e.check.checked = r.include && !r.exists;
    for (const col of META_COLUMNS) {
      const v = r.meta[col.key], el = e.inputs[col.key];
      const text = blank(v) ? "" : String(v);
      if (el.value !== text) el.value = text;
      paint(r, col.key);
    }
    drawName(r);
  }

  // A detail typed into a row: no longer a guess, and maybe a new name.
  function edited(r, key, el) {
    const col = COL[key];
    r.meta[key] = el.value === "" ? "" : col.type === "int" ? Math.round(Number(el.value)) : col.type === "number" ? Number(el.value) : el.value;
    if (r.sources[key]) delete r.sources[key];
    paint(r, key);
    if (NAME_FIELDS.has(key)) replan();
    drawLegend();
  }

  function buildRow(r, i) {
    const check = h("input", { type: "checkbox", disabled: r.exists, "aria-label": `Import ${r.name}` });
    check.addEventListener("click", (ev) => toggle(i, check.checked, ev.shiftKey));
    const name = h("input", { type: "text", class: "name-input", spellcheck: "false", autocomplete: "off",
      disabled: r.exists, "aria-label": `Name for ${r.name}`, dataset: { col: "name" } });
    name.addEventListener("input", () => {
      const v = name.value.trim();
      r.typed = v && !(r.planned && r.planned.from !== "typed" && v === r.planned.name) ? v : "";
      replan();
    });
    // Emptied by hand, it shows the name it will get again once left.
    name.addEventListener("blur", () => drawName(r));
    const sub = h("div", { class: "specimen-sub" });
    const cells = {}, inputs = {};
    const tds = META_COLUMNS.map((col) => {
      const el = control(col);
      el.dataset.col = col.key;
      el.disabled = r.exists;
      el.addEventListener(col.type === "select" ? "change" : "input", () => edited(r, col.key, el));
      inputs[col.key] = el;
      return (cells[col.key] = h("td", { dataset: { col: col.key } }, el));
    });
    const tr = h("tr", { dataset: { i }, class: r.exists ? "exists" : null },
      h("td", { class: "pin" }, check), h("td", { class: "pin2 name-cell", dataset: { col: "name" } }, name, sub), tds);
    els.set(r.key, { tr, check, name, sub, cells, inputs });
    refreshRow(r);
    return tr;
  }

  // ------------------------------------------------ selecting
  function toggle(i, on, shift) {
    const [a, b] = shift && anchor != null ? [Math.min(anchor, i), Math.max(anchor, i)] : [i, i];
    for (let j = a; j <= b; j++) if (!rows[j].exists) { rows[j].include = on; refreshRow(rows[j]); }
    anchor = i;
    counts();
    replan();
  }

  // ------------------------------------------------ undo
  function snapshot() {
    history.push(rows.map((r) => ({ meta: { ...r.meta }, sources: { ...r.sources }, typed: r.typed })));
    if (history.length > 40) history.shift();
    undoBtn.disabled = false;
  }
  function undo() {
    const last = history.pop();
    if (!last) return;
    last.forEach((s, i) => Object.assign(rows[i], s));
    rows.forEach(refreshRow);
    undoBtn.disabled = !history.length;
    if (undoNote) { undoNote.remove(); undoNote = null; }
    refreshLists();
    drawLegend();
    replan();
  }
  // One toast at a time, so its Undo is always the latest change.
  function done(message) {
    if (undoNote) undoNote.remove();
    undoNote = toast(message, "good", { action: { label: "Undo", onClick: undo }, timeout: 9000 });
  }
  function changed(keys) {
    refreshLists();
    drawLegend();
    if ([...keys].some((k) => k === "name" || NAME_FIELDS.has(k))) replan();
  }

  // ------------------------------------------------ set a whole column
  function setAll(key, value) {
    const targets = live();
    if (!targets.length) return toast("Select the specimens to set first.", "info");
    snapshot();
    for (const r of targets) { r.meta[key] = value; delete r.sources[key]; refreshRow(r); }
    changed([key]);
    const what = META_LABELS[key].toLowerCase();
    done(blank(value) ? `Cleared ${what} for ${plural(targets.length)}.`
      : `Set ${what} to ${shown(key, value)} for ${plural(targets.length)}.`);
  }

  const fillLabel = h("span");
  const fillRow = h("tr", { class: "fill" }, h("th", { class: "pin fill-label", colspan: 2 }, fillLabel),
    META_COLUMNS.map((col) => {
      if (col.key === "replicate") {
        return h("th", { title: "Left blank, each specimen takes the next free number." }, h("span", { class: "muted" }, "auto"));
      }
      const el = control(col, true);
      if (col.type !== "select") el.placeholder = "All";
      const apply = () => {
        const raw = el.value;
        if (raw === "") return;
        el.value = "";
        if (col.type === "select") return setAll(col.key, raw === "-" ? "" : raw);
        const v = readMetaValue(col, raw);
        if (v === undefined || v === "") return toast(`“${raw}” is not a ${META_LABELS[col.key].toLowerCase()}.`, "warning");
        setAll(col.key, v);
      };
      if (col.type === "select") el.addEventListener("change", apply);
      else {
        el.addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); apply(); } });
        el.addEventListener("blur", apply);
      }
      return h("th", null, el);
    }));

  // ------------------------------------------------ drag a cell's corner to copy it
  const handle = h("span", { class: "fill-handle", title: "Drag up or down to copy this value", "aria-hidden": "true" });
  handle.addEventListener("mousedown", (ev) => ev.preventDefault());   // keep the cell focused
  handle.addEventListener("pointerdown", startFill);

  function startFill(ev) {
    const td = handle.parentElement;
    if (!td) return;
    const key = td.dataset.col, from = Number(td.parentElement.dataset.i);
    ev.preventDefault();
    handle.setPointerCapture(ev.pointerId);
    let to = from, x = ev.clientX, y = ev.clientY, raf = 0;
    const inRange = (i) => (from < to ? i > from && i <= to : i < from && i >= to);
    const mark = () => rows.forEach((r, i) => els.get(r.key).cells[key].classList.toggle("fill-range", inRange(i) && !r.exists));
    // The row under the pointer, held to the rows in view.
    const locate = () => {
      const box = wrap.getBoundingClientRect(), top = thead.getBoundingClientRect().bottom;
      const tr = document.elementFromPoint(x, Math.min(Math.max(y, top + 2), box.bottom - 2))?.closest?.("tr[data-i]");
      if (tr && tr.parentElement === tbody && Number(tr.dataset.i) !== to) { to = Number(tr.dataset.i); mark(); }
    };
    const tick = () => {
      // Past the edge of the table, keep scrolling to reach rows out of view.
      const box = wrap.getBoundingClientRect(), top = thead.getBoundingClientRect().bottom;
      if (y > box.bottom - 28) wrap.scrollTop += 14;
      else if (y < top + 18) wrap.scrollTop -= 14;
      locate();
      raf = requestAnimationFrame(tick);
    };
    const move = (e) => { x = e.clientX; y = e.clientY; locate(); };
    const end = (e) => {
      if (e?.type === "pointerup") move(e);
      cancelAnimationFrame(raf);
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", end);
      handle.removeEventListener("pointercancel", end);
      rows.forEach((r) => els.get(r.key).cells[key].classList.remove("fill-range"));
      if (to !== from) copyRange(from, to, key);
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
    raf = requestAnimationFrame(tick);
  }

  function copyRange(from, to, key) {
    const value = rows[from].meta[key] ?? "";
    const [a, b] = from < to ? [from + 1, to] : [to, from - 1];
    const targets = rows.slice(a, b + 1).filter((r) => !r.exists);
    if (!targets.length) return;
    snapshot();
    for (const r of targets) { r.meta[key] = value; delete r.sources[key]; refreshRow(r); }
    changed([key]);
    done(blank(value) ? `Cleared ${META_LABELS[key].toLowerCase()} for ${plural(targets.length)}.`
      : `Copied ${shown(key, value)} to ${plural(targets.length)}.`);
  }

  // ------------------------------------------------ paste from a spreadsheet
  function matcher() {
    const byName = new Map(), byNumber = new Map();
    const add = (map, k, i) => { if (k) map.set(k, map.has(k) && map.get(k) !== i ? -1 : i); };
    rows.forEach((r, i) => {
      for (const v of [r.name, r.planned?.name, r.existing, r.typed, r.recorded?.scanner_name, r.source?.path && basename(r.source.path)]) add(byName, squeeze(v), i);
      add(byNumber, digitsOf(r.name), i);
    });
    return (v) => {
      let i = byName.get(squeeze(v));
      if ((i == null || i < 0) && looksNumbered(v)) i = byNumber.get(digitsOf(v));
      return i == null || i < 0 ? null : i;
    };
  }

  // The pasted column that names the specimens, if one does: most of its
  // values are the name, file or scan number of a row here.
  function keyColumn(body) {
    const find = matcher();
    const width = Math.max(0, ...body.map((line) => line.length));
    let best = null;
    for (let j = 0; j < width; j++) {
      const values = body.map((line) => (line[j] || "").trim()).filter(Boolean);
      const hits = values.filter((v) => find(v) != null).length;
      if (hits && hits * 2 >= values.length && (!best || hits > best.hits)) best = { col: j, hits };
    }
    return best ? { col: best.col, find } : null;
  }

  function put(r, key, raw) {
    const text = String(raw ?? "").trim();
    if (!text) return false;          // a blank cell leaves what is there
    if (key === "name") { r.typed = text; return true; }
    const v = readMetaValue(COL[key], text);
    if (v === undefined) return "unread";
    r.meta[key] = v;
    delete r.sources[key];
    return true;
  }

  // Paste a block of cells: by header, matched by scan name when a column
  // holds them; otherwise by position, from the cell pasted into.
  function pasteText(text, row0 = 0, col0 = 1) {
    const grid = String(text || "").replace(/\r\n?/g, "\n").replace(/\n+$/, "").split("\n").map((line) => line.split("\t"));
    if (!grid.length || !grid.some((line) => line.some((c) => c.trim()))) return false;
    const heads = grid[0].map(headerKey);
    const named = heads.filter(Boolean).length;
    const header = named > 0 && named * 2 >= grid[0].filter((c) => c.trim()).length;
    const body = header ? grid.slice(1) : grid;
    const keys = header ? heads : grid[0].map((_, j) => ORDER[col0 + j] || null);
    const match = header ? keyColumn(body) : null;
    const ignored = header ? grid[0].filter((c, j) => c.trim() && !heads[j] && j !== match?.col).map((c) => c.trim()) : [];
    snapshot();
    let cells = 0, unread = 0, over = 0;
    const touched = new Set(), unmatched = [], columns = new Set();
    body.forEach((line, n) => {
      let i = row0 + n;
      if (match) {
        i = match.find(line[match.col]);
        if (i == null) { if ((line[match.col] || "").trim()) unmatched.push(line[match.col].trim()); return; }
      }
      const r = rows[i];
      if (!r) { if (line.some((c) => c.trim())) over++; return; }
      if (r.exists) return;
      line.forEach((raw, j) => {
        const key = keys[j];
        if (!key || (match && j === match.col)) return;
        const res = put(r, key, raw);
        if (res === "unread") unread++;
        else if (res) { cells++; touched.add(r); columns.add(key); }
      });
    });
    if (!cells) {
      history.pop();
      undoBtn.disabled = !history.length;
      toast(match || header ? "Nothing to fill in: no pasted row matched a specimen here with a value to give."
        : "Nothing to fill in from what was pasted.", "warning", { timeout: 8000 });
      return false;
    }
    touched.forEach(refreshRow);
    changed(columns);
    const notes = [`Filled ${plural(cells, "cell")} for ${plural(touched.size)}${match ? ", matched by name" : ""}.`];
    if (unmatched.length) notes.push(`${plural(unmatched.length, "row")} matched no specimen: ${unmatched.slice(0, 4).join(", ")}${unmatched.length > 4 ? "…" : ""}.`);
    if (over) notes.push(`${plural(over, "row")} ran past the last specimen.`);
    if (unread) notes.push(`${plural(unread, "value")} could not be read and ${unread === 1 ? "was" : "were"} left out.`);
    if (ignored.length) notes.push(`Not in the table: ${ignored.slice(0, 4).join(", ")}.`);
    done(notes.join(" "));
    return true;
  }

  function pasteDialog() {
    // Spaces, not tabs, in the example: a placeholder ignores tab stops.
    const area = h("textarea", { rows: 10, class: "path paste-area", spellcheck: "false",
      placeholder: "Scan        Age   Genotype   Sex\nC0004019    P52   WT         F\nC0004020    P52   cKO        M" });
    modal({
      title: "Paste from a spreadsheet", wide: true,
      body: h("div", { class: "stack" },
        h("p", { class: "soft" }, "Copy the rows from Excel, Numbers or Google Sheets with their header row, and paste them here. "
          + "Columns are matched to the table by their headers. When a column holds the scan or specimen names, each row goes to "
          + "the specimen it names, in any order; otherwise rows fill the table from the top."),
        area,
        h("span", { class: "muted" }, "Blank cells leave what is already filled in.")),
      actions: [
        { label: "Cancel" },
        { label: "Fill in", primary: true, onClick: () => pasteText(area.value, 0, 1) },
      ],
    });
  }

  // ------------------------------------------------ names
  const replan = debounce(async () => {
    const seq = ++planSeq;
    const targets = live();
    try {
      const res = await api.post("/api/names/plan", {
        items: targets.map((r) => ({ key: r.key, meta: clean(r.meta), found: r.name, name: r.typed })),
      });
      if (seq !== planSeq) return;
      const by = new Map(res.names.map((n) => [n.key, n]));
      for (const r of rows) {
        r.planned = by.get(r.key) || null;
        if (els.has(r.key)) { drawName(r); paint(r, "replicate"); }
      }
      counts();
    } catch (err) {
      if (seq === planSeq) errorToast(err);
    }
  }, 160);

  // ------------------------------------------------ the frame
  const legend = h("div", { class: "sheet-legend" });
  function drawLegend() {
    const open = rows.filter((r) => !r.exists);
    const guesses = open.reduce((n, r) => n + Object.values(r.sources).filter((f) => f.guess).length, 0);
    const needs = open.filter((r) => r.recorded?.needs_voxel && blank(r.meta.voxel_um)).length;
    replace(legend,
      guesses ? h("span", { class: "key" }, h("i", { class: "sw guess" }),
        `${plural(guesses, "detail")} guessed from folder names or the name typed at the scanner. Check them.`,
        h("button", { type: "button", class: "btn small ghost", onclick: () => {
          snapshot();
          for (const r of rows) for (const [k, f] of Object.entries(r.sources)) if (f.guess) r.sources[k] = { ...f, guess: false };
          rows.forEach(refreshRow);
          drawLegend();
          done("Marked every guess as checked.");
        } }, icon("check"), "Mark all as checked")) : null,
      needs ? h("span", { class: "key" }, h("i", { class: "sw need" }),
        `${plural(needs, "scan")} ${needs === 1 ? "records" : "record"} no voxel size. Set it in the Voxel column.`) : null);
  }

  const undoBtn = h("button", { class: "btn small", disabled: true, onclick: undo }, icon("back"), "Undo");
  const help = h("p", { class: "sheet-help" },
    "Type in the All row to set a column for every selected specimen. Drag the corner of a cell to copy it up or down, "
    + "or paste cells straight from a spreadsheet. Enter moves down a row; ", h("kbd", null, navigator.platform?.startsWith("Mac") ? "⌘D" : "Ctrl+D"),
    " copies the row above.");
  const bar = h("div", { class: "sheet-bar" }, help,
    h("div", { class: "sheet-tools" }, h("button", { class: "btn small", onclick: pasteDialog }, icon("file"), "Paste from a spreadsheet…"), undoBtn));

  const all = h("input", { type: "checkbox", "aria-label": "Select every specimen" });
  all.addEventListener("change", () => {
    for (const r of rows) if (!r.exists) r.include = all.checked;
    rows.forEach(refreshRow);
    counts();
    replan();
  });
  const widths = [PIN, NAME_W, ...META_COLUMNS.map((c) => c.width)];
  const groupsRow = h("tr", { class: "groups" }, h("th", { class: "pin", colspan: 2 }),
    GROUPS.map(([g, label]) => h("th", { colspan: META_COLUMNS.filter((c) => c.group === g).length },
      label ? h("span", { title: g === "name" ? "Age, line, genotype, sex and replicate make up the specimen's name" : null }, label) : null)));
  const labelsRow = h("tr", { class: "labels" }, h("th", { class: "pin" }, all), h("th", { class: "pin2" }, "Specimen"),
    META_COLUMNS.map((c) => h("th", { title: META_LABELS[c.key] }, SHORT_LABELS[c.key] || META_LABELS[c.key],
      c.unit ? h("span", { class: "unit" }, ` ${c.unit}`) : null)));
  const thead = h("thead", null, groupsRow, labelsRow, fillRow);
  // Specimens already imported come last, under a label of their own.
  const firstHeld = rows.findIndex((r) => r.exists);
  const tbody = h("tbody", null, rows.map((r, i) => [
    i === firstHeld ? h("tr", { class: "sep" }, h("td", { colspan: widths.length },
      h("span", null, i ? `Already in the library (${rows.length - i})` : "Everything found is already in the library"))) : null,
    buildRow(r, i)]));
  const table = h("table", { class: "data sheet", style: { width: `${widths.reduce((a, b) => a + b, 0)}px` } },
    h("colgroup", null, widths.map((w) => h("col", { style: { width: `${w}px` } }))), thead, tbody);
  const wrap = h("div", { class: "table-wrap sheet-wrap" }, table, Object.values(lists));
  // Each header row sticks below the ones above it, whatever their height.
  new ResizeObserver(() => {
    const g = groupsRow.getBoundingClientRect().height, l = labelsRow.getBoundingClientRect().height;
    table.style.setProperty("--t1", `${g}px`);
    table.style.setProperty("--t2", `${g + l}px`);
    table.style.setProperty("--t3", `${g + l + fillRow.getBoundingClientRect().height}px`);
  }).observe(thead);

  tbody.addEventListener("focusin", (ev) => {
    const td = ev.target.closest?.("td[data-col]");
    if (!td || td.dataset.col === "name" || td.parentElement.classList.contains("exists")) handle.remove();
    else td.appendChild(handle);
  });
  const nextRow = (i, dir) => {
    for (let j = i + dir; j >= 0 && j < rows.length; j += dir) if (!rows[j].exists) return j;
    return null;
  };
  const focusCell = (i, key) => {
    if (i == null) return;
    const e = els.get(rows[i].key), el = key === "name" ? e.name : e.inputs[key];
    el.focus({ preventScroll: true });
    el.scrollIntoView({ block: "nearest", inline: "nearest" });
    if (el.type === "text") el.select();
  };
  tbody.addEventListener("keydown", (ev) => {
    const key = ev.target.dataset?.col;
    if (!key) return;
    const i = Number(ev.target.closest("tr").dataset.i);
    if (ev.key === "Enter" && !ev.metaKey && !ev.ctrlKey && !ev.altKey) {
      ev.preventDefault();
      focusCell(nextRow(i, ev.shiftKey ? -1 : 1), key);
    } else if ((ev.metaKey || ev.ctrlKey) && !ev.altKey && !ev.shiftKey && ev.key.toLowerCase() === "d" && key !== "name") {
      ev.preventDefault();
      const j = nextRow(i, -1);
      if (j == null) return;
      rows[i].meta[key] = rows[j].meta[key] ?? "";
      delete rows[i].sources[key];
      refreshRow(rows[i]);
      changed([key]);
    }
  });
  tbody.addEventListener("change", (ev) => { if (ev.target.list) refreshLists(); });
  tbody.addEventListener("paste", (ev) => {
    const key = ev.target.dataset?.col;
    const text = ev.clipboardData?.getData("text/plain") || "";
    if (!key) return;
    const i = Number(ev.target.closest("tr").dataset.i);
    // A block spreads over the table.
    if (/[\t\n]/.test(text.replace(/\r?\n$/, ""))) {
      ev.preventDefault();
      pasteText(text, i, ORDER.indexOf(key));
      return;
    }
    // One value pastes into a text cell as usual; a date or number is read
    // the way a spreadsheet writes it (3/14/2025, 9 µm) rather than refused.
    if (key === "name" || COL[key].type === "text") return;
    ev.preventDefault();
    if (put(rows[i], key, text) === "unread") return toast(`“${text.trim()}” is not a ${META_LABELS[key].toLowerCase()}.`, "warning");
    refreshRow(rows[i]);
    changed([key]);
  });

  // ------------------------------------------------ importing
  const runAfter = h("input", { type: "checkbox" });
  const importBtn = h("button", { class: "btn primary" });
  const footNote = h("span", { class: "muted sheet-note" });
  function counts() {
    const targets = live(), n = targets.length;
    const open = rows.filter((r) => !r.exists);
    all.checked = n > 0 && n === open.length;
    all.indeterminate = n > 0 && n < open.length;
    fillLabel.textContent = n ? `All ${plural(n)} selected` : "Select specimens to set them together";
    const bad = targets.filter((r) => r.planned?.error).length;
    const kept = targets.filter((r) => r.planned?.from === "found" && !composes(r.meta)).length;
    importBtn.textContent = `Import ${plural(n)}`;
    importBtn.disabled = !n || bad > 0;
    footNote.textContent = bad ? `Fix the ${plural(bad, "name")} marked in red first.`
      : kept ? `${kept} ${kept === 1 ? "keeps" : "keep"} the name ${kept === 1 ? "it was" : "they were"} found under until ${kept === 1 ? "it has" : "they have"} an age and a genotype.`
        : "";
  }
  importBtn.addEventListener("click", async () => {
    const targets = live();
    importBtn.disabled = true;
    try {
      await commit(targets.map((r) => ({ found: r.name, name: r.typed || "", meta: clean(r.meta), source: r.source, artifacts: r.artifacts })),
        runAfter.checked);
    } catch (err) {
      errorToast(err);
      counts();
    }
  });

  refreshLists();
  drawLegend();
  counts();
  replan();
  return h("div", { class: "sheet" }, legend, bar, wrap,
    h("div", { class: "sheet-foot" }, importBtn, h("label", { class: "check" }, runAfter, "Run missing steps after import"), footNote));
}
