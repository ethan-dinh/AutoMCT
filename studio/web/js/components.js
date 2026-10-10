// Pieces several views share: the parameter form built from a step's spec,
// the pipeline track, the specimen metadata form, the flag dialog, the
// library's folders, and the folder browser.

import { h, icon, replace, spinner, statusInfo, fmtBytes, relTime, modal, errorToast, slotColor } from "./lib.js";
import { api, enc } from "./api.js";
import { state, livePipeline, libraryFolders } from "./store.js";
import { ANALYSES, exclusionText } from "./compare.js";

// ---------------------------------------------------------------- parameters

export function paramsForm(specs, values = {}) {
  const controls = new Map();
  const make = (p) => {
    const v = values[p.key] !== undefined ? values[p.key] : p.default;
    let input, wrap;
    if (p.type === "bool") {
      input = h("input", { type: "checkbox", checked: !!v });
      wrap = h("label", { class: "check field", title: p.help || null }, input, h("span", null, p.label));
      controls.set(p.key, () => input.checked);
      return h("div", { class: "field" }, wrap, p.help ? h("span", { class: "help" }, p.help) : null);
    }
    if (p.type === "select") {
      input = h("select", null, (p.options || []).map((o) => h("option", { value: String(o.value), selected: String(o.value) === String(v) }, o.label)));
      controls.set(p.key, () => {
        const raw = input.value;
        const opt = (p.options || []).find((o) => String(o.value) === raw);
        return opt ? opt.value : raw;
      });
    } else if (p.type === "paths") {
      input = h("textarea", { rows: 3, class: "path", placeholder: p.placeholder || "One path per line" });
      input.value = Array.isArray(v) ? v.join("\n") : (v || "");
      controls.set(p.key, () => input.value.split("\n").map((x) => x.trim()).filter(Boolean));
      const add = h("button", { type: "button", class: "btn small", onclick: async () => {
        const picked = await pickPath({ title: p.label, mode: "file" });
        if (picked) input.value = (input.value.trim() ? input.value.trim() + "\n" : "") + picked;
      } }, "Add…");
      return h("div", { class: "field wide" }, h("span", { class: "lbl" }, p.label), input, h("div", { class: "row" }, add),
        p.help ? h("span", { class: "help" }, p.help) : null);
    } else if (p.type === "path") {
      input = h("input", { type: "text", class: "path", placeholder: p.placeholder || "", value: v || "" });
      controls.set(p.key, () => input.value.trim());
      const browse = h("button", { type: "button", class: "btn", onclick: async () => {
        const picked = await pickPath({ title: p.label, mode: "any", start: input.value });
        if (picked) input.value = picked;
      } }, "Browse…");
      return h("div", { class: "field wide" }, h("span", { class: "lbl" }, p.label),
        h("div", { class: "row" }, input, browse), p.help ? h("span", { class: "help" }, p.help) : null);
    } else if (p.type === "text") {
      input = h("input", { type: "text", placeholder: p.placeholder || "", value: v ?? "" });
      controls.set(p.key, () => input.value.trim());
    } else {
      input = h("input", {
        type: "number", placeholder: p.placeholder || "", value: v ?? "",
        step: p.step ?? (p.type === "int" ? 1 : "any"), min: p.min ?? null, max: p.max ?? null,
      });
      controls.set(p.key, () => {
        if (input.value === "") return null;
        const n = Number(input.value);
        return p.type === "int" ? Math.round(n) : n;
      });
    }
    const control = p.unit ? h("span", { class: "unit-wrap" }, input, h("span", { class: "unit" }, p.unit)) : input;
    return h("label", { class: "field" }, h("span", { class: "lbl" }, p.label), control,
      p.help ? h("span", { class: "help" }, p.help) : null);
  };

  const main = specs.filter((p) => !p.advanced);
  const extra = specs.filter((p) => p.advanced);
  const byGroup = (list) => {
    const groups = new Map();
    for (const p of list) {
      const g = p.group || "";
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(p);
    }
    return [...groups.entries()];
  };
  const render = (list) => byGroup(list).map(([g, ps], i) =>
    h("div", { class: i || g ? "form-group" : "" }, g ? h("h4", null, g) : null, h("div", { class: "form-grid" }, ps.map(make))));
  const el = h("div", null,
    main.length ? render(main) : null,
    extra.length ? h("details", { class: "advanced" }, h("summary", null, `More options (${extra.length})`), render(extra)) : null);
  return {
    el,
    get() {
      const out = {};
      for (const [k, fn] of controls) out[k] = fn();
      return out;
    },
  };
}

// ---------------------------------------------------------------- pipeline track

export function track(summary) {
  // A step switched off for everyone is left out; a step this specimen does
  // not need keeps its place, faint, so the columns line up down the library.
  const steps = livePipeline(summary).filter((p) => p.status !== "disabled");
  return h("span", { class: "track", role: "img", "aria-label": steps.map((p) => `${p.label}: ${statusInfo(p.status).label}`).join(", ") },
    steps.map((p) => {
      const title = `${p.label}: ${statusInfo(p.status).label}${p.stage ? ` (${p.stage})` : ""}${p.reason ? `. ${p.reason}` : ""}${p.finished ? `, ${relTime(p.finished)}` : ""}`;
      if (p.status === "running") return h("span", { class: "m running", title }, spinner(p.progress));
      return h("span", { class: `m ${p.status}`, title }, h("i", { class: "dot" }));
    }));
}

// ---------------------------------------------------------------- flags

/**
 * Flag specimens -- left out of the results, with a note saying why -- or
 * edit the flag they have. Resolves "flagged" or "included" once saved, false
 * when dismissed. `current` is the flag being edited ({from, note}), if any.
 */
export function flagDialog(names, current = null) {
  return new Promise((resolve) => {
    let answered = false;
    const done = (value) => { answered = true; resolve(value); };
    const note = h("textarea", { rows: 3, placeholder: "Why: motion in the scan, a broken incisor tip, the wrong genotype…" });
    note.value = current?.note || "";
    const ticked = new Set(!current || current.from === "all" ? ANALYSES.map(([k]) => k) : current.from);
    const boxes = ANALYSES.map(([k, label]) => h("label", { class: "check" },
      h("input", { type: "checkbox", checked: ticked.has(k), onchange: (e) => { e.target.checked ? ticked.add(k) : ticked.delete(k); } }), label));
    const problem = h("div", { class: "field-error", role: "alert" });
    const save = async (flag) => {
      try {
        await api.post("/api/flags", { names, flag });
        done(flag ? "flagged" : "included");
      } catch (err) { errorToast(err); return false; }
      return true;
    };
    const who = names.length === 1 ? names[0] : `${names.length} specimens`;
    modal({
      title: current ? `Flag on ${who}` : `Flag ${who}`,
      onClose: () => { if (!answered) resolve(false); },
      body: h("div", { class: "stack" },
        h("p", { class: "soft", style: { margin: 0 } }, `${names.length === 1 ? "It stays" : "They stay"} in the library, but the results `
          + `page and the exports leave ${names.length === 1 ? "it" : "them"} out of the analyses ticked below. The note is listed with the results.`),
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Note"), note),
        h("div", { class: "field", role: "group", "aria-label": "Leave out of" },
          h("span", { class: "lbl" }, "Leave out of"), h("div", { class: "row wrap", style: { gap: "8px 18px" } }, boxes)),
        problem),
      actions: [
        { label: "Cancel", onClick: () => done(false) },
        current ? { label: "Include again", onClick: () => save(null) } : null,
        { label: current ? "Save" : "Flag and exclude", primary: true, onClick: () => {
          if (!ticked.size) { problem.textContent = "Tick at least one analysis to leave it out of."; return false; }
          const from = ticked.size === ANALYSES.length ? "all" : ANALYSES.map(([k]) => k).filter((k) => ticked.has(k));
          return save({ note: note.value.trim(), from });
        } },
      ].filter(Boolean),
    });
  });
}

/** "Excluded" beside a flagged specimen's name, its note on hover. */
export function flagTag(flag) {
  if (!flag) return null;
  const what = flag.from === "all" ? "Excluded" : `Excluded from ${exclusionText(flag).toLowerCase()}`;
  return h("span", { class: "tag flagged", title: flag.note || "Flagged, with no note" }, icon("flag"), what);
}

export function groupLabel(summary) {
  return h("span", { class: "group-label" }, h("i", { class: "sw", style: { background: slotColor(summary.slot) } }), summary.group);
}

// ---------------------------------------------------------------- library folders
// The library's own folders, for sorting specimens: not folders on disk.
// Each is a path from the top of the library, "Odam study/Batch 2"; "" is the top.

export const parentOf = (folder) => (folder.includes("/") ? folder.slice(0, folder.lastIndexOf("/")) : "");
export const leafOf = (folder) => folder.slice(folder.lastIndexOf("/") + 1);
export const inFolder = (folder, parent) => !parent || folder === parent || folder.startsWith(`${parent}/`);
export const folderHref = (folder) => (folder ? `#/?folder=${encodeURIComponent(folder)}` : "#/");

/**
 * Library › Odam study › Batch 2, each level a link to it. `decorate(el, folder)`
 * can make a level more than a link, such as somewhere to drop specimens.
 * Another page's folders (the Viewer's) give their own `href` and `root` name.
 */
export function folderCrumbs(folder, decorate = null, { href = folderHref, root = "Library" } = {}) {
  const levels = ["", ...(folder ? folder.split("/").map((_, i, parts) => parts.slice(0, i + 1).join("/")) : [])];
  return h("nav", { class: "folder-crumbs", "aria-label": "Folder" }, levels.map((f, i) => {
    const link = h("a", { href: href(f) }, f ? leafOf(f) : root);
    if (decorate) decorate(link, f);
    return [i ? h("span", { class: "sep", "aria-hidden": "true" }, icon("chevron")) : null, link];
  }));
}

/** A select of every folder, indented under the one it is in, with the top of the library first. */
export function folderSelect(value = "", attrs = {}) {
  const folders = libraryFolders();
  const known = !value || folders.includes(value);
  return h("select", attrs,
    h("option", { value: "", selected: !value }, "Library (no folder)"),
    folders.map((f) => h("option", { value: f, selected: f === value, title: f },
      `${"\u2003".repeat(f.split("/").length - 1)}${leafOf(f)}`)),
    known ? null : h("option", { value, selected: true }, value));
}

/**
 * Choose a folder: every folder in the library as a list to click, and a new
 * one named inside whichever is chosen. Resolves the folder's path ("" for the
 * top of the library), or null when dismissed. Folders in `exclude` (and
 * inside it) cannot be chosen: a folder cannot move into itself.
 */
export function pickFolder({ title, current = "", exclude = null, confirm = "Move", help = null }) {
  return new Promise((resolve) => {
    let answered = false, chosen = current;
    const done = (value) => { answered = true; resolve(value); };
    const list = h("ul", { role: "listbox", "aria-label": "Folders" });
    const fresh = h("input", { type: "text", placeholder: "Leave blank to use the folder chosen above", maxlength: 80 });
    const where = h("span", { class: "help" });
    const draw = () => {
      replace(list, ["", ...libraryFolders()].map((f) => {
        const off = exclude != null && f && inFolder(f, exclude);
        return h("li", null, h("button", {
          type: "button", role: "option", "aria-selected": f === chosen ? "true" : "false", disabled: off,
          style: { paddingLeft: `${8 + 18 * (f ? f.split("/").length : 0)}px` },
          onclick: () => { chosen = f; draw(); },
          ondblclick: () => { chosen = f; finish(); },
        }, icon(f ? "folder" : "library"), h("span", null, f ? leafOf(f) : "Library (no folder)"),
          f === current ? h("span", { class: "sz" }, "now here") : null));
      }));
      where.textContent = `Made inside ${chosen ? chosen.split("/").join(" › ") : "the top of the library"}.`;
    };
    const target = () => {
      const name = fresh.value.trim();
      return name ? [chosen, name].filter(Boolean).join("/") : chosen;
    };
    const finish = () => { done(target()); m.close(); };
    fresh.addEventListener("keydown", (e) => { if (e.key === "Enter") finish(); });
    draw();
    const m = modal({
      title,
      onClose: () => { if (!answered) resolve(null); },
      body: h("div", { class: "stack" },
        help ? h("p", { class: "soft", style: { margin: 0 } }, help) : null,
        h("div", { class: "browser folder-pick" }, list),
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Or a new folder"), fresh, where)),
      actions: [
        { label: "Cancel", onClick: () => done(null) },
        { label: confirm, primary: true, onClick: () => done(target()) },
      ],
    });
  });
}

/** Ask for a folder's new name. Resolves the name (one level), or null when dismissed. */
export function nameFolder({ title, value = "", confirm = "Save", help = null }) {
  return new Promise((resolve) => {
    let answered = false;
    const input = h("input", { type: "text", value, maxlength: 80, placeholder: "Odam study, Batch 2, UCSF scans…" });
    const problem = h("div", { class: "field-error", role: "alert" });
    const check = () => {
      const name = input.value.trim().replace(/\s+/g, " ");
      if (!name) { problem.textContent = "Give the folder a name."; return null; }
      if (name.includes("/") || name.includes("\\")) { problem.textContent = "A folder name cannot hold a slash."; return null; }
      return name;
    };
    const m = modal({
      title,
      onClose: () => { if (!answered) resolve(null); },
      body: h("div", { class: "stack" },
        help ? h("p", { class: "soft", style: { margin: 0 } }, help) : null,
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Name"), input), problem),
      actions: [
        { label: "Cancel", onClick: () => { answered = true; resolve(null); } },
        { label: confirm, primary: true, onClick: () => {
          const name = check();
          if (!name) return false;
          answered = true; resolve(name);
          return true;
        } },
      ],
    });
    input.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      const name = check();
      if (name) { answered = true; resolve(name); m.close(); }
    });
    setTimeout(() => input.select(), 0);
  });
}

// ---------------------------------------------------------------- metadata

export const META_LABELS = {
  age: "Age", line: "Mouse line", genotype: "Genotype", replicate: "Replicate", sex: "Sex", side: "Side",
  study: "Study", scan_date: "Scan date", scanner: "Scanner", voxel_um: "Voxel size", notes: "Notes",
};

// Every detail a specimen carries, in form order, with how it is entered.
// `group` says which details make up the name ({Age}_{BaseMouse}_{Gene}_{Sex}_{N}).
export const META_COLUMNS = [
  { key: "age", type: "text", group: "name", width: 78, placeholder: "P52" },
  { key: "line", type: "text", group: "name", width: 124, placeholder: "Odam-Cre" },
  { key: "genotype", type: "text", group: "name", width: 124, placeholder: "WT" },
  { key: "sex", type: "select", group: "name", width: 96, options: [["F", "Female"], ["M", "Male"]] },
  { key: "replicate", type: "int", group: "name", width: 62 },
  { key: "side", type: "select", group: "animal", width: 86, options: [["L", "Left"], ["R", "Right"]] },
  { key: "study", type: "text", group: "scan", width: 150 },
  { key: "scan_date", type: "date", group: "scan", width: 142 },
  { key: "scanner", type: "text", group: "scan", width: 158, placeholder: "Scanco µCT 50" },
  { key: "voxel_um", type: "number", group: "scan", width: 86, unit: "µm" },
  { key: "notes", type: "text", group: "notes", width: 220 },
];

// Where an autofilled detail came from, as its tooltip says it.
export function provenance(info) {
  if (!info) return "";
  return info.guess ? `${info.from}. A guess: check it.` : `${info.from}.`;
}

export function fmtVoxel(v) {
  if (v == null) return "";
  return Array.isArray(v) ? `${v.join(" × ")} µm (Z × Y × X)` : `${v} µm`;
}

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

function isoDay(y, m, d) {
  let year = Number(y);
  const month = Number(m), day = Number(d);
  if (year < 100) year += 2000;
  const t = new Date(Date.UTC(year, month - 1, day));
  if (t.getUTCFullYear() !== year || t.getUTCMonth() !== month - 1 || t.getUTCDate() !== day) return null;
  if (year < 1990 || year > new Date().getFullYear() + 1) return null;
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

// A date as people and spreadsheets write one, as YYYY-MM-DD; null if unreadable.
// Slashes read month first (3/14/2025), dots day first (14.03.2025).
export function parseDate(text) {
  const s = String(text ?? "").trim();
  const month = (w) => MONTHS.indexOf(w.slice(0, 3).toLowerCase()) + 1;
  let m;
  if ((m = /^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b/.exec(s))) return isoDay(m[1], m[2], m[3]);
  if ((m = /^(\d{4})(\d{2})(\d{2})$/.exec(s))) return isoDay(m[1], m[2], m[3]);
  if ((m = /^(\d{1,2})\/(\d{1,2})\/(\d{4}|\d{2})\b/.exec(s))) return isoDay(m[3], m[1], m[2]);
  if ((m = /^(\d{1,2})\.(\d{1,2})\.(\d{4})\b/.exec(s))) return isoDay(m[3], m[2], m[1]);
  if ((m = /^(\d{1,2})[\s-]+([A-Za-z]{3,9})\.?[\s,-]+(\d{4})\b/.exec(s))) return month(m[2]) ? isoDay(m[3], month(m[2]), m[1]) : null;
  if ((m = /^([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b/.exec(s))) return month(m[1]) ? isoDay(m[3], month(m[1]), m[2]) : null;
  return null;
}

const SEX_WORDS = { f: "F", female: "F", fem: "F", "♀": "F", m: "M", male: "M", "♂": "M" };
const SIDE_WORDS = { l: "L", left: "L", lt: "L", r: "R", right: "R", rt: "R" };

// A detail's value read from typed or pasted text: "" when blank, undefined
// when the text is not something the detail can hold.
export function readMetaValue(col, text) {
  const s = String(text ?? "").trim();
  if (!s) return "";
  if (col.type === "select") return (col.key === "sex" ? SEX_WORDS : SIDE_WORDS)[s.toLowerCase()] ?? (col.options.some(([v]) => v === s) ? s : undefined);
  if (col.type === "int") {
    const n = Number(s.replace(/^[#nN]\s*/, ""));
    return Number.isInteger(n) && n > 0 ? n : undefined;
  }
  if (col.type === "number") {
    const m = /\d+(?:[.,]\d+)?/.exec(s);
    if (!m) return undefined;
    let n = Number(m[0].replace(",", "."));
    if (/\bmm\b/i.test(s) && n < 1) n *= 1000;
    return n > 0 ? Math.round(n * 1000) / 1000 : undefined;
  }
  if (col.type === "date") return parseDate(s) ?? undefined;
  return s;
}

// The metadata form. `sources` says where autofilled values came from (a guess
// is tinted until it is edited) and `recorded` what the file itself states,
// such as its own voxel size.
export function metaForm(values = {}, suggestions = {}, onInput = () => {}, { sources = {}, recorded = {} } = {}) {
  const inputs = {};
  const marks = { ...sources };
  let facts = { ...recorded };
  const list = (key) => {
    const id = `dl-${key}-${Math.random().toString(36).slice(2, 7)}`;
    return [id, h("datalist", { id }, (suggestions[key] || []).map((v) => h("option", { value: v })))];
  };
  const edited = (key) => () => {
    if (marks[key]) { delete marks[key]; paint(key); }
    if (key === "voxel_um") paint(key);
    onInput();
  };
  const text = (key, placeholder) => {
    const [id, dl] = list(key);
    inputs[key] = h("input", { type: "text", list: id, placeholder, value: values[key] ?? "", oninput: edited(key) });
    return h("label", { class: "field" }, h("span", { class: "lbl" }, META_LABELS[key]), inputs[key], dl);
  };
  const select = (key, options) => {
    inputs[key] = h("select", { onchange: edited(key) }, [["", "Not recorded"], ...options].map(([v, l]) =>
      h("option", { value: v, selected: String(values[key] ?? "") === v }, l)));
    return h("label", { class: "field" }, h("span", { class: "lbl" }, META_LABELS[key]), inputs[key]);
  };
  inputs.replicate = h("input", { type: "number", min: 1, step: 1, placeholder: "next free", value: values.replicate ?? "", oninput: edited("replicate") });
  inputs.voxel_um = h("input", { type: "number", min: 0, step: "any", placeholder: "from the file", value: values.voxel_um ?? "", oninput: edited("voxel_um") });
  inputs.scan_date = h("input", { type: "date", value: values.scan_date ?? "", oninput: edited("scan_date") });
  inputs.notes = h("textarea", { rows: 2, oninput: edited("notes") });
  inputs.notes.value = values.notes ?? "";
  const voxelHelp = h("span", { class: "help" });

  // Tint a guess, and say in the tooltip where any autofilled value came from.
  function paint(key) {
    const input = inputs[key], info = marks[key];
    input.classList.toggle("guess", Boolean(info?.guess));
    input.title = provenance(info);
    if (key !== "voxel_um") return;
    const need = facts.needs_voxel && input.value.trim() === "";
    input.classList.toggle("need", Boolean(need));
    input.placeholder = facts.voxel_um != null && !Array.isArray(facts.voxel_um) ? String(facts.voxel_um) : "from the file";
    voxelHelp.textContent = need ? "This scan records no voxel size. Set it here, or its measurements come out wrong."
      : facts.voxel_um != null ? `The file records ${fmtVoxel(facts.voxel_um)}. Leave blank to use it.`
        : "Only needed when the file records none (BMP stacks).";
  }

  const el = h("div", { class: "form-grid" },
    text("age", "P52"),
    text("line", "Odam-Cre"),
    text("genotype", "WT"),
    select("sex", [["F", "Female"], ["M", "Male"]]),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Replicate"), inputs.replicate),
    select("side", [["L", "Left"], ["R", "Right"]]),
    text("study", ""),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Scan date"), inputs.scan_date),
    text("scanner", "Scanco µCT 50"),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Voxel size"),
      h("span", { class: "unit-wrap" }, inputs.voxel_um, h("span", { class: "unit" }, "µm")), voxelHelp),
    h("label", { class: "field wide" }, h("span", { class: "lbl" }, "Notes"), inputs.notes));
  Object.keys(inputs).forEach(paint);
  return {
    el,
    get() {
      const out = {};
      for (const [k, input] of Object.entries(inputs)) {
        const v = input.value.trim();
        if (v === "") continue;
        out[k] = ["replicate"].includes(k) ? Math.round(Number(v)) : k === "voxel_um" ? Number(v) : v;
      }
      return out;
    },
    // Fill the fields still empty, noting where each value came from; returns the keys filled.
    set(values, from = {}, file = null) {
      if (file) facts = { ...facts, ...file };
      const filled = [];
      for (const [k, v] of Object.entries(values || {})) {
        if (!inputs[k] || v == null || v === "" || inputs[k].value !== "") continue;
        inputs[k].value = v;
        if (inputs[k].value === "") continue;
        if (from[k]) marks[k] = from[k];
        filled.push(k);
      }
      Object.keys(inputs).forEach(paint);
      return filled;
    },
    sources: () => ({ ...marks }),
    recorded: () => ({ ...facts }),
  };
}

// ---------------------------------------------------------------- folder browser

export function browser({ start, onSelect, onOpen, mode = "any" }) {
  const el = h("div", { class: "browser" });
  let current = null, selected = null;
  const load = async (path) => {
    try {
      const data = await api.get(`/api/fs?path=${enc(path || "")}`);
      current = data;
      render();
    } catch (err) { errorToast(err); }
  };
  const render = () => {
    // The server splits the path, so a Windows drive (E:\) reads as one step.
    const steps = current.crumbs || [{ label: current.path, path: current.path }];
    const crumbs = h("div", { class: "crumbs" }, steps.map((c, i) => [
      i > 1 ? h("span", { class: "muted" }, "/") : null,
      h("button", { type: "button", onclick: () => load(c.path) }, c.label)]));
    const roots = h("div", { class: "roots" }, current.roots.map((r) =>
      h("button", { type: "button", class: "chip", onclick: () => load(r.path) }, icon("folder"), r.label)));
    const items = h("ul", { role: "listbox" },
      current.parent ? h("li", null, h("button", { type: "button", onclick: () => load(current.parent) }, icon("back"), "..")) : null,
      current.entries.map((e) => h("li", null, h("button", {
        type: "button", "aria-selected": selected === e.path ? "true" : "false",
        onclick: () => { selected = e.path; onSelect && onSelect(e); render(); },
        ondblclick: () => { if (e.dir && !(onOpen && onOpen(e))) load(e.path); else if (!e.dir && onOpen) onOpen(e); },
      }, icon(e.dir ? "folder" : e.hint === "isq" || e.hint === "volume" ? "cube" : "file"), h("span", null, e.name),
        e.hint ? h("span", { class: e.hint === "workspace" ? "tag accent" : "tag" }, e.hint) : null,
        h("span", { class: "sz" }, e.dir ? "" : fmtBytes(e.size))))));
    replace(el, crumbs, roots, items);
  };
  load(start);
  return { el, reload: () => load(current?.path), path: () => current?.path, selected: () => selected, mode };
}

export function pickPath({ title = "Choose a file or folder", start = "", mode = "any",
  help = "Double-click a folder to open it. Select a folder itself to import it as one sample." } = {}) {
  return new Promise((resolve) => {
    let chosen = null, done = false;
    const pathInput = h("input", { type: "text", class: "path", placeholder: "/path/to/data", style: { width: "100%" }, value: start || "" });
    const b = browser({
      start, mode,
      onSelect: (e) => { chosen = e.path; pathInput.value = e.path; },
    });
    const m = modal({
      title, wide: true,
      body: h("div", { class: "stack" }, b.el, pathInput, h("span", { class: "muted" }, help)),
      onClose: () => { if (!done) resolve(null); },
      actions: [
        { label: "Cancel", onClick: () => { done = true; resolve(null); } },
        { label: "Use this folder", onClick: () => { done = true; resolve(b.path()); } },
        { label: "Choose", primary: true, onClick: () => { done = true; resolve(pathInput.value.trim() || chosen || b.path()); } },
      ],
    });
    return m;
  });
}

export function stepSpec(id) {
  return state.app?.steps.find((s) => s.id === id);
}
