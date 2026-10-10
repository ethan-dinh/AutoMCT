// Results: every specimen's measurements side by side, compared by the
// details the reader picks. The first is the main comparison -- its levels
// side by side, one colour each -- and any further ones split the specimens
// into clusters and panels, with the main comparison repeated in each.
// One bar above everything scopes the whole page: what to compare, which
// specimens (a checklist, or a selection brought from the Library) and which
// values. The page opens on its projects (projectsPage, at the end): every
// specimen, or a project -- a named set of specimens with its own comparison
// and filters, kept in the workspace so every browser has it. The filters sit
// in a region that opens from a one-line summary, and reach only the
// project's own specimens. Flagged specimens are left out of the analyses
// their flag names and listed at the end with its note.
// Every colour and every chart's title can be changed (chartstyle.js); the
// figures exported take the same.

import { h, icon, replace, fmtPct, pctFmt, sig, numFmt, slotColor, errorToast, statusBadge, debounce, rafThrottle, ordinal, toMgHA, relTime, toast, modal, confirmDialog } from "../lib.js";
import { api } from "../api.js";
import { on, state, libraryFolders } from "../store.js";
import { lineChart, stripPlot, chartCard, dataTable, extent, landmarkTone } from "../charts.js";
import { smoothProfile, coverageFloor, valueAt } from "../trajectory.js";
import { go } from "../app.js";
import { FACTORS, factor, levelOf, levelName, levelOrder, factorsWithValues, arrange, describe, comparisons, excludedFrom, exclusionText } from "../compare.js";
import { flagDialog, flagTag, inFolder } from "../components.js";
import { stars } from "../figure.js";
import { exportFigure, exportAll } from "../exporter.js";
import { colorEditor, onStyle, setStyle, groupKey, groupColor, landmarkColor, markColor, titleOf, labelOf, figureInk, applyPageInk } from "../chartstyle.js";

// The enamel curves use the slices the server's cleanup kept. The dentin peak is
// fitted wherever a slice resolved, and dentin is there before enamel starts,
// so its line runs from the cervical loop.
const DENTIN_MEASURED = ["resolved_hi", "resolved_lo", "coincident", "saturated", "pre_onset"];
const MAX_SPLITS = 2;
const GRID = Array.from({ length: 101 }, (_, i) => i);

const dens = (v) => sig(v, 4);
const finite = (v) => v != null && Number.isFinite(v);

// The molar section's three cards and the switches each one has. Axis ticks
// take as many decimals as their step needs ("0.09", not "0.0900").
const ticked = (fy) => (v, step) => (step != null ? numFmt(v, step) : fy(v));
const sig3 = ticked((v) => sig(v, 3));
const dens4 = ticked(dens);
const DENSITY_UNITS = [["mgHA", "mg HA/cm³"], ["raw", "Raw intensity"], ["ratio", "Enamel/dentin"]];
const DENSITY_STATS = ["mean", "low", "high"];
const VOLUME_MEASURES = {
  share: { label: "Enamel share", yLabel: "Enamel share of volume", fy: pctFmt, cell: (v) => fmtPct(v, 1) },
  enamel: { label: "Enamel", yLabel: "Enamel volume (mm³)", fy: sig3 },
  dentin: { label: "Dentin", yLabel: "Dentin volume (mm³)", fy: sig3 },
  total: { label: "Whole tooth", yLabel: "Volume (mm³)", fy: sig3 },
};
const THICKNESS_MEASURES = {
  thickness: { label: "Thickness", yLabel: "Enamel thickness, median (µm)", fy: sig3 },
  junction: { label: "Junction area", yLabel: "Enamel–dentin junction area (mm²)", fy: sig3 },
  perArea: { label: "Thickness ÷ area", yLabel: "Thickness ÷ junction area (µm/mm²)", fy: sig3 },
};
// Patches of enamel thinner, or less dense, than the enamel around them on the
// same tooth (quantify_molar/enamel_patches.py): shares are of the judged
// enamel–dentin junction; a shortfall is the one-sided spread, no threshold.
const PATCH_MEASURES = {
  thinShare: { label: "Thin patches", key: "thin_patch_share", share: true, fy: pctFmt,
    title: "junction under thin patches (%)", sub: "share of the judged enamel–dentin junction under thin patches: enamel thinner than the enamel around it as only 1% of the P22 wildtypes' is (45% thinner by default)" },
  thinPerMm2: { label: "Thin, per mm²", key: "thin_patches_per_mm2", fy: sig3,
    title: "thin patches per mm² of junction", sub: "thin patches per mm² of judged enamel–dentin junction" },
  thinShortfall: { label: "Thickness shortfall", key: "thickness_shortfall", share: true, fy: pctFmt,
    title: "thickness shortfall (%)", sub: "how far the enamel falls short of the enamel around it, root-mean-square over the judged junction: no threshold" },
  lowShare: { label: "Less dense patches", key: "low_density_patch_share", share: true, fy: pctFmt,
    title: "junction under less dense patches (%)", sub: "share of the judged enamel–dentin junction under enamel whose core holds less mineral than the enamel around it as only 1% of the P22 wildtypes' does (12% less by default)" },
  lowPerMm2: { label: "Less dense, per mm²", key: "low_density_patches_per_mm2", fy: sig3,
    title: "less dense patches per mm² of junction", sub: "less dense patches per mm² of judged enamel–dentin junction" },
  densityShortfall: { label: "Density shortfall", key: "density_shortfall", share: true, fy: pctFmt,
    title: "density shortfall (%)", sub: "how far the core's mineral falls short of the enamel around it, root-mean-square over the judged junction: no threshold" },
};
const TEETH = [["all", "All molars"], ["M1", "M1"], ["M2", "M2"], ["M3", "M3"]];

function fmtP(p) {
  if (p == null) return "–";
  return p < 0.001 ? "< 0.001" : sig(p, 2);
}

// What the page remembers per workspace: how it compares, not what it hides.
// What it hides lasts as long as the browser tab (below), so opening a
// specimen and coming back keeps it, and a new session shows every specimen.
const VIEW_KEYS = ["by", "k", "specimens", "landmarks", "units", "dentin",
  "molarDensity", "densityStat", "volumeMeasure", "volumeTooth", "thicknessMeasure", "thicknessTooth",
  "patchMeasure", "patchTooth"];
const viewKey = () => `studio.results:${state.app?.workspace?.root || ""}`;
function loadView() {
  try { return JSON.parse(localStorage.getItem(viewKey()) || "{}") || {}; } catch { return {}; }
}
function saveView(ui) {
  try { localStorage.setItem(viewKey(), JSON.stringify(Object.fromEntries(VIEW_KEYS.map((k) => [k, ui[k]])))); } catch { /* private window */ }
}
const scopeKey = () => `studio.results.scope:${state.app?.workspace?.root || ""}`;
function loadScope() {
  try { return JSON.parse(sessionStorage.getItem(scopeKey()) || "{}") || {}; } catch { return {}; }
}
function saveScope(ui) {
  try { sessionStorage.setItem(scopeKey(), JSON.stringify({ hidden: [...ui.hidden], left: [...ui.left] })); } catch { /* private window */ }
}
// The project last opened, marked on the projects page; and whether the filters are open.
const lastKey = () => `studio.results.last:${state.app?.workspace?.root || ""}`;
function lastOpened() {
  try { return localStorage.getItem(lastKey()); } catch { return null; }
}
function rememberOpened(id) {
  try { localStorage.setItem(lastKey(), id); } catch { /* private window */ }
}
const FILTERS_KEY = "studio.results.filtersOpen";
function filtersOpen() {
  try { return localStorage.getItem(FILTERS_KEY) === "1"; } catch { return false; }
}
const projects = () => state.app?.workspace?.settings?.results_projects || [];
const ALL = "all";
const resultsHref = (id) => `#/results/${encodeURIComponent(id)}`;
const plural = (n, one) => `${n} ${one}${n === 1 ? "" : "s"}`;
const sameSet = (list, set) => (list || []).length === set.size && (list || []).every((v) => set.has(v));

// #/results is the projects page; #/results/<id> a project's results, and
// #/results/all every specimen's. `query.s`, a comma-separated list of names
// (the Library's "Show in Results"), shows those specimens alone among every
// specimen's.
export async function mount(root, [scope] = [], query = {}) {
  const id = scope || query.project || (query.s != null ? ALL : null);
  if (!id) return projectsPage(root);
  return resultsPage(root, id, query);
}

async function resultsPage(root, scopeId, query) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  const saved = loadView();
  const scope = loadScope();
  const ui = {
    // In a project (`project`, its id), `members` are its specimens. The filters:
    // `hidden`, values hidden by their chips; `left`, specimens unticked for now.
    hidden: new Set(scope.hidden || []), left: new Set(scope.left || []),
    project: scopeId === ALL ? null : scopeId, members: new Set(), filtersOpen: filtersOpen(),
    specimens: true, k: 21, landmarks: true,
    molarDensity: null, densityStat: "mean", volumeMeasure: "share", volumeTooth: "all",
    thicknessMeasure: "thickness", thicknessTooth: "all", patchMeasure: "thinShare", patchTooth: "all",
    by: null, units: null, dentin: true, moreTests: false,
    ...Object.fromEntries(VIEW_KEYS.filter((k) => saved[k] !== undefined).map((k) => [k, saved[k]])),
  };
  // A remembered switch the page no longer offers falls back to the default.
  if (!DENSITY_UNITS.some(([k]) => k === ui.molarDensity)) ui.molarDensity = null;
  if (!DENSITY_STATS.includes(ui.densityStat)) ui.densityStat = "mean";
  if (!VOLUME_MEASURES[ui.volumeMeasure]) ui.volumeMeasure = "share";
  if (!THICKNESS_MEASURES[ui.thicknessMeasure]) ui.thicknessMeasure = "thickness";
  if (!PATCH_MEASURES[ui.patchMeasure]) ui.patchMeasure = "thinShare";
  for (const key of ["volumeTooth", "thicknessTooth", "patchTooth"]) if (!TEETH.some(([k]) => k === ui[key])) ui[key] = "all";
  let data = null, order = {}, metaOf = {}, flagOf = {}, folderOf = {};
  let only = query.s != null && !ui.project ? String(query.s).split(",").filter(Boolean) : null;
  if (ui.project && !projects().some((p) => p.id === ui.project)) {
    page.appendChild(h("div", { class: "empty" }, h("h3", null, "There is no such project"),
      h("span", null, "It may have been removed, or the link is from another workspace."),
      h("a", { class: "btn primary", href: "#/results" }, icon("back"), "Every project")));
    return;
  }
  rememberOpened(ui.project || ALL);
  if (ui.project) applyProject(projects().find((p) => p.id === ui.project));
  // Smoothed profiles, by quantity, smoothing and specimen: redrawing after a
  // chip or a switch then costs no smoothing at all.
  const curveCache = new Map();

  // What the page can export, gathered as it renders: each card's figure, and
  // every measurement under every switch for the Prism tables.
  let figures = [], tables = [];
  const sub = h("div", { class: "sub" });
  // The CSVs, of the specimens the page shows once any are hidden (linkCsvs).
  const csvs = [["metadata", "Metadata"], ["specimens", "R specimen list"], ["molar", "Molar enamel"], ["incisor", "Incisor profiles"],
    ["landmarks", "Landmarks"], ["segmentation", "Segmentation"]].map(([kind, label]) => ({ kind, label,
    link: h("a", { class: "btn small", href: `/api/export/${kind}`, download: "", title: `Download ${label.toLowerCase()} as CSV` }, icon("download"), label) }));
  const exportsRow = h("div", { class: "actions" },
    h("button", { class: "btn small", title: "Pick the colour of every group, landmark, line, axis and text", onclick: () => openColors() },
      icon("palette"), "Colours"),
    h("button", { class: "btn small primary", title: "Every chart as a Prism-style figure and every measurement as Prism tables",
      onclick: () => exportAll({ figures, data: tables, notes: exportNotes, name: exportName() }) }, icon("download"), "Export for Prism"),
    csvs.map((c) => c.link));
  const title = h("h1");
  const projectRow = h("div", { class: "project-row" });
  page.appendChild(h("div", { class: "page-head" },
    h("div", null, h("nav", { class: "folder-crumbs", "aria-label": "Results" }, h("a", { href: "#/results" }, "Results"),
      h("span", { class: "sep", "aria-hidden": "true" }, icon("chevron"))), title, sub, projectRow),
    exportsRow));

  const filters = h("div", { class: "compare-bar filters-panel" });
  page.appendChild(filters);
  const content = h("div");
  page.appendChild(content);

  const load = async () => {
    content.classList.add("refetching");
    try {
      data = await api.get("/api/results");
      curveCache.clear();
      order = levelOrder(data.samples);
      metaOf = Object.fromEntries(data.samples.map((s) => [s.name, s.meta || {}]));
      flagOf = Object.fromEntries(data.samples.map((s) => [s.name, s.excluded || null]));
      folderOf = Object.fromEntries(data.samples.map((s) => [s.name, s.folder || ""]));
      if (only) {
        // Brought from the Library: those specimens, whatever was hidden before.
        const keep = new Set(only);
        ui.left = new Set(data.samples.map((s) => s.name).filter((n) => !keep.has(n)));
        ui.hidden.clear();
        only = null;
        keepScope();
      }
      if (query.s != null || query.project) history.replaceState(history.state, "", resultsHref(ui.project || ALL));
      ui.by = validBy(ui.by);
      render();
    } catch (err) { errorToast(err); }
    content.classList.remove("refetching");
  };
  // A project keeps its own comparison and hidden values; the page's own stay as they were for "All specimens".
  const remember = () => {
    if (!ui.project) return saveView(ui);
    saveView({ ...ui, by: loadView().by });
    saveProject({ by: ui.by });
  };
  const keepScope = () => (ui.project ? saveProject({ hidden: [...ui.hidden], left: [...ui.left] }) : saveScope(ui));

  // ------------------------------------------------------------ comparison
  const validBy = (by) => validByFor(by, scoped());

  function setBy(i, key) {
    const by = [...ui.by];
    if (!key) ui.by = by.slice(0, i);
    else {
      const j = by.indexOf(key);
      if (j >= 0 && j !== i) by[j] = by[i];  // a factor picked twice swaps places
      by[i] = key;
      ui.by = by.filter(Boolean);
    }
    remember();
    render();
  }

  const mainKey = () => ui.by[0];
  // The specimens among `names` that an analysis keeps: its flagged ones are left out.
  const included = (names, analysis) => names.filter((n) => !excludedFrom(flagOf[n], analysis));
  // Flagged out of everything: no chip, count or warning counts them.
  const inAnyAnalysis = (s) => flagOf[s.name]?.from !== "all";
  const hiddenKey = (key, level) => `${key}\u0001${level}`;
  const hiddenBy = (name) => FACTORS.find((f) => ui.hidden.has(hiddenKey(f.key, levelOf(metaOf[name], f.key))));
  // In the project open (every specimen, outside one), and of those, not unticked for now.
  const inProject = (name) => !ui.project || ui.members.has(name);
  const scoped = () => (data ? data.samples.filter((s) => inProject(s.name)) : []);
  const inScope = (name) => inProject(name) && !ui.left.has(name);
  const isShown = (name) => inScope(name) && !hiddenBy(name);
  // The shown specimens among `names`, as compare.js takes them.
  const pool = (names) => {
    const want = new Set(names);
    return data.samples.filter((s) => want.has(s.name) && isShown(s.name)).map((s) => ({ name: s.name, meta: s.meta || {} }));
  };
  // Colour follows the main comparison's level over every specimen, so hiding one never recolours the rest;
  // a colour picked for the level wins.
  const paletteSlot = (level) => {
    const i = (order[mainKey()] || []).filter(Boolean).indexOf(level);
    return level && i >= 0 ? i : null;
  };
  const colorOf = (level) => (level && groupColor(mainKey(), level)) || slotColor(paletteSlot(level));

  // ------------------------------------------------------------ colours and titles
  // The compared detail's levels, for the colour editor.
  const editorGroups = () => (order[mainKey()] || []).filter(Boolean)
    .map((lv) => ({ id: groupKey(mainKey(), lv), label: levelName(mainKey(), lv), fallback: slotColor(paletteSlot(lv)) }));
  const editor = (onChange) => colorEditor({ groups: editorGroups(), heading: factor(mainKey())?.label || "Groups", onChange });
  function openColors() {
    modal({ title: "Chart colours", wide: true, body: editor(), actions: [{ label: "Done", primary: true }] });
  }
  // A card's title as the reader named it, and how to rename it: its figure takes the same.
  const titled = (key, fallback) => ({ title: titleOf(key) || fallback, rename: { fallback, set: (text) => setStyle("titles", key, text) } });
  const cellLabel = (cell, st) => (st.label ? `${cell.name}, ${st.label}` : cell.name);

  // ------------------------------------------------------------ the project
  const current = () => (ui.project ? projects().find((p) => p.id === ui.project) || null : null);

  function applyProject(p) {
    ui.members = new Set(p.names || []);
    ui.hidden = new Set(p.hidden || []);
    ui.left = new Set(p.left || []);
    ui.by = data ? validBy(p.by) : p.by;
  }

  // Changes to the project go to the server in one request once they settle.
  // The local copy follows at once, so the server's echo changes nothing.
  let pending = null, pendingTimer = null;
  function saveProject(fields) {
    const p = current();
    if (!p) return;
    const changed = Object.fromEntries(Object.entries(fields).filter(([k, v]) => JSON.stringify(p[k] || []) !== JSON.stringify(v)));
    if (!Object.keys(changed).length) return;
    Object.assign(p, changed);
    pending = { id: p.id, fields: { ...(pending?.fields || {}), ...changed } };
    clearTimeout(pendingTimer);
    pendingTimer = setTimeout(flushProject, 400);
  }
  function flushProject() {
    clearTimeout(pendingTimer);
    const sending = pending;
    pending = null;
    if (sending) api.patch(`/api/results/projects/${sending.id}`, sending.fields).catch(errorToast);
  }

  // Another browser's edits, a specimen renamed, the project removed: the page follows.
  function settingsChanged() {
    if (!data || pending || !ui.project) return;
    const p = current();
    if (!p) { toast("This project was removed."); go("#/results"); return; }
    if (!sameSet(p.names, ui.members) || !sameSet(p.hidden, ui.hidden) || !sameSet(p.left, ui.left)
      || JSON.stringify(validBy(p.by)) !== JSON.stringify(ui.by)) {
      applyProject(p);
      render();
      return;
    }
    renderHead();
  }

  // The project's name and what can be done with it; outside one, keeping the specimens shown as one.
  function renderHead() {
    const p = current();
    title.textContent = p ? p.name : "All specimens";
    document.title = `${p ? p.name : "All specimens"} · Results`;
    replace(projectRow, p ? [
      h("button", { class: "btn small ghost", title: "Add specimens to the project or take them out", onclick: () => editMembers(p) },
        icon("edit"), "Edit specimens"),
      h("button", { class: "btn small ghost", onclick: () => renameProjectDialog(p, renderHead) }, "Rename"),
      h("button", { class: "btn small ghost", onclick: () => newProjectDialog({ everyone: data.samples, copyOf: p,
        onMade: (made) => go(resultsHref(made.id)) }) }, "Duplicate"),
      h("button", { class: "btn small ghost danger", onclick: async () => {
        pending = null;
        clearTimeout(pendingTimer);
        if (await removeProjectConfirm(p)) go("#/results");
      } }, icon("trash"), "Remove"),
    ] : h("button", { class: "btn small ghost", title: "Keep the specimens shown, and how they are compared, as a project",
      onclick: () => newProjectDialog({ everyone: data.samples, shownNow: data.samples.filter((s) => isShown(s.name)).map((s) => s.name),
        by: ui.by, onMade: (made) => go(resultsHref(made.id)) }) }, icon("plus"), "Save as a project"));
  }

  // What the project holds: any specimen of the workspace, ticked in or out.
  async function editMembers(p) {
    flushProject();
    const chosen = await specimenChecklist({
      specimens: data.samples, ticked: ui.members, by: ui.by, order, colorOf,
      title: `Specimens in ${p.name}`, confirm: "Save",
      intro: "The project holds the specimens ticked, for every browser that opens the workspace. A specimen added to the library "
        + "later joins it once ticked here. To leave some out of the charts for now, untick them in the filters instead.",
    });
    if (!chosen) return;
    ui.members = chosen;
    for (const n of [...ui.left]) if (!chosen.has(n)) ui.left.delete(n);
    saveProject({ names: [...chosen], left: [...ui.left] });
    ui.by = validBy(ui.by);
    render();
  }

  // The filter: which of the project's specimens the page shows for now.
  async function openPicker() {
    const listed = scoped().filter(inAnyAnalysis);
    const chosen = await specimenChecklist({
      specimens: listed, ticked: new Set(listed.map((s) => s.name).filter((n) => !ui.left.has(n))), by: ui.by, order, colorOf,
      note: (n) => { const by = hiddenBy(n); return by ? `hidden by the ${levelName(by.key, levelOf(metaOf[n], by.key))} chip` : null; },
      title: "Specimens to show", confirm: "Show these",
      intro: `The charts, statistics and exports take the specimens ticked${ui.project ? "; the rest stay in the project" : ""}. `
        + "Flagged specimens stay out of the analyses their flags name.",
    });
    if (!chosen) return;
    ui.left = new Set(listed.map((s) => s.name).filter((n) => !chosen.has(n)));
    keepScope();
    render();
  }

  function renderFilters() {
    const mine = scoped();
    const counted = mine.filter(inAnyAnalysis);
    const have = factorsWithValues(mine);
    const pick = (i) => {
      const current = ui.by[i];
      const taken = new Set(ui.by.filter((_, j) => j < i));
      const options = FACTORS.filter((f) => (have.includes(f.key) || f.key === current) && (!taken.has(f.key) || f.key === current));
      return h("select", { "aria-label": i === 0 ? "Main comparison" : `Split ${i}`, onchange: (e) => setBy(i, e.target.value) },
        i ? h("option", { value: "", selected: !current }, "Nothing") : null,
        options.map((f) => h("option", { value: f.key, selected: f.key === current }, f.label)));
    };
    // A split's picker appears once the one before it is set.
    const splits = [];
    for (let i = 1; i <= Math.min(ui.by.length, MAX_SPLITS); i++) {
      splits.push(h("span", { class: "muted" }, i === 1 ? "split by" : "and"), pick(i));
    }

    // Chips for every detail with more than one value among the project's
    // specimens, the compared ones first; a value hidden stays offered.
    const levelsOf = (k) => {
      const present = new Set(counted.map((s) => levelOf(s.meta, k)));
      return (order[k] || []).filter((lv) => present.has(lv) || ui.hidden.has(hiddenKey(k, lv)));
    };
    const chipKeys = [...ui.by, ...FACTORS.map((f) => f.key).filter((k) => !ui.by.includes(k))].filter((k) => levelsOf(k).length > 1);
    const counts = {};
    for (const s of counted.filter((x) => !ui.left.has(x.name))) for (const k of chipKeys) {
      const id = hiddenKey(k, levelOf(s.meta, k));
      counts[id] = (counts[id] || 0) + 1;
    }
    const chipRows = chipKeys.map((k) => h("div", { class: "level-group", role: "group", "aria-label": factor(k).label },
      h("span", { class: "muted" }, factor(k).label),
      levelsOf(k).map((lv) => {
        const id = hiddenKey(k, lv);
        return h("button", {
          class: "chip", "aria-pressed": ui.hidden.has(id) ? "false" : "true",
          onclick: () => { ui.hidden.has(id) ? ui.hidden.delete(id) : ui.hidden.add(id); keepScope(); render(); },
        }, k === mainKey() ? h("i", { class: "sw", style: { background: colorOf(lv) } }) : null, `${levelName(k, lv)} (${counts[id] || 0})`);
      })));

    // Pooling ages is the one grouping that misleads: say so, with the fix beside it, filters open or not.
    let note = null;
    if (!ui.by.includes("age")) {
      const ages = [...new Set(counted.filter((s) => isShown(s.name)).map((s) => levelOf(s.meta, "age")).filter(Boolean))];
      if (ages.length > 1) {
        const ordered = (order.age || []).filter((a) => ages.includes(a));
        const canSplit = ui.by.length <= MAX_SPLITS;
        note = h("div", { class: "notice warning compare-note" }, icon("alert"),
          h("span", null, `Ages ${ordered.slice(0, -1).join(", ")} and ${ordered[ordered.length - 1]} are pooled in every group below. `
            + "Age changes the whole trajectory, so compare within one age."),
          canSplit ? h("button", { class: "btn small", onclick: () => setBy(ui.by.length, "age") }, "Split by age too") : null);
      }
    }

    // Which of the project's specimens: all of them, or the ones ticked.
    const ticked = counted.filter((s) => !ui.left.has(s.name)).length;
    const some = ticked < counted.length;
    const hiddenNames = chipKeys.flatMap((k) => levelsOf(k).filter((lv) => ui.hidden.has(hiddenKey(k, lv))).map((lv) => levelName(k, lv)));
    const filtering = some || hiddenNames.length > 0;
    const clearFilters = () => { ui.left.clear(); ui.hidden.clear(); keepScope(); render(); };
    const specimensButton = [
      h("span", { class: "muted" }, "in"),
      h("button", { class: "chip", "aria-pressed": some ? "true" : "false", "aria-haspopup": "dialog",
        title: ui.project ? "Choose which of the project's specimens the page shows" : "Choose which specimens the page shows", onclick: openPicker },
      some ? `${ticked} of ${counted.length} specimens` : `all ${counted.length} specimens`, icon("edit")),
      some ? h("button", { class: "btn small ghost", onclick: () => { ui.left.clear(); keepScope(); render(); } }, "Tick all") : null,
    ];

    // The summary, always there: what is compared, which specimens, what is hidden.
    const lower = (k) => factor(k)?.label.toLowerCase() || k;
    const [main, ...rest] = ui.by;
    const summary = [
      `${factor(main)?.label || "–"}${rest.length ? `, split by ${rest.map(lower).join(" and ")}` : ""}`,
      some ? `${ticked} of ${counted.length} specimens` : `all ${counted.length} specimens`,
      hiddenNames.length ? `${hiddenNames.join(", ")} hidden` : null,
    ].filter(Boolean);
    const toggle = () => {
      ui.filtersOpen = !ui.filtersOpen;
      try { localStorage.setItem(FILTERS_KEY, ui.filtersOpen ? "1" : "0"); } catch { /* private window */ }
      renderFilters();
    };
    filters.classList.toggle("open", ui.filtersOpen);
    replace(filters,
      h("div", { class: "filters-head" },
        h("button", { class: "filters-toggle", "aria-expanded": ui.filtersOpen ? "true" : "false", "aria-controls": "results-filters", onclick: toggle },
          icon("chevron"), "Filters", filtering ? h("span", { class: "tag accent" }, "On") : null),
        h("span", { class: "filters-summary", onclick: toggle }, summary.map((s, i) => [i ? h("span", { class: "dot", "aria-hidden": "true" }, "·") : null, s])),
        h("span", { class: "spacer" }),
        filtering ? h("button", { class: "btn small ghost", onclick: clearFilters, title: "Show every specimen of the project again" }, "Clear filters") : null),
      h("div", { class: "filters-body", id: "results-filters", hidden: !ui.filtersOpen },
        h("div", { class: "compare-row" },
          h("span", { class: "muted" }, "Compare"), pick(0), splits, specimensButton,
          h("span", { class: "spacer" }),
          h("label", { class: "check" }, h("input", { type: "checkbox", checked: ui.specimens, onchange: (e) => { ui.specimens = e.target.checked; remember(); render(); } }), "Individual specimens"),
          h("label", { class: "check" }, h("input", { type: "checkbox", checked: ui.landmarks, onchange: (e) => { ui.landmarks = e.target.checked; remember(); render(); } }), "Landmarks"),
          h("label", { class: "row" }, h("span", { class: "muted" }, "Smoothing"),
            h("select", { onchange: (e) => { ui.k = Number(e.target.value); remember(); render(); } },
              [[1, "None"], [11, "11 slices"], [21, "21 slices"], [51, "51 slices"]].map(([v, l]) => h("option", { value: v, selected: ui.k === v }, l))))),
        chipRows.length ? h("div", { class: "level-row" }, chipRows) : null,
        flaggedPointer()),
      note);
  }

  // ------------------------------------------------------------ helpers
  const openSample = (n) => () => go(`#/sample/${encodeURIComponent(n)}`);

  // A strip plot of one measurement: a column per cell, clustered by stratum.
  function strip(valueOf, names, { yLabel, fy, zero = false }) {
    const groups = [];
    for (const st of arrange(pool(names), ui.by, order)) {
      for (const cell of st.cells) {
        const values = cell.members.map((n) => ({ label: n, value: valueOf(n), onClick: openSample(n) })).filter((v) => finite(v.value));
        if (values.length) groups.push({ name: cell.name, color: colorOf(cell.level), cluster: st.label, values });
      }
    }
    if (!groups.length) return null;
    return { chart: stripPlot({ groups, yLabel, fy, zero, height: 270 }), groups };
  }

  // ------------------------------------------------------------ exports
  // A measurement as the figure and the Prism tables take it: the specimens
  // shown, in the page's groups, order and colours, with the Welch tests of
  // the main comparison within each stratum. `scale` multiplies every value
  // (a share as a percentage).
  function columnSpec(names, { title, yLabel, valueOf, scale = 1 }) {
    const specimens = pool(names);
    const value = (n) => { const v = valueOf(n); return finite(v) ? v * scale : null; };
    const strata = arrange(specimens, ui.by, order).map((st) => ({
      label: st.label,
      cells: st.cells.map((cell) => ({ level: cell.level, name: cell.name, color: colorOf(cell.level),
        values: cell.members.map((n) => ({ label: n, value: value(n) })).filter((v) => v.value != null) }))
        .filter((c) => c.values.length),
    })).filter((st) => st.cells.length);
    const present = new Set(strata.flatMap((st) => st.cells.map((c) => c.level)));
    return {
      kind: "column", title, yLabel, splits: ui.by.slice(1).map((k) => factor(k)?.label),
      levels: (order[mainKey()] || []).filter((lv) => present.has(lv)).map((lv) => ({ level: lv, name: levelName(mainKey(), lv), color: colorOf(lv) })),
      strata,
      tests: comparisons(specimens, ui.by, order, value).filter((t) => t.factor === mainKey() && !t.pooled),
    };
  }

  // A figure as the reader titled and coloured it: the card's title above it
  // (and naming its files once renamed), axis titles renamed, and the marks'
  // own colours. `defaults` keeps what it had before, for the export dialog.
  function styled(key, cardTitle, spec) {
    if (!spec) return spec;
    const custom = titleOf(key);
    return {
      ...spec, title: custom || spec.title, heading: custom || cardTitle,
      yLabel: labelOf(spec.yLabel), xLabel: labelOf(spec.xLabel), ink: figureInk(),
      defaults: { heading: cardTitle, yLabel: spec.yLabel, xLabel: spec.xLabel },
    };
  }

  // A card's figure, offered to the export dialogs; returns the card's Export handler.
  function offerFigure(key, cardTitle, spec) {
    const entry = {
      key, spec: () => styled(key, cardTitle, spec()),
      edit: {
        title: (text) => setStyle("titles", key, text),
        label: (from, text) => { if (from) setStyle("labels", from, text); },
        colors: editor,
      },
    };
    figures.push(entry);
    return () => exportFigure(entry, { notes: exportNotes });
  }
  // A measurement for the Prism tables, whether or not a card shows it now.
  const offerTable = (spec) => tables.push({ spec });
  const offerColumn = (names, measure) => offerTable(() => columnSpec(names, measure));

  // The export's files carry the project's name.
  const exportName = () => {
    const slug = (current()?.name || "").replace(/[^A-Za-z0-9._+-]+/g, "_").replace(/^_+|_+$/g, "");
    return slug ? `microct_results_${slug}` : "microct_results";
  };

  // What a Prism file's info sheet, and the export's README, say about it.
  function exportNotes() {
    const [main, ...splits] = ui.by;
    const lower = (k) => factor(k)?.label.toLowerCase() || k;
    const hidden = [...ui.hidden].map((id) => { const [k, lv] = id.split("\u0001"); return `${levelName(k, lv)} (${lower(k)})`; });
    const flagged = scoped().filter((s) => flagOf[s.name]).map((s) => s.name);
    const project = current();
    const unticked = scoped().filter((s) => ui.left.has(s.name) && inAnyAnalysis(s)).map((s) => s.name);
    return [
      `Exported from MicroCT Studio's Results page on ${new Date().toLocaleString()}.`,
      project ? `Project: ${project.name}, ${plural(data.samples.filter((s) => ui.members.has(s.name)).length, "specimen")}.` : null,
      `Compared by ${lower(main)}${splits.length ? `, split by ${splits.map(lower).join(" and ")}` : ""}.`,
      unticked.length ? `Specimens left out on the page and here: ${unticked.join(", ")}.` : null,
      hidden.length ? `Hidden on the page and left out here: ${hidden.join(", ")}.` : null,
      flagged.length ? `Flagged, and left out of the analyses their flags name: ${flagged.join(", ")}.` : null,
      "Every value is one specimen's own.",
      `Incisor profiles are each specimen's cleaned profile${ui.k > 1 ? ` (${ui.k}-slice running median)` : ""} read at every 1% along the tooth; `
        + "a group's enamel values start where its specimens measure half their slices, as the page draws them.",
      "Significance, on the page and in the figures: Welch t-tests between the groups within each stratum, uncorrected for multiple tests "
        + "(* p < 0.05, ** p < 0.01, *** p < 0.001, **** p < 0.0001).",
    ].filter(Boolean);
  }

  // The per-specimen table behind a chart: its details as columns, in the chart's order.
  function specimenTable(names, columns, rowOf) {
    const rows = [];
    for (const st of arrange(pool(names), ui.by, order)) {
      for (const cell of st.cells) {
        for (const n of cell.members) {
          rows.push({ name: n, ...Object.fromEntries(ui.by.map((k, i) => [`f${i}`, levelName(k, levelOf(metaOf[n], k))])), ...rowOf(n) });
        }
      }
    }
    return dataTable([{ key: "name", label: "Specimen" }, ...ui.by.map((k, i) => ({ key: `f${i}`, label: factor(k).label })), ...columns], rows);
  }

  function statsBlock(valueOf, names, fy) {
    const specimens = pool(names);
    const [, ...splits] = ui.by;
    const rows = [];
    for (const st of arrange(specimens, ui.by, order)) {
      for (const cell of st.cells) {
        const values = cell.members.map(valueOf).filter(finite);
        if (values.length) rows.push({ st, cell, d: describe(values) });
      }
    }
    if (!rows.length) return null;
    const tests = comparisons(specimens, ui.by, order, valueOf);
    const main = tests.filter((t) => t.factor === mainKey());
    const others = tests.filter((t) => t.factor !== mainKey());
    const testTable = (list) => {
      const within = list.some((t) => t.within);
      return h("div", { class: "table-wrap" }, h("table", { class: "data tests" },
        h("thead", null, h("tr", null, h("th", null, "Welch t-test"), within ? h("th", null, "Within") : null,
          h("th", { class: "n" }, "n"), h("th", { class: "n" }, "t"), h("th", { class: "n" }, "df"), h("th", { class: "n" }, "p"),
          h("th", { class: "stars", title: "* p < 0.05, ** p < 0.01, *** p < 0.001, **** p < 0.0001" }, h("span", { class: "sr-only" }, "Significance")))),
        h("tbody", null, list.map((t, i) => h("tr", { class: i && t.factor !== list[i - 1].factor ? "first" : "" },
          h("td", null, `${t.b} vs ${t.a}`), within ? h("td", null, t.within || "–") : null,
          h("td", { class: "n" }, `${t.nb} vs ${t.na}`),
          h("td", { class: "n" }, t.t != null ? t.t.toFixed(2) : "–"), h("td", { class: "n" }, t.df != null ? t.df.toFixed(1) : "–"),
          h("td", { class: "n" }, fmtP(t.p)),
          h("td", { class: `stars ${t.p != null && t.p < 0.05 ? "sig" : ""}` }, stars(t.p)))))));
    };
    const otherNames = [...new Set(others.map((t) => factor(t.factor)?.label.toLowerCase()))];
    return h("div", { class: "stats-table" },
      h("div", { class: "table-wrap" }, h("table", { class: "data" },
        h("thead", null, h("tr", null, h("th", null, factor(mainKey())?.label || "Group"), splits.map((k) => h("th", null, factor(k)?.label)),
          h("th", { class: "n" }, "n"), h("th", { class: "n" }, "Mean"), h("th", { class: "n" }, "SD"))),
        h("tbody", null, rows.map(({ st, cell, d }) => h("tr", null,
          h("td", null, h("span", { class: "group-label" }, h("i", { class: "sw", style: { background: colorOf(cell.level) } }), cell.name)),
          splits.map((k) => h("td", null, levelName(k, levelOf(st.meta, k)))),
          h("td", { class: "n" }, d.n), h("td", { class: "n" }, fy(d.mean)), h("td", { class: "n" }, d.sd != null ? fy(d.sd) : "–")))))),
      main.length ? testTable(main) : null,
      others.length ? h("details", { class: "more-tests", open: ui.moreTests, ontoggle: (e) => { ui.moreTests = e.target.open; } },
        h("summary", null, `By ${otherNames.join(" and ")} (${others.length} test${others.length === 1 ? "" : "s"})`), testTable(others)) : null,
      tests.length ? h("div", { class: "muted", style: { marginTop: "6px" } },
        "Exploratory: no correction for multiple tests. A specimen whose detail is not set is shown, never tested.") : null);
  }

  // ------------------------------------------------------------ molar
  // A row of buttons, one pressed: [value, label, disabled?].
  function switcher(label, options, current, pick) {
    return h("div", { class: "seg", role: "group", "aria-label": label },
      options.map(([value, text, disabled]) => h("button", {
        "aria-pressed": current === value ? "true" : "false", disabled: Boolean(disabled),
        onclick: () => { pick(value); remember(); render(); },
      }, text)));
  }

  // One measurement as a card: the strip plot, its statistics beside it, a table behind it.
  function measureCard({ title, sub, controls, valueOf, names, yLabel, fy, columns, rowOf, empty, extra, exportKey, exportAs }) {
    const s = strip(valueOf, names, { yLabel, fy });
    return chartCard({
      ...titled(exportKey, title), sub, controls,
      onExport: s && exportAs ? offerFigure(exportKey, title, () => columnSpec(names, exportAs())) : null,
      chart: s ? s.chart : h("div", { class: "empty" }, h("span", null, empty)),
      table: s ? () => specimenTable(names, columns, rowOf) : null,
      aside: s ? statsBlock(valueOf, names, fy) : null,
      extra,
    });
  }

  function molarSection() {
    const names = included(Object.keys(data.molar).filter(inProject), "molar");
    if (!names.length) return emptySection("Molar enamel", Object.keys(data.molar).some(inProject)
      ? "Every specimen with molar results is flagged and left out." : "Run Molar enamel on segmented specimens to compare them here.");
    const m = (n) => data.molar[n];
    const tooth = (n, which) => m(n).teeth.find((t) => t.name === which);
    const shown = pool(names).map((s) => s.name);
    // How each specimen's enamel was found: between its halfway surfaces, or by a threshold.
    const thresholds = new Set(names.map((n) => (m(n).enamel_mask === "local-halfway" ? "local" : m(n).threshold_method)));
    const fmt = (fy) => (v) => (v == null || !Number.isFinite(v) ? "–" : fy(v));
    const noteLine = (text) => h("div", { class: "muted", style: { fontSize: "12.5px", marginTop: "8px" } }, text);

    // ---- Enamel density: converted, as stored, or against the specimen's own dentin.
    const calibrated = shown.filter((n) => m(n).enamel_mean_mgHA != null);
    const units = ui.molarDensity || (calibrated.length ? "mgHA" : "raw");
    const pcts = new Set(shown.map((n) => (m(n).density_percentiles || []).join("/")).filter(Boolean));
    const uniform = pcts.size === 1 ? [...pcts][0].split("/").map(Number) : null;
    const statLabel = { mean: "Mean", low: uniform ? `${ordinal(uniform[0])} percentile` : "Lower percentile",
      high: uniform ? `${ordinal(uniform[1])} percentile` : "Upper percentile" };
    const stat = ui.densityStat;
    // The dentin peak in the card's units, for the table: the ratio's denominator.
    const dentinPeak = (n, u) => {
      const raw = m(n).dentin_mode;
      if (raw == null) return null;
      if (u === "raw") return raw;
      const cal = m(n).calibration;
      return cal && cal.divisor ? toMgHA(raw, cal) : null;
    };
    let density;
    if (units === "ratio") {
      density = {
        sub: "median enamel over the dentin peak, both from the specimen's own scan, so no calibration is needed",
        valueOf: (n) => m(n).density_ratio, yLabel: "Enamel / dentin", fy: sig3,
        columns: [{ key: "ratio", label: "Enamel/dentin", numeric: true, format: fmt(sig3) },
          { key: "enamel", label: "Enamel median (raw)", numeric: true, format: fmt(dens) },
          { key: "dentin", label: "Dentin peak (raw)", numeric: true, format: fmt(dens) }],
        rowOf: (n) => ({ ratio: m(n).density_ratio, enamel: m(n).enamel_median, dentin: m(n).dentin_mode }),
        empty: "Run Molar enamel again to add the enamel/dentin ratio.",
      };
    } else {
      const unitName = units === "mgHA" ? "mg HA/cm³" : "raw";
      density = {
        sub: units === "mgHA"
          ? `${statLabel[stat].toLowerCase()} of each specimen's enamel voxels, converted with its scan's HA calibration`
          : `${statLabel[stat].toLowerCase()} of each specimen's enamel voxels, as stored in the scan`,
        valueOf: (n) => m(n)[`enamel_${stat}_${units}`],
        yLabel: units === "mgHA" ? `Enamel density, ${statLabel[stat].toLowerCase()} (mg HA/cm³)` : `Enamel intensity, ${statLabel[stat].toLowerCase()} (raw)`,
        fy: dens4,
        columns: [...DENSITY_STATS.map((k) => ({ key: k, label: `${statLabel[k]} (${unitName})`, numeric: true, format: fmt(dens) })),
          { key: "dentin", label: `Dentin peak (${unitName})`, numeric: true, format: fmt(dens) },
          { key: "ratio", label: "Enamel/dentin", numeric: true, format: fmt(sig3) }],
        rowOf: (n) => ({ ...Object.fromEntries(DENSITY_STATS.map((k) => [k, m(n)[`enamel_${k}_${units}`]])),
          dentin: dentinPeak(n, units), ratio: m(n).density_ratio }),
        empty: units === "mgHA"
          ? "No HA calibration for these scans. Molar enamel reads the one a Scanco scanner writes into the ISQ, or takes one in its Density calibration settings. Raw intensity shows every specimen."
          : "Run Molar enamel again to add the enamel intensity.",
      };
    }
    const uncalibrated = units === "mgHA" ? shown.filter((n) => m(n).enamel_mean_mgHA == null) : [];
    const densityNotes = [];
    if (uncalibrated.length && calibrated.length) densityNotes.push(`Not shown, with no HA calibration: ${uncalibrated.join(", ")}. Raw intensity and the enamel/dentin ratio include them.`);
    if (units === "raw") densityNotes.push("Raw values are comparable only between scans made with one scanner setting and calibration.");
    const densityCard = measureCard({
      title: "Enamel density", sub: density.sub, names,
      controls: [
        switcher("Density units", DENSITY_UNITS, units, (v) => { ui.molarDensity = v; }),
        units === "ratio" ? null : switcher("Statistic", DENSITY_STATS.map((k) => [k, statLabel[k]]), stat, (v) => { ui.densityStat = v; }),
      ],
      valueOf: density.valueOf, yLabel: density.yLabel, fy: density.fy,
      columns: density.columns, rowOf: density.rowOf, empty: density.empty,
      extra: densityNotes.length ? noteLine(densityNotes.join(" ")) : null,
      exportKey: "molar-density", exportAs: () => densityMeasure(units, stat),
    });

    // ---- Volume: the enamel, the dentin and the whole tooth, for the row or one molar.
    const hasTooth = (which) => shown.some((n) => tooth(n, which));
    const vTooth = ui.volumeTooth === "all" || hasTooth(ui.volumeTooth) ? ui.volumeTooth : "all";
    const volumes = (n, which) => {
      if (which === "all") {
        return { share: m(n).enamel_fraction, enamel: m(n).enamel_mm3, dentin: m(n).dentin_mm3, total: m(n).total_mm3 };
      }
      const t = tooth(n, which);
      if (!t) return {};
      return { share: t.enamel_fraction, enamel: t.enamel_mm3, total: t.volume_mm3,
        dentin: t.volume_mm3 != null && t.enamel_mm3 != null ? t.volume_mm3 - t.enamel_mm3 : null };
    };
    const vm = VOLUME_MEASURES[ui.volumeMeasure];
    const where = vTooth === "all" ? "the three molars together" : `${vTooth} alone`;
    const volumeCard = measureCard({
      title: "Volume",
      sub: ui.volumeMeasure === "share" ? `enamel volume over the tooth's volume, ${where}`
        : ui.volumeMeasure === "total" ? `enamel and dentin, ${where}` : `${where}`,
      names,
      controls: [
        switcher("Volume measured", Object.entries(VOLUME_MEASURES).map(([k, d]) => [k, vTooth === "all" && k === "total" ? "Whole row" : d.label]),
          ui.volumeMeasure, (v) => { ui.volumeMeasure = v; }),
        switcher("Tooth", TEETH.map(([k, l]) => [k, l, k !== "all" && !hasTooth(k)]), vTooth, (v) => { ui.volumeTooth = v; }),
      ],
      valueOf: (n) => volumes(n, vTooth)[ui.volumeMeasure],
      yLabel: ui.volumeMeasure === "total" && vTooth === "all" ? "Molar row volume (mm³)" : vm.yLabel, fy: vm.fy,
      columns: Object.entries(VOLUME_MEASURES).map(([k, d]) => ({ key: k, label: k === "share" ? "Enamel share" : `${d.label} (mm³)`,
        numeric: true, format: fmt(d.cell || d.fy) })),
      rowOf: (n) => volumes(n, vTooth),
      empty: `No ${vTooth === "all" ? "" : `${vTooth} `}volumes for the specimens shown.`,
      exportKey: "molar-volume", exportAs: () => volumeMeasure(ui.volumeMeasure, vTooth),
    });

    // ---- Thickness: only once Molar enamel has measured it.
    const thick = (n, which) => {
      if (which === "all") {
        return { thickness: m(n).thickness_um, junction: m(n).junction_area_mm2, perArea: m(n).thickness_um_per_mm2,
          p25: m(n).thickness_p25_um, p75: m(n).thickness_p75_um };
      }
      const t = tooth(n, which);
      return t ? { thickness: t.thickness_um, junction: t.junction_area_mm2, perArea: t.thickness_um_per_mm2 } : {};
    };
    const measured = shown.some((n) => m(n).thickness_um != null);
    let thicknessCard = null;
    if (measured) {
      const hasToothThickness = (which) => shown.some((n) => tooth(n, which)?.thickness_um != null);
      const tTooth = ui.thicknessTooth === "all" || hasToothThickness(ui.thicknessTooth) ? ui.thicknessTooth : "all";
      // Summaries from before the junction was measured have thickness alone.
      const known = (k) => shown.some((n) => thick(n, tTooth)[k] != null);
      const measure = known(ui.thicknessMeasure) ? ui.thicknessMeasure : "thickness";
      const tm = THICKNESS_MEASURES[measure];
      const teethText = tTooth === "all" ? "the three molars together" : `${tTooth} alone`;
      thicknessCard = measureCard({
        title: "Enamel thickness",
        sub: { thickness: `median across the enamel, ${teethText}`,
          junction: `the area where enamel meets dentin, ${teethText}`,
          perArea: `median thickness divided by the area where enamel meets dentin, ${teethText}` }[measure],
        names,
        controls: [
          switcher("Thickness measured", Object.entries(THICKNESS_MEASURES).map(([k, d]) => [k, d.label, !known(k)]),
            measure, (v) => { ui.thicknessMeasure = v; }),
          switcher("Tooth", TEETH.map(([k, l]) => [k, l, k !== "all" && !hasToothThickness(k)]), tTooth, (v) => { ui.thicknessTooth = v; }),
        ],
        valueOf: (n) => thick(n, tTooth)[measure],
        yLabel: tm.yLabel, fy: tm.fy,
        columns: [{ key: "thickness", label: "Median (µm)", numeric: true, format: fmt(sig3) },
          ...(tTooth === "all" ? [{ key: "p25", label: "25th percentile (µm)", numeric: true, format: fmt(sig3) },
            { key: "p75", label: "75th percentile (µm)", numeric: true, format: fmt(sig3) }] : []),
          { key: "junction", label: "Junction area (mm²)", numeric: true, format: fmt(sig3) },
          { key: "perArea", label: "Thickness ÷ area (µm/mm²)", numeric: true, format: fmt(sig3) }],
        rowOf: (n) => thick(n, tTooth),
        empty: `No thickness measured${tTooth === "all" ? "" : ` for ${tTooth}`} in the specimens shown. Run Molar enamel again to measure it.`,
        extra: measure === "perArea" ? noteLine("The ratio still depends on tooth size: area grows with the square of size and thickness "
          + "only in proportion, so a tooth of the same shape but 10% smaller reads 11% higher.") : null,
        exportKey: "molar-thickness", exportAs: () => thicknessMeasure(measure, tTooth),
      });
    }

    // ---- Patches: only once Molar enamel has looked for them (version 6 on).
    const patchOf = (n, which) => (which === "all" ? m(n) : tooth(n, which) || {});
    const patched = shown.some((n) => m(n).thin_patch_share != null);
    let patchCard = null;
    if (patched) {
      const hasToothPatches = (which) => shown.some((n) => tooth(n, which)?.thin_patch_share != null);
      const pTooth = ui.patchTooth === "all" || hasToothPatches(ui.patchTooth) ? ui.patchTooth : "all";
      const knownPatch = (k) => shown.some((n) => patchOf(n, pTooth)[PATCH_MEASURES[k].key] != null);
      const pm = knownPatch(ui.patchMeasure) ? ui.patchMeasure : "thinShare";
      const spec = PATCH_MEASURES[pm];
      const teethText = pTooth === "all" ? "the three molars together" : `${pTooth} alone`;
      patchCard = measureCard({
        title: "Enamel patches",
        sub: `${spec.sub}, ${teethText}`,
        names,
        controls: [
          switcher("Patches measured", Object.entries(PATCH_MEASURES).map(([k, d]) => [k, d.label, !knownPatch(k)]),
            pm, (v) => { ui.patchMeasure = v; }),
          switcher("Tooth", TEETH.map(([k, l]) => [k, l, k !== "all" && !hasToothPatches(k)]), pTooth, (v) => { ui.patchTooth = v; }),
        ],
        valueOf: (n) => patchOf(n, pTooth)[spec.key],
        // (toothName is declared with the exports, further down.)
        yLabel: `${pTooth === "all" ? "Molar row" : pTooth} ${spec.title}`, fy: spec.fy,
        columns: [{ key: "thin_patch_share", label: "Thin patches", numeric: true, format: fmt(pctFmt) },
          { key: "thin_patches_per_mm2", label: "Thin per mm²", numeric: true, format: fmt(sig3) },
          { key: "thickness_shortfall", label: "Thickness shortfall", numeric: true, format: fmt(pctFmt) },
          { key: "low_density_patch_share", label: "Less dense patches", numeric: true, format: fmt(pctFmt) },
          { key: "low_density_patches_per_mm2", label: "Less dense per mm²", numeric: true, format: fmt(sig3) },
          { key: "density_shortfall", label: "Density shortfall", numeric: true, format: fmt(pctFmt) }],
        rowOf: (n) => patchOf(n, pTooth),
        empty: `No patches looked for${pTooth === "all" ? "" : ` on ${pTooth}`} in the specimens shown. Run Molar enamel again to find them.`,
        extra: noteLine("Each point of the enamel–dentin junction is compared with the enamel within about 0.1 mm around it on the same "
          + "tooth. Fissure floors are thinner than their walls on every tooth, so read a knockout against its wildtype, not against zero."
          + (pTooth === "M3" || pTooth === "all" ? " At P22, M3 is still maturing." : "")),
        exportKey: "molar-patches", exportAs: () => patchMeasure(pm, pTooth),
      });
    }

    // Each measurement as the exports name it, under any switch; the Prism
    // tables take every one of them, the figures the ones the cards show.
    const unitName = { mgHA: "mg HA/cm³", raw: "raw" };
    const densityMeasure = (u, st) => {
      if (u === "ratio") return { title: "Molar enamel/dentin density ratio", yLabel: "Enamel / dentin density", valueOf: (n) => m(n).density_ratio };
      const what = `${u === "mgHA" ? "density" : "intensity"}, ${statLabel[st].toLowerCase()} (${unitName[u]})`;
      return { title: `Molar enamel ${what}`, yLabel: `Enamel ${what}`, valueOf: (n) => m(n)[`enamel_${st}_${u}`] };
    };
    const toothName = (t) => (t === "all" ? "Molar row" : t);
    const volumeMeasure = (k, t) => {
      const title = { share: `${toothName(t)} enamel share of volume (%)`, enamel: `${toothName(t)} enamel volume (mm³)`,
        dentin: `${toothName(t)} dentin volume (mm³)`, total: `${toothName(t)} volume (mm³)` }[k];
      return { title, yLabel: title, valueOf: (n) => volumes(n, t)[k], scale: k === "share" ? 100 : 1 };
    };
    const patchMeasure = (k, t) => {
      const spec = PATCH_MEASURES[k], title = `${toothName(t)} ${spec.title}`;
      return { title, yLabel: title, valueOf: (n) => patchOf(n, t)[spec.key], scale: spec.share ? 100 : 1 };
    };
    const thicknessMeasure = (k, t) => {
      const title = { thickness: `${toothName(t)} enamel thickness, median (µm)`, junction: `${toothName(t)} enamel–dentin junction area (mm²)`,
        perArea: `${toothName(t)} enamel thickness ÷ junction area (µm/mm²)` }[k];
      return { title, yLabel: title, valueOf: (n) => thick(n, t)[k] };
    };
    for (const u of ["mgHA", "raw"]) for (const st of DENSITY_STATS) offerColumn(names, densityMeasure(u, st));
    offerColumn(names, densityMeasure("ratio"));
    for (const [t] of TEETH) for (const k of Object.keys(VOLUME_MEASURES)) offerColumn(names, volumeMeasure(k, t));
    if (measured) for (const [t] of TEETH) for (const k of Object.keys(THICKNESS_MEASURES)) offerColumn(names, thicknessMeasure(k, t));
    if (patched) for (const [t] of TEETH) for (const k of Object.keys(PATCH_MEASURES)) offerColumn(names, patchMeasure(k, t));

    return h("section", { class: "section molar-section" },
      h("div", { class: "section-head" }, h("h2", null, "Molar enamel"),
        h("span", { class: "sub" }, `${shown.length < names.length ? `${shown.length} of ${names.length}` : names.length} specimen${names.length === 1 ? "" : "s"}. Enamel: ${[...thresholds].map((t) => (t === "local" ? "between its halfway surfaces, read per specimen" : t === "fixed" ? "one threshold, fixed for the cohort" : t === "otsu" ? "Otsu threshold per specimen" : "log-histogram threshold per specimen")).join("; ")}.`)),
      h("div", { class: "stack-cards" }, densityCard, volumeCard, thicknessCard, patchCard));
  }

  // ------------------------------------------------------------ incisor

  // Each specimen's profile of one quantity: [[% along, value or null], ...],
  // median-smoothed along the scan, empty wherever the median would bridge a
  // gap. `quantity` names what `pick` reads, for the cache.
  function curves(specimens, quantity, pick) {
    const out = {};
    for (const { name } of specimens) {
      const key = `${quantity}|${ui.k}|${name}`;
      if (!curveCache.has(key)) {
        const p = data.incisor[name];
        const pts = [];
        p.z_percent.forEach((z, i) => {
          const v = z != null ? pick(p, i, name) : null;
          if (finite(v)) pts.push([z, v, p.z[i]]);
        });
        const curve = pts.length >= 5 ? smoothProfile(pts, ui.k) : [];
        curveCache.set(key, curve.some((q) => q[1] != null) ? curve : null);
      }
      if (curveCache.get(key)) out[name] = curveCache.get(key);
    }
    return out;
  }

  // A cell's mean profile on a 1% grid, with a +/- 1 SD band where two or more
  // specimens reach. Nothing is drawn cervical to `from`.
  function cellMean(curveOf, members, from = null) {
    const have = members.filter((n) => curveOf[n]);
    if (!have.length) return null;
    const mean = [], band = [];
    for (const x of GRID) {
      const vals = from != null && x < from ? [] : have.map((n) => valueAt(curveOf[n], x)).filter((v) => v != null);
      const d = describe(vals);
      mean.push([x, d.mean]);
      band.push(d.sd != null ? [x, d.mean - d.sd, d.mean + d.sd] : [x, null, null]);
    }
    return mean.some((q) => q[1] != null) ? { mean, band, n: have.length } : null;
  }

  // Where a cell's enamel curve starts: the report's coverage floor over its specimens.
  const floorOf = (members) => coverageFloor(members.map((n) => data.incisor[n]?.coverage));

  // Landmarks averaged over a panel's specimens, with how far each end
  // spreads between them: only meaningful within one age.
  function landmarkSpans(members) {
    const withMarks = members.filter((n) => data.landmarks[n]);
    if (!ui.landmarks || !withMarks.length) return { spans: [], mixed: false };
    if (new Set(members.map((n) => levelOf(metaOf[n], "age"))).size > 1) return { spans: [], mixed: true };
    const acc = {};
    for (const n of withMarks) for (const r of data.landmarks[n]) {
      (acc[r.landmark] ||= { starts: [], ends: [], kind: r.kind });
      if (r.start != null) acc[r.landmark].starts.push(r.start);
      if (r.end != null) acc[r.landmark].ends.push(r.end);
    }
    const said = (d) => `${sig(d.mean, 3)}${d.sd != null ? ` ± ${sig(d.sd, 2)}` : ""}%`;
    return {
      spans: Object.entries(acc).filter(([, v]) => v.starts.length).map(([lm, v]) => {
        const start = describe(v.starts);
        const end = v.kind === "span" && v.ends.length ? describe(v.ends) : null;
        return {
          x0: start.mean, x1: end ? end.mean : null, label: lm, tone: landmarkTone(lm),
          sd0: start.sd, sd1: end ? end.sd : null,
          detail: `${end ? `${said(start)} to ${said(end)}` : said(start)} (mean ± SD of ${start.n})`,
        };
      }),
      mixed: false, n: withMarks.length,
    };
  }

  /**
   * Profiles along the tooth, one panel per stratum, the main comparison's
   * means overlaid in each. `dentin`, when given, adds each cell's mean dentin
   * as a dashed line in the cell's colour. `spec()` is the same for the figure
   * export, with each specimen's values on the 1% grid for the Prism tables:
   * the ones the means are taken over, so Prism's means match the page's.
   */
  function trajectory({ specimens, enamel, dentin = null, yLabel, fy, rules = [], label, title }) {
    const strata = arrange(specimens, ui.by, order);
    const panels = [];
    const drawn = new Set();
    const replicates = [];
    const slotOf = (level) => (order[mainKey()] || []).indexOf(level);
    // Each line in its group's colour, or the one picked for its kind; the
    // landmarks in theirs. Painted when drawn, so a figure asked for after a
    // colour changes has the new one.
    const paint = (sr) => {
      const own = sr.role === "specimen" ? markColor("specimen") : sr.role === "dentin" ? markColor("dentin") : null;
      const group = colorOf(sr.level);
      return { ...sr, color: own || group, bandColor: sr.band ? markColor("ribbon") || group : undefined };
    };
    const paintSpan = (sp) => ({ ...sp, color: landmarkColor(sp.label) || undefined });
    const onGrid = (curveOf, members, from) => members.filter((n) => curveOf[n]).map((n) => ({
      label: n, values: GRID.map((x) => (from != null && x < from ? null : valueAt(curveOf[n], x))) }));
    let mixedAges = false, marked = 0;
    for (const st of strata) {
      const series = [];
      const members = st.cells.flatMap((c) => c.members);
      if (ui.specimens) {
        for (const cell of st.cells) for (const n of cell.members) {
          if (enamel[n]) series.push({ id: n, label: n, level: cell.level, points: enamel[n], width: 1, opacity: 0.35, tip: false,
            role: "specimen", slot: slotOf(cell.level) });
        }
      }
      const counts = [];
      for (const cell of st.cells) {
        const from = floorOf(cell.members);
        const m = cellMean(enamel, cell.members, from);
        if (!m) continue;
        drawn.add(cell.level);
        counts.push(`${cell.name} n = ${m.n}`);
        series.push({ id: `${st.key}|${cell.level}`, table: cellLabel(cell, st), label: `${cell.name} ${label} (n = ${m.n})`,
          level: cell.level, points: m.mean, band: m.band, width: 2, role: "mean", slot: slotOf(cell.level) });
        replicates.push({ title: cellLabel(cell, st), specimens: onGrid(enamel, cell.members, from) });
      }
      if (dentin) {
        for (const cell of st.cells) {
          const m = cellMean(dentin, cell.members);
          if (!m) continue;
          series.push({ id: `${st.key}|${cell.level}|dentin`, table: `${cellLabel(cell, st)}, dentin`, label: `${cell.name} dentin (n = ${m.n})`,
            level: cell.level, points: m.mean, width: 1.5, dash: "6 4", role: "dentin", slot: slotOf(cell.level) });
          replicates.push({ title: `${cellLabel(cell, st)}, dentin`, specimens: onGrid(dentin, cell.members, null) });
        }
      }
      if (!series.some((s) => s.width === 2)) continue;
      const lm = landmarkSpans(members);
      mixedAges ||= lm.mixed;
      if (lm.spans.length) marked += lm.n;
      panels.push({ st, series, spans: lm.spans, counts });
    }
    if (!panels.length) return null;
    const values = panels.flatMap((p) => p.series.flatMap((s) => [...s.points.map((q) => q[1]), ...(s.band || []).flatMap((b) => [b[1], b[2]])]));
    for (const r of rules) if (r.y != null) values.push(r.y);
    const yDomain = extent(values);
    const single = panels.length === 1 && !panels[0].st.label;
    const chartOf = (p) => lineChart({
      series: p.series.map(paint), spans: p.spans.map(paintSpan), rules, xDomain: [0, 100], yDomain,
      xLabel: "Position along the incisor (%)", yLabel, height: single ? 360 : 300, fy,
      fx: (v) => `${Math.round(v)}`, tipHead: (x) => `${x.toFixed(0)}% along the tooth${p.st.label ? `, ${p.st.label}` : ""}`,
    });
    const chart = single ? chartOf(panels[0]) : h("div", { class: "panels" }, panels.map((p) => h("div", { class: "panel-chart" },
      h("div", { class: "panel-title" }, h("h4", null, p.st.label || "All"), h("span", { class: "muted" }, p.counts.join(", "))),
      chartOf(p))));
    const tabled = panels.flatMap((p) => p.series.filter((s) => s.table));
    const table = () => dataTable(
      [{ key: "x", label: "Position (%)", numeric: true }, ...tabled.map((s) => ({ key: s.id, label: s.table, numeric: true, format: (v) => (v == null ? "–" : fy(v)) }))],
      GRID.map((x) => Object.fromEntries([["x", x], ...tabled.map((s) => [s.id, s.points[x]?.[1]])])));
    const legendOf = () => {
      const items = (order[mainKey()] || []).filter((lv) => drawn.has(lv))
        .map((lv) => ({ label: levelName(mainKey(), lv), color: colorOf(lv), slot: slotOf(lv) }));
      if (dentin && items.length) items.push({ label: "Dentin", color: markColor("dentin") || "var(--ink-2)", kind: "dash" });
      return items;
    };
    const legendItems = legendOf();
    const notes = [];
    if (ui.landmarks && mixedAges) notes.push("Landmark bands appear in panels whose specimens share one age.");
    else if (ui.landmarks && marked) notes.push(`Landmarks over the panel's specimens (${marked} with landmarks): a line marks each mean end, in a lighter strip one standard deviation either side of it.`);
    const spec = () => ({
      kind: "xy", title, xLabel: "Position along the incisor (%)", yLabel, xDomain: [0, 100], yDomain, rules,
      legend: legendOf().map((it) => ({ label: it.label, color: it.color, slot: it.slot, dentin: it.kind === "dash" })),
      panels: panels.map((p) => ({ title: p.st.label, spans: p.spans.map(paintSpan), series: p.series.map(paint) })),
      x: GRID, replicates,
    });
    return { chart, table, legendItems, notes, spec };
  }

  const noteLine = (text) => h("div", { class: "muted", style: { fontSize: "12.5px", marginTop: "6px" } }, text);

  function incisorSection() {
    const all = included(Object.keys(data.incisor).filter(inProject), "incisor");
    if (!all.length) return emptySection("Incisor mineralization", Object.keys(data.incisor).some(inProject)
      ? "Every specimen with an incisor profile is flagged and left out." : "Run Incisor enamel to trace density along each tooth.");
    const specimens = pool(all);

    // The ratio over the slices the cleanup kept.
    const ratio = trajectory({
      specimens, label: "mean", yLabel: "Enamel / dentin", fy: (v) => sig(v, 3),
      enamel: curves(specimens, "ratio", (p, i) => (p.kept[i] ? p.ratio[i] : null)),
      rules: [{ y: 1, label: "dentin density" }], title: "Incisor enamel/dentin ratio along the tooth",
    });
    if (ratio) offerTable(() => ratio.spec());
    const outliers = specimens.reduce((sum, s) => sum + data.incisor[s.name].outlier.filter(Boolean).length, 0);
    const stale = specimens.filter((s) => data.incisor[s.name].stale).map((s) => s.name);
    const cleanupNotes = [];
    if (outliers) cleanupNotes.push(`${outliers.toLocaleString()} slice${outliers === 1 ? "" : "s"} left out as outliers, more than 0.08 off their specimen's trend.`);
    if (stale.length) cleanupNotes.push(`Written by an older build of Incisor enamel, so they can open high and drop. Run Incisor enamel on them again: ${stale.join(", ")}.`);
    const ratioNotes = [...cleanupNotes, ...(ratio?.notes || [])];
    const ratioCard = chartCard({
      ...titled("incisor-ratio", "Mineralization trajectory"), sub: "enamel/dentin density ratio, cervical loop (0%) to incisal tip (100%); bands are ±1 SD across specimens",
      legendItems: ratio?.legendItems, chart: ratio ? ratio.chart : h("div", { class: "empty" }, h("span", null, "No measured slices in the specimens shown.")),
      table: ratio?.table, extra: ratioNotes.length ? noteLine(ratioNotes.join(" ")) : null,
      onExport: ratio ? offerFigure("incisor-ratio", "Mineralization trajectory", () => ratio.spec()) : null,
    });

    // The same profile in density: converted with each scan's HA calibration, or as stored.
    const calOf = (n) => data.incisor[n].calibration;
    const calibrated = specimens.filter((s) => calOf(s.name));
    const units = ui.units || (calibrated.length ? "mgHA" : "raw");
    const mg = units === "mgHA";
    // The profile in either unit, with the dentin line or without: the card
    // draws the one picked, the Prism tables take both units with dentin.
    const densityIn = (u, withDentin) => {
      const inMg = u === "mgHA";
      const group = inMg ? calibrated : specimens;
      if (!group.length) return null;
      const convert = (n, v) => (v == null ? null : inMg ? toMgHA(v, calOf(n)) : v);
      const enamel = curves(group, `enamel|${u}`, (p, i, n) => (p.kept[i] ? convert(n, p.density[i]) : null));
      const dentin = withDentin ? curves(group, `dentin|${u}`,
        (p, i, n) => (DENTIN_MEASURED.includes(p.status[i]) ? convert(n, p.dentin_mode?.[i] ?? null) : null)) : null;
      const withLine = Boolean(dentin && Object.keys(dentin).length);
      const result = trajectory({
        specimens: group, enamel, dentin: withLine ? dentin : null, label: "enamel", fy: numFmt,
        yLabel: inMg ? "Density (mg HA/cm³)" : "Intensity (raw)",
        title: `Incisor enamel ${inMg ? "density" : "intensity"} along the tooth (${inMg ? "mg HA/cm³" : "raw"})`,
      });
      return result ? { ...result, withLine } : null;
    };
    const density = densityIn(units, ui.dentin);
    const hasDentin = Boolean(density?.withLine);
    for (const u of ["mgHA", "raw"]) offerTable(() => densityIn(u, true)?.spec());
    const unitSeg = h("div", { class: "seg", role: "group", "aria-label": "Density units" },
      [["mgHA", "mg HA/cm³"], ["raw", "Raw intensity"]].map(([v, l]) => h("button", {
        "aria-pressed": units === v ? "true" : "false", onclick: () => { ui.units = v; remember(); render(); },
      }, l)));
    const dentinCheck = h("label", { class: "check" }, h("input", { type: "checkbox", checked: ui.dentin,
      onchange: (e) => { ui.dentin = e.target.checked; remember(); render(); } }), "Dentin line");
    const missing = mg ? specimens.filter((s) => !calOf(s.name)).map((s) => s.name) : [];
    const densityNotes = [];
    if (mg && missing.length && calibrated.length) densityNotes.push(`Not shown, with no HA calibration: ${missing.join(", ")}. Raw intensity includes them.`);
    if (density && hasDentin) densityNotes.push("Dentin is the dentin peak fitted in each slice. Toward the tip, where little dentin is left, it drifts up toward the enamel.");
    if (!mg) densityNotes.push("Raw values are comparable only between scans made with one scanner setting and calibration.");
    if (density) densityNotes.push(...density.notes.filter((t) => !ratio?.notes.includes(t)));
    const densityCard = chartCard({
      ...titled("incisor-density", "Enamel density along the tooth"),
      sub: mg ? "converted to mg HA/cm³ with each scan's HA calibration; bands are ±1 SD" : "stored scan values; bands are ±1 SD",
      controls: [unitSeg, dentinCheck],
      legendItems: density?.legendItems,
      chart: density ? density.chart : h("div", { class: "empty" }, h("span", null, mg
        ? "No HA calibration for these scans. The converter copies it from a Scanco ISQ; for other scans, enter it in the Molar enamel step's Density calibration settings. Raw intensity shows every specimen."
        : "No measured slices in the specimens shown.")),
      table: density?.table,
      extra: densityNotes.length ? noteLine(densityNotes.join(" ")) : null,
      onExport: density ? offerFigure("incisor-density", "Enamel density along the tooth", () => density.spec()) : null,
    });

    // Each specimen's landmarks, read off its cleaned 21-slice median as the report reads a curve.
    const metrics = data.incisor_metrics;
    const metricNames = included(Object.keys(metrics).filter(inProject), "incisor");
    const fy3 = (v) => sig(v, 3);
    const metricCards = [
      ["crossover_z", "Where enamel overtakes dentin", "where the ratio last rises through 1.0 and stays above", "Crosses 1.0 at (% along)",
        "Incisor crossover, where enamel overtakes dentin (% along)"],
      ["plateau_ratio", "Mature plateau", "mean ratio 70–95% along the tooth", "Mean enamel/dentin ratio, 70–95%",
        "Incisor mature plateau, enamel/dentin ratio 70–95% along"],
      ["peak_ratio", "Peak mineralization", "highest ratio along the tooth", "Peak enamel/dentin ratio", "Incisor peak enamel/dentin ratio"],
    ].map(([key, title, sub, yLabel, exportTitle]) => {
      const valueOf = (n) => metrics[n]?.[key];
      const measure = { title: exportTitle, yLabel, valueOf };
      offerColumn(metricNames, measure);
      const s = strip(valueOf, metricNames, { yLabel, fy: fy3 });
      return s ? chartCard({ ...titled(`incisor-${key}`, title), sub, chart: s.chart, extra: statsBlock(valueOf, metricNames, fy3),
        onExport: offerFigure(`incisor-${key}`, title, () => columnSpec(metricNames, measure)) }) : null;
    }).filter(Boolean);
    return h("section", { class: "section" },
      h("div", { class: "section-head" }, h("h2", null, "Incisor mineralization"),
        h("span", { class: "sub" }, "Along the tooth from the cervical loop, cleaned as the R report cleans it: slices that pass its quality filters, "
          + "outliers left out, and a running median that leaves gaps empty. A group's curve starts where its specimens measure half their slices.")),
      ratioCard,
      h("div", { style: { marginTop: "14px" } }, densityCard),
      metricCards.length ? h("div", { class: ui.by.length > 1 ? "charts-2" : "charts-3", style: { marginTop: "14px" } }, metricCards) : null);
  }

  // ------------------------------------------------------------ segmentation
  function segmentationSection() {
    const names = included(Object.keys(data.segmentation).filter(inProject), "segmentation");
    if (!names.length) return emptySection("Segmentation", Object.keys(data.segmentation).some(inProject)
      ? "Every segmented specimen is flagged and left out." : "Segment specimens to compare structure volumes and check results.");
    const charts = [["incisor_mm3", "Incisor"], ["bone_mm3", "Bone"], ["molar_mm3", "Molars"]].map(([key, label]) => {
      const valueOf = (n) => data.segmentation[n][key];
      const measure = { title: `Segmented ${label.toLowerCase()} volume (mm³)`, yLabel: `${label} volume (mm³)`, valueOf };
      offerColumn(names, measure);
      const s = strip(valueOf, names, { yLabel: `${label} volume (mm³)`, fy: (v) => sig(v, 3) });
      return s ? chartCard({ ...titled(`segmentation-${key}`, `${label} volume`), sub: "mm³", chart: s.chart, extra: statsBlock(valueOf, names, (v) => sig(v, 3)),
        onExport: offerFigure(`segmentation-${key}`, `${label} volume`, () => columnSpec(names, measure)) }) : null;
    }).filter(Boolean);
    const rows = arrange(pool(names), ui.by, order).flatMap((st) => st.cells.flatMap((cell) => cell.members.map((n) => ({ n, st, cell }))));
    const [, ...splits] = ui.by;
    return h("section", { class: "section" },
      h("div", { class: "section-head" }, h("h2", null, "Segmentation"),
        h("span", { class: "sub" }, "Incisor and bone volumes come from the hemimandible preview; build it to include a specimen.")),
      charts.length ? h("div", { class: ui.by.length > 1 ? "charts-2" : "charts-3" }, charts) : null,
      h("div", { class: "table-wrap", style: { marginTop: "14px" } }, h("table", { class: "data" },
        h("thead", null, h("tr", null, h("th", null, "Specimen"), h("th", null, factor(mainKey())?.label),
          splits.map((k) => h("th", null, factor(k)?.label)), h("th", null, "Checks"), h("th", { class: "n" }, "Problems"))),
        h("tbody", null, rows.map(({ n, st, cell }) => h("tr", { class: "clickable", onclick: openSample(n) },
          h("td", null, h("span", { class: "specimen", style: { fontSize: "15px" } }, n)),
          h("td", null, h("span", { class: "group-label" }, h("i", { class: "sw", style: { background: colorOf(cell.level) } }), cell.name)),
          splits.map((k) => h("td", null, levelName(k, levelOf(st.meta, k)))),
          h("td", null, data.segmentation[n].validation ? statusBadge(data.segmentation[n].validation) : "–"),
          h("td", { class: "n" }, data.segmentation[n].problems ?? "–")))))));
  }

  // ------------------------------------------------------------ flagged
  async function editFlag(names, current) {
    const outcome = await flagDialog(names, current);
    if (outcome === "included") toast(`${names.length === 1 ? names[0] : `${names.length} specimens`} back in the results.`);
    if (outcome) load();
  }

  function flaggedPointer() {
    const flagged = scoped().filter((s) => flagOf[s.name]);
    if (!flagged.length) return null;
    return h("div", { class: "flag-pointer" }, icon("flag"),
      h("span", null, `${flagged.length} specimen${flagged.length === 1 ? " is" : "s are"} flagged and left out.`),
      h("button", { class: "btn small ghost", onclick: () => document.getElementById("flagged")?.scrollIntoView({ behavior: "smooth" }) },
        "Read the notes"));
  }

  // Every flagged specimen, what it is left out of and why: the page's record of its exclusions.
  function flaggedSection() {
    const flagged = scoped().filter((s) => flagOf[s.name]).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    if (!flagged.length) return null;
    const include = (name) => api.post("/api/flags", { names: [name], flag: null })
      .then(() => { toast(`${name} is back in the results.`); load(); }).catch(errorToast);
    return h("section", { class: "section", id: "flagged" },
      h("div", { class: "section-head" }, h("h2", null, "Flagged and excluded"),
        h("span", { class: "sub" }, `${flagged.length} specimen${flagged.length === 1 ? "" : "s"}, left out of the charts, statistics and exports `
          + "of the analyses named. The metadata export lists every specimen with these notes."),
        h("a", { class: "btn small", href: "/api/export/excluded", download: "", title: "Download these notes as CSV" }, icon("download"), "Download notes")),
      h("div", { class: "table-wrap" }, h("table", { class: "data flag-table" },
        h("thead", null, h("tr", null, h("th", null, "Specimen"), h("th", null, "Group"), h("th", null, "Left out of"),
          h("th", null, "Note"), h("th", { class: "hide-sm" }, "Flagged"), h("th", null, h("span", { class: "sr-only" }, "Actions")))),
        h("tbody", null, flagged.map((s) => {
          const flag = flagOf[s.name];
          return h("tr", null,
            h("td", { class: "who" }, h("a", { class: "specimen", style: { fontSize: "15px" }, href: `#/sample/${encodeURIComponent(s.name)}` }, s.name)),
            h("td", { class: "nowrap group" }, s.group),
            h("td", { class: "scope" }, exclusionText(flag).replace(/^./, (c) => c.toUpperCase())),
            h("td", { class: "note" }, flag.note || h("span", { class: "muted" }, "No note")),
            h("td", { class: "hide-sm muted nowrap", title: flag.at || "" }, flag.at ? relTime(flag.at) : "–"),
            h("td", { class: "actions-cell" }, h("div", { class: "row" },
              h("button", { class: "btn small ghost", onclick: () => editFlag([s.name], flag) }, icon("edit"), "Edit"),
              h("button", { class: "btn small ghost", onclick: () => include(s.name) }, "Include again"))));
        })))));
  }

  // Every specimen in the CSVs, or once some are hidden, the ones shown (flagged
  // ones too: each CSV leaves out the specimens flagged out of its analysis).
  function linkCsvs() {
    const shown = data.samples.filter((s) => isShown(s.name)).map((s) => s.name);
    const some = shown.length < data.samples.length;
    for (const { kind, label, link } of csvs) {
      link.href = some ? `/api/export/${kind}?s=${shown.map(encodeURIComponent).join(",")}` : `/api/export/${kind}`;
      link.title = some ? `Download ${label.toLowerCase()} of the ${shown.length} specimen${shown.length === 1 ? "" : "s"} shown, as CSV`
        : `Download ${label.toLowerCase()} as CSV`;
      if (shown.length) link.removeAttribute("aria-disabled"); else link.setAttribute("aria-disabled", "true");
    }
  }

  function emptySection(title, text) {
    return h("section", { class: "section" }, h("div", { class: "section-head" }, h("h2", null, title)),
      h("div", { class: "empty" }, h("span", null, text)));
  }

  function render() {
    if (!data) return;
    const mine = scoped();
    const total = mine.length;
    const flagged = mine.filter((s) => flagOf[s.name]).length;
    const shown = mine.filter((s) => isShown(s.name) && inAnyAnalysis(s)).length;
    const inAny = mine.filter(inAnyAnalysis).length;
    sub.textContent = `${plural(total, "specimen")}${ui.project ? ` of the ${data.samples.length} in the workspace` : ""}`
      + `${flagged ? `, ${flagged} flagged` : ""}${shown < inAny ? `, ${shown} shown` : ""}`;
    renderHead();
    if (!total) {
      replace(filters);
      replace(content, ui.project
        ? h("div", { class: "empty" }, h("h3", null, "No specimens in this project yet"),
          h("button", { class: "btn primary", onclick: () => editMembers(current()) }, icon("plus"), "Add specimens"))
        : h("div", { class: "empty" }, h("h3", null, "No results yet"), h("a", { class: "btn primary", href: "#/import" }, "Import specimens")));
      return;
    }
    renderFilters();
    linkCsvs();
    applyPageInk(page);
    figures = [];
    tables = [];
    replace(content, molarSection(), incisorSection(), segmentationSection(), flaggedSection());
  }

  const reload = debounce(load, 1500);
  const off = on("jobs", (job) => {
    if (job && job.status === "done" && ["molar_quant", "incisor_quant", "landmarks", "preview", "segment"].includes(job.step)) reload();
  });
  // A colour or title changed, here or in another browser: redrawn once it settles, the scroll kept.
  const offStyle = onStyle(debounce(render, 120));
  const offSettings = on("settings", settingsChanged);
  // A specimen renamed or removed (an event with no summary) changes the names the results are kept under.
  const offSamples = on("samples", (summary) => { if (!summary) reload(); });
  await load();
  return () => { flushProject(); off(); offStyle(); offSettings(); offSamples(); };
}

// ================================================================ comparison
function defaultByFor(samples) {
  const several = factorsWithValues(samples, 2);
  const main = several.includes("genotype") ? "genotype" : several[0] || factorsWithValues(samples)[0] || "genotype";
  // Age is split out from the start: it moves everything along the tooth.
  return main !== "age" && several.includes("age") ? [main, "age"] : [main];
}

/** The details to compare `samples` by: those `by` names that exist, or a default from what varies among them. */
function validByFor(by, samples) {
  const known = new Set(FACTORS.map((f) => f.key));
  const out = [...new Set((Array.isArray(by) ? by : []).filter((k) => known.has(k)))].slice(0, MAX_SPLITS + 1);
  return out.length ? out : defaultByFor(samples);
}

// ================================================================ choosing specimens
/**
 * A checklist of specimens in groups (`by`, over `order`), ticked as `ticked`
 * says; resolves with the names ticked, or null on Cancel. It works on a copy.
 * The search reads names, details and library folders ("KO P52", a batch's
 * folder), and the buttons act on what it found. `note(name)` adds a word
 * beside a specimen.
 *
 *   specimens  [{name, meta, folder, excluded}]
 */
function specimenChecklist({ specimens, ticked, by, order, colorOf, note = () => null, title, intro, confirm }) {
  return new Promise((resolve) => {
    let answered = false;
    const names = specimens.map((s) => s.name);
    const draft = new Set(names.filter((n) => ticked.has(n)));
    const bySpec = Object.fromEntries(specimens.map((s) => [s.name, s]));
    const textOf = Object.fromEntries(specimens.map((s) => [s.name, [s.name, s.folder || "",
      ...FACTORS.flatMap((f) => { const lv = levelOf(s.meta, f.key); return lv ? [lv, levelName(f.key, lv)] : []; })].join(" ").toLowerCase()]));
    let terms = [];
    // Each word typed starts a word of the specimen's: "male" finds the males, not the females.
    const startsWord = (text, term) => {
      for (let i = text.indexOf(term); i >= 0; i = text.indexOf(term, i + 1)) if (!i || !/[a-z0-9]/.test(text[i - 1])) return true;
      return false;
    };
    const matches = (n) => terms.every((t) => startsWord(textOf[n], t));
    const found = (list) => list.filter(matches);
    const tick = (list, on) => { for (const n of list) on ? draft.add(n) : draft.delete(n); sync(); };

    const rows = {}, groups = [];
    const empty = h("div", { class: "pick-empty muted" }, "No specimen matches.");
    const list = h("div", { class: "pick-list" });
    for (const st of arrange(specimens.map((s) => ({ name: s.name, meta: s.meta || {} })), by, order)) {
      for (const cell of st.cells) {
        const where = st.label ? `${cell.name}, ${st.label}` : cell.name;
        const head = h("input", { type: "checkbox", "aria-label": `Every ${where} specimen`,
          onchange: (e) => tick(found(cell.members), e.target.checked) });
        const count = h("span", { class: "muted" });
        const members = cell.members.map((n) => {
          const box = h("input", { type: "checkbox", onchange: (e) => tick([n], e.target.checked) });
          const said = note(n);
          const folder = bySpec[n].folder || "";
          const el = h("label", { class: `pick-row${said ? " dim" : ""}` }, box,
            h("span", { class: "who" }, n),
            folder ? h("span", { class: "in-folder" }, icon("folder"), folder.split("/").join(" › ")) : null,
            flagTag(bySpec[n].excluded),
            said ? h("span", { class: "muted" }, said) : null);
          rows[n] = { box, el };
          return el;
        });
        const el = h("div", { class: "pick-group", role: "group", "aria-label": where },
          h("label", { class: "pick-head" }, head, h("i", { class: "sw", style: { background: colorOf(cell.level) } }),
            h("span", null, cell.name), st.label ? h("span", { class: "muted" }, st.label) : null, h("span", { class: "spacer" }), count),
          members);
        groups.push({ head, count, el, members: cell.members });
        list.appendChild(el);
      }
    }
    list.appendChild(empty);

    const search = h("input", { type: "search", placeholder: "Find by name, detail or folder", "aria-label": "Find specimens",
      oninput: (e) => { terms = e.target.value.toLowerCase().split(/\s+/).filter(Boolean); sync(); } });
    const tickAll = h("button", { class: "btn small", onclick: () => tick(found(names), true) });
    const untickAll = h("button", { class: "btn small", onclick: () => tick(found(names), false) });
    const onlyThese = h("button", { class: "btn small", title: "Untick everything the search did not find",
      onclick: () => { draft.clear(); tick(found(names), true); } });
    const status = h("span", { class: "muted", "aria-live": "polite" });
    const problem = h("div", { class: "field-error", role: "alert" });

    function sync() {
      const hits = found(names);
      const searching = terms.length > 0;
      for (const n of names) { rows[n].box.checked = draft.has(n); rows[n].el.hidden = !matches(n); }
      for (const g of groups) {
        const seen = found(g.members);
        const seenOn = seen.filter((n) => draft.has(n)).length;
        g.el.hidden = !seen.length;
        g.head.checked = seen.length > 0 && seenOn === seen.length;
        g.head.indeterminate = seenOn > 0 && seenOn < seen.length;
        g.count.textContent = `${g.members.filter((n) => draft.has(n)).length} of ${g.members.length}`;
      }
      empty.hidden = hits.length > 0;
      tickAll.textContent = searching ? `Tick the ${hits.length} found` : "Tick all";
      untickAll.textContent = searching ? `Untick the ${hits.length} found` : "Untick all";
      onlyThese.textContent = `Only the ${hits.length} found`;
      onlyThese.hidden = !searching;
      tickAll.disabled = untickAll.disabled = onlyThese.disabled = !hits.length;
      status.textContent = `${draft.size} of ${names.length} ticked`;
      problem.textContent = "";
    }

    modal({
      title,
      onClose: () => { if (!answered) resolve(null); },
      body: h("div", { class: "stack" },
        intro ? h("p", { class: "soft", style: { margin: 0 } }, intro) : null,
        h("div", { class: "pick-tools" }, search, tickAll, untickAll, onlyThese),
        list,
        h("div", { class: "row" }, status, problem)),
      actions: [
        { label: "Cancel" },
        { label: confirm, primary: true, onClick: () => {
          if (!draft.size) { problem.textContent = "Tick at least one specimen."; return false; }
          answered = true;
          resolve(new Set(draft));
          return true;
        } },
      ],
    });
    sync();
  });
}

// ================================================================ projects
// A project is a named set of specimens with its own comparison and filters,
// kept in the workspace (studio.json, results_projects) so every browser has it.

/** A name for a project, made or renamed; a taken name is said in the dialog. */
function projectDialog({ title, value = "", confirm, intro = null, body = null, submit }) {
  const nameIn = h("input", { type: "text", value, maxlength: 80, placeholder: "Odam study, TGM2 by sex…", "aria-label": "Project name" });
  const problem = h("div", { class: "field-error", role: "alert" });
  const send = async () => {
    const name = nameIn.value.trim();
    if (!name) { problem.textContent = "Give the project a name."; return false; }
    try { await submit(name); } catch (err) { problem.textContent = err.message; return false; }
    return true;
  };
  const dialog = modal({
    title,
    body: h("div", { class: "stack" }, intro ? h("p", { class: "soft", style: { margin: 0 } }, intro) : null,
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Name"), nameIn), body, problem),
    actions: [{ label: "Cancel" }, { label: confirm, primary: true, onClick: send }],
  });
  nameIn.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); dialog.dialog.querySelector("footer .primary")?.click(); } });
}

// Keep a new project in the local copy until the server's settings bring it.
function adopt(made) {
  const settings = state.app?.workspace?.settings;
  if (settings && !projects().some((p) => p.id === made.id)) settings.results_projects = [...projects(), made];
}

/**
 * Make a project, or a copy of `copyOf`. It starts with the specimens shown
 * now (`shownNow`), every one of `everyone`, those in a library folder, or
 * none, to tick next; `onMade(project, pick)` is told which.
 */
function newProjectDialog({ everyone, shownNow = null, by = [], copyOf = null, onMade }) {
  const names = everyone.map((s) => s.name);
  const folderOf = Object.fromEntries(everyone.map((s) => [s.name, s.folder || ""]));
  const radios = {};
  const sets = [];
  if (shownNow && shownNow.length && shownNow.length < names.length) {
    sets.push({ key: "shown", label: `The ${plural(shownNow.length, "specimen")} shown now`, names: () => shownNow });
  }
  const folders = libraryFolders().filter((f) => names.some((n) => inFolder(folderOf[n], f)));
  if (folders.length) {
    const pickFolder = () => { radios.folder.checked = true; };
    const folderSel = h("select", { "aria-label": "Library folder", onfocus: pickFolder, onchange: pickFolder },
      folders.map((f) => h("option", { value: f }, `${f.split("/").join(" › ")} (${names.filter((n) => inFolder(folderOf[n], f)).length})`)));
    sets.push({ key: "folder", label: "The specimens in a library folder", extra: folderSel,
      names: () => names.filter((n) => inFolder(folderOf[n], folderSel.value)) });
  }
  sets.push({ key: "all", label: `Every specimen (${names.length})`, names: () => names });
  sets.push({ key: "pick", label: "None yet: tick them next", names: () => [], pick: true });
  const choices = copyOf ? null : h("div", { class: "field", role: "radiogroup", "aria-label": "Start with" },
    h("span", { class: "lbl" }, "Start with"),
    h("div", { class: "stack", style: { gap: "6px" } }, sets.map((s, i) => {
      radios[s.key] = h("input", { type: "radio", name: "project-start", value: s.key, checked: i === 0 });
      return h("div", { class: "row wrap" }, h("label", { class: "check" }, radios[s.key], s.label), s.extra || null);
    })));
  projectDialog({
    title: copyOf ? `Duplicate ${copyOf.name}` : "New project", confirm: copyOf ? "Duplicate" : "Make project",
    value: copyOf ? `${copyOf.name} copy` : "",
    intro: copyOf ? "The copy starts with the same specimens, comparison and filters."
      : "A project keeps a study's specimens together with how they are compared and filtered. It is kept in the workspace, "
        + "and its specimens can be changed at any time.",
    body: choices,
    submit: async (name) => {
      const start = copyOf ? null : sets.find((s) => radios[s.key].checked) || sets[0];
      const made = await api.post("/api/results/projects", copyOf
        ? { name, names: copyOf.names || [], by: copyOf.by || [], hidden: copyOf.hidden || [], left: copyOf.left || [] }
        : { name, names: start.names(), by, hidden: [], left: [] });
      adopt(made);
      toast(`Made the project ${made.name}.`);
      onMade(made, Boolean(start?.pick));
    },
  });
}

function renameProjectDialog(p, done) {
  projectDialog({
    title: `Rename ${p.name}`, value: p.name, confirm: "Rename",
    submit: async (name) => {
      Object.assign(p, await api.patch(`/api/results/projects/${p.id}`, { name }));
      done();
    },
  });
}

/** Ask, then remove a project; resolves true once it is gone. */
async function removeProjectConfirm(p) {
  const ok = await confirmDialog(`Remove the project ${p.name}?`,
    "Its specimens, results and flags stay as they are; only the project's list, comparison and filters go.", "Remove project", { danger: true });
  if (!ok) return false;
  try {
    await api.del(`/api/results/projects/${p.id}`);
    const settings = state.app?.workspace?.settings;
    if (settings) settings.results_projects = projects().filter((x) => x.id !== p.id);
    toast(`Removed the project ${p.name}.`);
    return true;
  } catch (err) { errorToast(err); return false; }
}

// ---------------------------------------------------------------- the projects page
// Where the Results page opens: every specimen, then each project as a card
// with what it holds and how it compares them. A card opens its results;
// its buttons edit it here.
function projectsPage(root) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  document.title = "Results";
  const sub = h("div", { class: "sub" });
  page.appendChild(h("div", { class: "page-head" }, h("div", null, h("h1", null, "Results"), sub),
    h("div", { class: "actions" }, h("button", { class: "btn primary", onclick: () => create() }, icon("plus"), "New project"))));
  const body = h("div");
  page.appendChild(body);

  const everyone = () => [...state.samples.values()].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const lower = (k) => factor(k)?.label.toLowerCase() || k;
  const compared = (by) => (by.length ? `Compared by ${lower(by[0])}${by.length > 1 ? `, split by ${by.slice(1).map(lower).join(" and ")}` : ""}` : null);
  const swatches = (list) => {
    const groups = [...new Map(list.map((s) => [s.group, s])).values()].sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99));
    return h("div", { class: "pc-groups" }, groups.slice(0, 8).map((g) => h("span", { class: "group-label" },
      h("i", { class: "sw", style: { background: slotColor(g.slot) } }), g.group)), groups.length > 8 ? h("span", { class: "muted" }, `+${groups.length - 8}`) : null);
  };

  function create() {
    newProjectDialog({ everyone: everyone(), onMade: (made, pick) => (pick ? editMembers(made, true) : go(resultsHref(made.id))) });
  }

  // Tick the project's specimens; a project just made opens once they are chosen.
  async function editMembers(p, thenOpen = false) {
    const list = everyone();
    const by = validByFor(p.by, list.filter((s) => (p.names || []).includes(s.name)).length ? list.filter((s) => p.names.includes(s.name)) : list);
    const order = levelOrder(list);
    const levels = (order[by[0]] || []).filter(Boolean);
    const chosen = await specimenChecklist({
      specimens: list, ticked: new Set(p.names || []), by, order,
      colorOf: (lv) => (lv && groupColor(by[0], lv)) || slotColor(levels.includes(lv) ? levels.indexOf(lv) : null),
      title: `Specimens in ${p.name}`, confirm: thenOpen ? "Save and open" : "Save",
      intro: "The project holds the specimens ticked, for every browser that opens the workspace. A specimen added to the library later joins it once ticked here.",
    });
    if (!chosen) { render(); return; }
    try {
      await api.patch(`/api/results/projects/${p.id}`, { names: [...chosen], left: (p.left || []).filter((n) => chosen.has(n)) });
      if (thenOpen) go(resultsHref(p.id)); else render();
    } catch (err) { errorToast(err); }
  }

  function card({ id, name, members, by, filtered, foot, actions, explain }) {
    const href = resultsHref(id);
    const last = lastOpened() === id;
    return h("div", { class: `project-card${last ? " last" : ""}`, onclick: () => go(href) },
      h("div", { class: "pc-head" }, h("a", { class: "pc-name", href, onclick: (e) => e.stopPropagation() }, name),
        last ? h("span", { class: "tag accent" }, "Last opened") : null),
      h("div", { class: "pc-facts" }, [plural(members.length, "specimen"), explain, compared(by), filtered ? "filtered" : null].filter(Boolean).join(" · ")),
      members.length ? swatches(members) : null,
      h("div", { class: "pc-foot" }, h("span", { class: "muted" }, foot), h("span", { class: "spacer" }),
        actions ? h("div", { class: "pc-actions", onclick: (e) => e.stopPropagation() }, actions) : null));
  }

  const render = rafThrottle(() => {
    const list = everyone();
    const have = new Map(list.map((s) => [s.name, s]));
    const mine = projects();
    sub.textContent = `${plural(list.length, "specimen")}, ${mine.length ? plural(mine.length, "project") : "no projects yet"}. Choose what to compare.`;
    const ownScope = loadScope();
    const cards = [card({
      id: ALL, name: "All specimens", members: list, by: validByFor(loadView().by, list),
      explain: "every one in the workspace", filtered: (ownScope.left || []).length || (ownScope.hidden || []).length,
      foot: "Its filters last as long as the browser tab",
    }), ...mine.map((p) => {
      const members = (p.names || []).filter((n) => have.has(n)).map((n) => have.get(n));
      const button = (label, onclick, cls = "") => h("button", { class: `btn small ghost ${cls}`, onclick }, label);
      return card({
        id: p.id, name: p.name, members, by: (p.by || []).length ? p.by : (members.length ? validByFor([], members) : []),
        filtered: (p.hidden || []).length || (p.left || []).some((n) => have.has(n)),
        foot: `Updated ${relTime(p.updated)}`,
        actions: [
          button("Specimens…", () => editMembers(p)),
          button("Rename", () => renameProjectDialog(p, render)),
          button("Duplicate", () => newProjectDialog({ everyone: list, copyOf: p, onMade: render })),
          button("Remove", async () => { if (await removeProjectConfirm(p)) render(); }, "danger"),
        ],
      });
    })];
    replace(body, h("div", { class: "project-grid" }, cards,
      h("button", { class: "project-card add", onclick: () => create() }, icon("plus"),
        h("span", null, h("b", null, "New project"),
          h("span", { class: "muted" }, mine.length ? "Another study, or the same specimens compared another way."
            : "Keep a study's specimens, with how they are compared and filtered, one click away.")))));
  });

  const offs = [on("settings", render), on("samples", render), on("app", render)];
  render();
  return () => offs.forEach((f) => f());
}
