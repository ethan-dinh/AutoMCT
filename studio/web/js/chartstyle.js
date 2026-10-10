// The Results page's chart colours and titles. One set per workspace, kept in
// its studio.json (results_style) so every browser that opens the workspace
// draws the same charts and exports the same figures. Anything left unset
// follows the page's theme on screen and Prism's black on white in a figure.
//
//   groups     {"genotype\u0001KO": "#rrggbb"}    a level of the detail compared
//   landmarks  {M1: "#rrggbb", ...}
//   marks      {mean: "#rrggbb", ...}             every other line, mark, axis and text (MARKS)
//   titles     {"molar-density": "..."}           a card's title, on the page and on its figure
//   labels     {"Density (mg HA/cm³)": "..."}     a figure's axis title, keyed by the one it
//                                                 replaces, so a switch that changes the quantity
//                                                 (units, statistic) never keeps a stale title

import { h, debounce, errorToast, cssColor } from "./lib.js";
import { api } from "./api.js";
import { state, on } from "./store.js";

const SECTIONS = ["groups", "landmarks", "marks", "titles", "labels"];
const HEX = /^#[0-9a-f]{6}$/i;

/** The landmarks the incisor charts mark: [name, label, --lm-N tone]. */
export const LANDMARKS = [["M1", "M1", 1], ["M2", "M2", 2], ["M3", "M3", 3], ["eruption", "Eruption", 4]];

/**
 * Every other aspect a colour can be picked for. `where`: "figures" or "page"
 * when only one draws it; `group`: unset, each group's line takes its group's
 * colour; `fallback`: the colour the editor shows while it is unset.
 */
export const MARKS = [
  { id: "mean", label: "Mean lines (dot plots)", kind: "lines", fallback: "#000000" },
  { id: "error", label: "Error bars (SD or SEM)", kind: "lines", fallback: "#000000" },
  { id: "specimen", label: "Specimen curves", kind: "lines", group: true },
  { id: "ribbon", label: "±1 SD bands", kind: "lines", group: true },
  { id: "dentin", label: "Dentin line", kind: "lines", group: true },
  { id: "rule", label: "Reference line (ratio 1)", kind: "lines", fallback: "#6e6e6e" },
  { id: "brackets", label: "Significance brackets and stars", kind: "lines", where: "figures", fallback: "#000000" },
  { id: "title", label: "Figure title", kind: "text", where: "figures", fallback: "#000000" },
  { id: "axisTitle", label: "Axis titles", kind: "text", fallback: "#000000" },
  { id: "text", label: "Tick labels, names and legend", kind: "text", fallback: "#000000" },
  { id: "axes", label: "Axes and ticks", kind: "text", fallback: "#000000" },
  { id: "grid", label: "Grid lines", kind: "text", where: "page", fallback: "var(--grid)" },
  { id: "background", label: "Background", kind: "text", where: "figures", fallback: "#ffffff" },
];

// The page's charts read these custom properties (charts.js, app.css), falling back to the theme's.
const PAGE_VARS = { mean: "--chart-mean", error: "--chart-error", rule: "--chart-rule", axisTitle: "--chart-axis-title",
  text: "--chart-text", axes: "--chart-axis", grid: "--chart-grid" };

// ---------------------------------------------------------------- the store

let current = null, currentRoot = null;
let edits = 0, saved = 0;          // a save is pending while these differ
const listeners = new Set();

const blank = () => Object.fromEntries(SECTIONS.map((k) => [k, {}]));
const workspaceRoot = () => state.app?.workspace?.root || "";

function fromSettings() {
  const stored = state.app?.workspace?.settings?.results_style || {};
  const out = blank();
  for (const k of SECTIONS) if (stored[k] && typeof stored[k] === "object") out[k] = { ...stored[k] };
  return out;
}

/** The current workspace's style. */
export function resultsStyle() {
  if (!current || workspaceRoot() !== currentRoot) { current = fromSettings(); currentRoot = workspaceRoot(); }
  return current;
}

function notify() {
  for (const fn of listeners) { try { fn(); } catch (err) { console.error(err); } }
}

const save = debounce(async () => {
  if (workspaceRoot() !== currentRoot) return;      // the workspace changed under the edit
  const sent = edits;
  try { await api.patch("/api/settings", { results_style: current }); } catch (err) { errorToast(err); }
  saved = sent;
}, 500);

function changed() {
  edits += 1;
  save();
  notify();
}

// Another browser's edits arrive with the workspace's settings; ours echo back and change nothing.
on("settings", () => {
  if (edits !== saved) return;
  const next = fromSettings();
  if (workspaceRoot() === currentRoot && JSON.stringify(next) === JSON.stringify(current)) return;
  current = next;
  currentRoot = workspaceRoot();
  notify();
});

/** Set one entry (`value` null or empty clears it back to the default). */
export function setStyle(section, id, value) {
  const s = resultsStyle()[section];
  const v = typeof value === "string" ? value.trim() : value;
  if (v == null || v === "") delete s[id]; else s[id] = v;
  changed();
}

/** Every colour back to its default; titles stay. */
export function resetColors() {
  const s = resultsStyle();
  s.groups = {}; s.landmarks = {}; s.marks = {};
  changed();
}

/** Runs `fn` after every change, from this page or another browser. Returns the unsubscribe. */
export function onStyle(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// ---------------------------------------------------------------- reading it

const hex = (v) => (HEX.test(v || "") ? v : null);
export const groupKey = (factorKey, level) => `${factorKey}\u0001${level}`;
export const groupColor = (factorKey, level) => hex(resultsStyle().groups[groupKey(factorKey, level)]);
export const landmarkColor = (name) => hex(resultsStyle().landmarks[name]);
export const markColor = (id) => hex(resultsStyle().marks[id]);
export const titleOf = (key) => resultsStyle().titles[key] || null;
export const labelOf = (text) => (text && resultsStyle().labels[text]) || text;

/** The colours set for a figure's own marks (figure.js `spec.ink`): only the ones picked. */
export function figureInk() {
  const out = {};
  for (const m of MARKS) {
    const c = !m.group && m.where !== "page" ? markColor(m.id) : null;
    if (c) out[m.id] = c;
  }
  return out;
}

/** The page's chart colours, as custom properties on `el` (set, or removed for the theme's). */
export function applyPageInk(el) {
  for (const [id, prop] of Object.entries(PAGE_VARS)) {
    const c = markColor(id);
    if (c) el.style.setProperty(prop, c); else el.style.removeProperty(prop);
  }
}

// ---------------------------------------------------------------- the editor

let probe = null;
/** Any CSS colour (a var() included) as #rrggbb, for a colour input. */
function toHex(color) {
  const value = cssColor(color);
  if (HEX.test(value)) return value.toLowerCase();
  probe ||= document.createElement("canvas").getContext("2d");
  probe.fillStyle = "#000000";
  probe.fillStyle = value;
  return HEX.test(probe.fillStyle) ? probe.fillStyle : "#808080";
}

/**
 * A swatch for every aspect, each changing the charts as it moves, with a way
 * back to its default. `groups`: [{id, label, fallback}], the compared
 * detail's levels (`id` from groupKey, `fallback` their CSS colour), under
 * `heading`. `onChange` runs after each change.
 */
export function colorEditor({ groups = [], heading = "Groups", onChange = () => {} } = {}) {
  const rows = [];
  const row = (section, id, label, { fallback, group = false, where = null }) => {
    const custom = () => hex(resultsStyle()[section][id]);
    const input = h("input", { type: "color", "aria-label": label });
    const status = h("span", { class: "cs-state muted" });
    const reset = h("button", { type: "button", class: "btn small ghost", title: "Back to the default colour",
      onclick: () => { setStyle(section, id, null); sync(); onChange(); } }, "Reset");
    const wrap = h("div", { class: "cs-row" },
      h("label", null, input, h("span", { class: "cs-label" }, label,
        where ? h("span", { class: "cs-where" }, where === "figures" ? "figures only" : "page only") : null)),
      status, reset);
    const sync = () => {
      const c = custom();
      input.value = c || (group ? "#808080" : toHex(fallback));
      status.textContent = c ? "" : group ? "Group colours" : "Default";
      reset.hidden = !c;
      wrap.classList.toggle("is-default", !c);
    };
    input.addEventListener("input", () => { setStyle(section, id, input.value.toLowerCase()); sync(); onChange(); });
    sync();
    rows.push(sync);
    return wrap;
  };
  const block = (title, items, help) => h("section", { class: "cs-section" },
    h("h4", null, title), help ? h("p", { class: "muted cs-help" }, help) : null,
    h("div", { class: "cs-grid" }, items));
  const marks = (kind) => MARKS.filter((m) => m.kind === kind).map((m) => row("marks", m.id, m.label, m));
  return h("div", { class: "color-editor" },
    groups.length ? block(heading, groups.map((g) => row("groups", g.id, g.label, { fallback: g.fallback })),
      "Each group's points, means, bands and legend, on the page and in every figure.") : null,
    block("Landmarks", LANDMARKS.map(([id, label, tone]) => row("landmarks", id, label, { fallback: `var(--lm-${tone})` }))),
    block("Lines and marks", marks("lines"), "Specimen curves, bands and the dentin line take each group's colour until one is picked."),
    block("Axes and text", marks("text"), "Left unset, the page keeps its theme's greys and a figure prints black on white."),
    h("div", { class: "row cs-foot" },
      h("button", { type: "button", class: "btn small ghost", onclick: () => { resetColors(); rows.forEach((sync) => sync()); onChange(); } },
        "Reset all colours"),
      h("span", { class: "muted" }, "Saved with the workspace.")));
}
