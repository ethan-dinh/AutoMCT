// App shell: the rail, the router, and the global drop target.

import { h, icon, replace, errorToast, clear, rafThrottle } from "./lib.js";
import { state, on, loadAll, start, runningCount, hasWorkspace } from "./store.js";
import { filesFromDrop } from "./api.js";

const NAV = [
  ["#/", "Library", "library", /^#\/?(\?|$)|^#\/sample/],
  ["#/import", "Import", "import", /^#\/import/],
  ["#/viewer", "Viewer", "viewer", /^#\/(viewer|crop)/],
  ["#/compare", "Compare", "compare", /^#\/compare/],
  ["#/results", "Results", "results", /^#\/results/],
  ["#/queue", "Queue", "queue", /^#\/queue/],
  ["#/tools", "Tools", "tools", /^#\/tools/],
  ["#/settings", "Settings", "settings", /^#\/settings/],
];

const ROUTES = [
  [/^#\/?$/, () => import("./views/library.js")],
  [/^#\/import$/, () => import("./views/importer.js")],
  [/^#\/sample\/([^/?]+)$/, () => import("./views/sample.js")],
  [/^#\/viewer(?:\/([^/?]+))?(?:\/([a-z]+))?$/, () => import("./views/viewer.js")],
  [/^#\/crop\/([^/?]+)$/, () => import("./views/crop.js")],
  [/^#\/compare$/, () => import("./views/compare3d.js")],
  [/^#\/results(?:\/([^/?]+))?$/, () => import("./views/results.js")],
  [/^#\/queue(?:\/([^/?]+))?$/, () => import("./views/queue.js")],
  [/^#\/tools$/, () => import("./views/tools.js")],
  [/^#\/settings$/, () => import("./views/settings.js")],
  [/^#\/workspaces$/, () => import("./views/workspaces.js")],
];

// Files dropped outside the import page wait here for it to pick them up.
export const pending = { files: null };

let cleanup = null;
let routeToken = 0;

const renderRail = rafThrottle(() => {
  const links = document.getElementById("rail-links");
  const hash = location.hash || "#/";
  const busy = runningCount();
  replace(links, NAV.map(([href, label, ic, match]) =>
    h("a", { href, "aria-current": match.test(hash) ? "page" : null },
      icon(ic), h("span", { class: "t" }, label),
      label === "Queue" && busy ? h("span", { class: "count", "aria-label": `${busy} active` }, busy) : null)));
  const foot = document.getElementById("rail-foot");
  const ws = state.app?.workspace;
  replace(foot,
    h("span", { class: "muted" }, "Workspace"),
    ws ? h("a", { class: "ws", href: "#/workspaces", title: `${ws.root}\nSwitch workspace` }, ws.name) : null,
    h("span", { class: `conn ${state.connected ? "" : "down"}` }, state.connected ? "Connected" : "Reconnecting…"));
});

async function route() {
  const token = ++routeToken;
  const full = location.hash || "#/";
  const [path, queryString] = full.split("?");
  const query = Object.fromEntries(new URLSearchParams(queryString || ""));
  // With no workspace open, or its drive unplugged, the only page is the one
  // that opens a workspace, and it has the window to itself.
  const bare = !hasWorkspace();
  document.querySelector(".shell").classList.toggle("bare", bare);
  const target = bare ? "#/workspaces" : path;
  renderRail();
  for (const [pattern, loader] of ROUTES) {
    const m = pattern.exec(target);
    if (!m) continue;
    const mod = await loader();
    if (token !== routeToken) return;
    if (cleanup) { try { cleanup(); } catch (err) { console.error(err); } cleanup = null; }
    const main = document.getElementById("view");
    clear(main);
    window.scrollTo(0, 0);
    try {
      cleanup = await mod.mount(main, m.slice(1).map((p) => (p ? decodeURIComponent(p) : p)), query) || null;
    } catch (err) {
      console.error(err);
      main.appendChild(h("div", { class: "page" }, h("div", { class: "notice critical" }, err.message || String(err))));
    }
    return;
  }
  location.hash = "#/";
}

export function go(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

function globalDrop() {
  let depth = 0, veil = null;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e) || location.hash.startsWith("#/import")) return;
    depth++;
    if (!veil) { veil = h("div", { class: "drop-veil" }, "Drop to import"); document.body.appendChild(veil); }
  });
  window.addEventListener("dragleave", () => {
    if (depth > 0 && --depth === 0 && veil) { veil.remove(); veil = null; }
  });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", async (e) => {
    if (!hasFiles(e) || location.hash.startsWith("#/import")) return;
    e.preventDefault();
    depth = 0;
    if (veil) { veil.remove(); veil = null; }
    pending.files = await filesFromDrop(e.dataTransfer);
    // Dropped on a folder of the library, the scan is filed in it.
    const [path, query] = location.hash.split("?");
    const folder = /^#\/?$/.test(path || "#/") ? new URLSearchParams(query || "").get("folder") : null;
    go(folder ? `#/import?folder=${encodeURIComponent(folder)}` : "#/import");
  });
}

function keys() {
  window.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "/") {
      const search = document.querySelector("input[type=search]");
      if (search) { e.preventDefault(); search.focus(); }
    }
  });
}

const workspaceKey = () => { const ws = state.app?.workspace; return ws ? `${ws.root}|${ws.available}` : ""; };

async function boot() {
  try {
    await loadAll();
  } catch (err) {
    errorToast(err);
  }
  start();
  let shownFor = workspaceKey();
  on("jobs", renderRail);
  on("connection", renderRail);
  on("app", () => {
    renderRail();
    // Another workspace opened (here or in another tab), or its drive went away.
    if (workspaceKey() !== shownFor) { shownFor = workspaceKey(); route(); }
  });
  window.addEventListener("studio:workspace-lost", () => loadAll().catch(() => {}));
  window.addEventListener("hashchange", route);
  globalDrop();
  keys();
  route();
}

boot();
