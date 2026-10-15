// Settings: the workspace, the queue, previews, appearance and defaults.

import { h, icon, replace, errorToast, toast, slotColor, statusBadge } from "../lib.js";
import { api } from "../api.js";
import { state, on, loadAll } from "../store.js";
import { paramsForm } from "../components.js";

const BUDGETS = [[8_000_000, "Fastest (8M voxels)"], [24_000_000, "Balanced (24M voxels)"], [64_000_000, "Sharpest (64M voxels)"]];

export function mount(root) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  page.appendChild(h("div", { class: "page-head" }, h("div", null, h("h1", null, "Settings"))));
  const list = h("div", { class: "panel settings-list" });
  page.appendChild(list);

  const patch = async (body, message) => {
    try { await api.patch("/api/settings", body); if (message) toast(message); await loadAll(); }
    catch (err) { errorToast(err); }
  };

  function set(title, text, ...controls) {
    return h("div", { class: "set" }, h("div", null, h("h3", null, title), text ? h("p", null, text) : null), h("div", { class: "stack" }, controls));
  }

  function render() {
    const app = state.app;
    if (!app) return;
    const s = app.workspace.settings;
    const reveal = { darwin: "Show in Finder", win32: "Show in Explorer" }[app.platform] || "Show folder";
    const sharing = app.workspace.sharing;
    const theme = (() => { try { return localStorage.getItem("studio.theme") || "system"; } catch { return "system"; } })();
    const groups = Object.entries(s.group_colors || {}).sort((a, b) => a[1] - b[1]);
    const defaults = app.steps.filter((st) => st.params.length).map((st) => {
      if (st.disabled) {
        return h("div", { class: "row", style: { padding: "4px 0" } },
          h("span", { style: { color: "var(--ink-3)" } }, st.label), statusBadge("disabled"));
      }
      const form = paramsForm(st.params, (s.step_defaults || {})[st.id] || {});
      return h("details", null, h("summary", { style: { cursor: "pointer", padding: "4px 0" } }, st.label),
        h("div", { class: "params-box", style: { marginTop: "8px" } }, form.el,
          h("div", { class: "row", style: { marginTop: "10px" } },
            h("button", { class: "btn small primary", onclick: () => patch({ step_defaults: { [st.id]: form.get() } }, `Saved defaults for ${st.label}.`) }, "Save defaults"),
            h("button", { class: "btn small ghost", onclick: () => patch({ step_defaults: { [st.id]: {} } }, `Restored the tool's own defaults for ${st.label}.`) }, "Use the tool's defaults"))));
    });

    replace(list,
      set("Workspace", "The folder that holds your specimens, their results and these settings. On an external drive "
        + "formatted as exFAT it opens from both a Mac and a PC. Avoid synced folders such as Box or Dropbox: previews and jobs write often.",
        h("div", { class: "stack", style: { gap: "2px" } }, h("b", null, app.workspace.name), h("code", { class: "soft" }, app.workspace.root)),
        sharing?.text ? h("div", { class: `ws-note ${sharing.tone}` }, icon(sharing.tone === "good" ? "checkCircle" : sharing.tone === "warning" ? "alert" : "info"),
          h("span", null, sharing.text)) : null,
        h("div", { class: "row wrap" },
          h("a", { class: "btn", href: "#/workspaces" }, icon("folder"), "Switch workspace"),
          app.remote ? null : h("button", { class: "btn ghost", onclick: () => api.post("/api/reveal", { path: app.workspace.root }).catch(errorToast) }, reveal))),
      set("Jobs at once", "Conversion, segmentation, CMPR and quantification each hold whole scans in memory; two at once can exhaust a 16-24 GB machine.",
        ["heavy", "preview", "light"].map((lane) => h("label", { class: "row" },
          h("span", { style: { width: "150px" } }, { heavy: "Heavy jobs", preview: "Preview builds", light: "Small jobs" }[lane]),
          h("select", { onchange: (e) => patch({ lanes: { [lane]: Number(e.target.value) } }, "Saved.") },
            [1, 2, 3, 4, 6, 8].map((n) => h("option", { value: n, selected: Number(s.lanes?.[lane] || 1) === n }, n)))))),
      set("Viewer previews", "Each volume is shown from a downsampled copy built once. A bigger budget is sharper and takes longer to build and load.",
        h("select", { onchange: (e) => patch({ preview_budget_voxels: Number(e.target.value) }, "Saved. Rebuild a view to apply it.") },
          BUDGETS.map(([v, l]) => h("option", { value: v, selected: Number(s.preview_budget_voxels) === v }, l))),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: s.auto_preview !== false,
          onchange: (e) => patch({ auto_preview: e.target.checked }, "Saved.") }), "Build previews automatically after each step")),
      set("Appearance", null,
        h("div", { class: "seg" }, [["system", "Match the system"], ["light", "Light"], ["dark", "Dark"]].map(([v, l]) =>
          h("button", { "aria-pressed": theme === v ? "true" : "false", onclick: () => {
            try { if (v === "system") localStorage.removeItem("studio.theme"); else localStorage.setItem("studio.theme", v); } catch { /* private window */ }
            if (v === "system") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = v;
            render();
          } }, l)))),
      set("Group colours", "Each group keeps its colour in every chart, so a filter never repaints the survivors.",
        groups.length ? groups.map(([g, slot]) => h("label", { class: "row" },
          h("i", { class: "sw", style: { background: slotColor(slot), width: "14px", height: "14px" } }),
          h("span", { style: { width: "180px" } }, g),
          h("select", { onchange: (e) => patch({ group_colors: { [g]: Number(e.target.value) } }, "Saved.") },
            Array.from({ length: 8 }, (_, i) => h("option", { value: i, selected: slot === i }, `Colour ${i + 1}`)))))
          : h("span", { class: "muted" }, "Groups appear once specimens have a genotype and age.")),
      set("Default options", "Used whenever a step runs without options of its own. The tools' own defaults apply until you change them here.", defaults),
      set("About", null,
        h("div", { class: "soft" }, `MicroCT Studio ${app.version}`),
        h("code", { class: "soft" }, `Python: ${app.python}`),
        h("code", { class: "soft" }, `Repository: ${app.repo}`)));
  }

  const offs = [on("app", render), on("settings", render)];
  render();
  return () => offs.forEach((f) => f());
}
