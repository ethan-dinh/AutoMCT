// Tools: the repository's standalone utilities, each with a form and its runs.

import { h, icon, replace, statusBadge, relTime, errorToast, toast, fmtBytes } from "../lib.js";
import { api } from "../api.js";
import { state, on } from "../store.js";
import { paramsForm } from "../components.js";
import { go } from "../app.js";

export function mount(root) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  page.appendChild(h("div", { class: "page-head" }, h("div", null, h("h1", null, "Tools"),
    h("div", { class: "sub" }, "Utilities that work on files directly, outside any one specimen."))));
  const panel = h("div", { class: "panel" });
  page.appendChild(panel);
  const forms = {};
  const runSlots = {};
  const outputs = new Map();

  async function run(tool) {
    try {
      const job = await api.post(`/api/tools/${tool.id}`, { params: forms[tool.id].get() });
      toast(`Started ${tool.label.toLowerCase()}.`, "good", { action: { label: "Open log", onClick: () => go(`#/queue/${job.id}`) } });
    } catch (err) { errorToast(err); }
  }

  async function renderRuns(tool) {
    const slot = runSlots[tool.id];
    const jobs = [...state.jobs.values()].filter((j) => j.step === `tool:${tool.id}`)
      .sort((a, b) => (a.created < b.created ? 1 : -1)).slice(0, 3);
    if (!jobs.length) { replace(slot); return; }
    const items = [];
    for (const j of jobs) {
      if (j.status === "done" && !outputs.has(j.id)) {
        try { outputs.set(j.id, (await api.get(`/api/jobs/${j.id}`)).outputs); } catch { outputs.set(j.id, []); }
      }
      const outs = outputs.get(j.id) || [];
      items.push(h("div", { class: "stack", style: { gap: "6px" } },
        h("div", { class: "row wrap" }, statusBadge(j.status, null, j.progress), h("span", { class: "muted" }, relTime(j.finished || j.created)),
          h("a", { href: `#/queue/${j.id}` }, "Log")),
        outs.filter((o) => /\.png$/i.test(o.name)).slice(0, 4).map((o) =>
          h("img", { src: `/api/jobs/${j.id}/output?name=${encodeURIComponent(o.name)}`, alt: o.name, loading: "lazy" })),
        outs.length ? h("div", { class: "row wrap" }, outs.map((o) =>
          h("a", { class: "btn small ghost", href: `/api/jobs/${j.id}/output?name=${encodeURIComponent(o.name)}&download=1` },
            icon("download"), `${o.name} (${fmtBytes(o.size)})`))) : null));
    }
    replace(slot, h("div", { class: "tool-runs" }, items));
  }

  function render() {
    const tools = state.app?.tools || [];
    replace(panel, tools.map((tool) => {
      forms[tool.id] ||= paramsForm(tool.params, {});
      runSlots[tool.id] ||= h("div");
      // A tool that is switched off stays on the page, greyed, and cannot run.
      const off = Boolean(tool.disabled);
      return h("div", { class: off ? "tool disabled" : "tool", "aria-disabled": off ? "true" : null },
        h("div", { class: "tool-head" }, h("h2", null, tool.label), off ? statusBadge("disabled") : null, h("p", null, tool.summary)),
        h("fieldset", { class: "plain", disabled: off, style: { marginTop: "12px" } }, forms[tool.id].el),
        h("div", { class: "row wrap", style: { marginTop: "12px" } },
          h("button", { class: "btn primary", disabled: off, onclick: () => run(tool) }, icon("play"), "Run"),
          tool.id === "ml_train" ? h("button", { class: "btn", disabled: off, onclick: buildTrainingSet }, "Build the training set from this workspace") : null,
          off ? h("span", { class: "off-note" }, tool.disabled) : null),
        runSlots[tool.id]);
    }));
    tools.forEach(renderRuns);
  }

  async function buildTrainingSet() {
    try {
      const res = await api.post("/api/tools/training_set");
      toast(`Training set: ${res.samples.length} specimen${res.samples.length === 1 ? "" : "s"} linked in ${res.folder}.` +
        (res.skipped.length ? ` ${res.skipped.length} skipped (segment them with TIFF output to include them).` : ""), "good", { timeout: 10000 });
    } catch (err) { errorToast(err); }
  }

  const off = on("jobs", (job) => {
    if (job && job.step?.startsWith("tool:")) {
      const tool = (state.app?.tools || []).find((t) => `tool:${t.id}` === job.step);
      if (tool) renderRuns(tool);
    }
  });
  render();
  return off;
}
