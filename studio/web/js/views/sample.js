// One specimen: its pipeline, its results, its views and files.

import {
  h, icon, replace, fmtPct, pctFmt, fmtBytes, fmtDuration, sig, relTime, statusBadge, statusInfo, errorToast, toast,
  modal, confirmDialog, debounce, slotColor, basename, elapsedSince, ordinal, toMgHA, calibrationFormula, numFmt,
} from "../lib.js";
import { api, enc } from "../api.js";
import { state, on, livePipeline, previewJob } from "../store.js";
import { paramsForm, metaForm, META_LABELS, stepSpec, flagDialog, folderCrumbs, folderHref, pickFolder, leafOf } from "../components.js";
import { exclusionText } from "../compare.js";
import { lineChart, barChart, histogramChart, chartCard, dataTable, landmarkTone } from "../charts.js";
import { MEASURABLE, smoothProfile, sliceStep, medianWindow } from "../trajectory.js";
import { go } from "../app.js";

const STATUS_WORDS = {
  resolved_hi: "denser than dentin", resolved_lo: "less dense than dentin", coincident: "same as dentin",
  saturated: "at the scan ceiling", insufficient: "too few voxels", unresolvable: "mostly rim", pre_onset: "before enamel onset",
};
// The enamel is drawn from the slices the server's cleanup kept. The dentin
// peak is fitted wherever a slice resolved, and dentin is there before enamel
// starts, so its line runs from the cervical loop.
const DENTIN_MEASURED = [...MEASURABLE, "saturated", "pre_onset"];

export async function mount(root, [name]) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  let detail = null, marks = null, molar = null, incisor = null;
  // The density profile's units (null: mg HA/cm³ when the scan is calibrated) and dentin line.
  const densityView = { units: null, dentin: true };
  const openOptions = new Set();
  const forms = {};

  const load = async () => {
    try {
      detail = await api.get(`/api/samples/${enc(name)}`);
    } catch (err) {
      replace(page, h("div", { class: "empty" }, h("h3", null, err.status === 404 ? `There is no specimen named ${name}` : err.message),
        h("a", { class: "btn", href: "#/" }, "Back to the library")));
      return false;
    }
    const arts = new Set(detail.artifacts.filter((a) => a.exists).map((a) => a.key));
    const tasks = [];
    tasks.push(arts.has("molar_quant") ? api.get(`/api/samples/${enc(name)}/molar`).then((r) => { molar = r; }) : Promise.resolve(molar = null));
    tasks.push(arts.has("incisor_profile") ? api.get(`/api/samples/${enc(name)}/incisor`).then((r) => { incisor = r; }) : Promise.resolve(incisor = null));
    tasks.push(arts.has("landmark_marks") || arts.has("landmarks") || arts.has("landmark_report") ? api.get(`/api/samples/${enc(name)}/marks`).then((r) => { marks = r; }) : Promise.resolve(marks = null));
    await Promise.all(tasks.map((t) => t.catch(() => null)));
    return true;
  };

  // ------------------------------------------------------------ head
  function head() {
    const s = detail.summary, m = s.meta || {};
    const facts = ["age", "line", "genotype", "sex", "replicate", "side", "study", "scan_date", "scanner"]
      .filter((k) => m[k] != null && m[k] !== "")
      .map((k) => h("span", null, `${META_LABELS[k]} `, h("b", null, k === "side" ? { L: "Left", R: "Right" }[m[k]] || m[k] : k === "sex" ? { F: "Female", M: "Male" }[m[k]] || m[k] : String(m[k]))));
    const src = detail.record.source || {};
    return h("div", { class: "sample-head" },
      h("div", null,
        h("div", { class: "sample-where" }, folderCrumbs(s.folder || ""),
          h("button", { class: "btn small ghost", onclick: moveTo, title: "File it in another folder of the library" }, "Move…")),
        h("h1", null, name),
        h("div", { class: "sample-meta" },
          h("span", { class: "group-label" }, h("i", { class: "sw", style: { background: slotColor(s.slot) } }), h("b", null, s.group)),
          facts),
        src.path ? h("div", { class: "sample-meta" }, h("span", null, "Source ", h("code", { title: src.path }, basename(src.path)),
          src.mode === "link" ? " (linked, not copied)" : "")) : null,
        storageLine(),
        m.notes ? h("p", { class: "soft", style: { marginTop: "8px" } }, m.notes) : null,
        flagNotice(s.excluded)),
      h("div", { class: "row wrap" },
        h("a", { class: "btn primary", href: `#/viewer/${enc(name)}` }, icon("viewer"), "Open viewer"),
        h("button", { class: "btn", onclick: editDetails }, icon("edit"), "Edit details"),
        s.excluded ? null : h("button", { class: "btn", onclick: () => flag(null), title: "Leave it out of the results, with a note" }, icon("flag"), "Flag"),
        h("button", { class: "btn", title: "Show the sample folder in Finder", onclick: () => api.post("/api/reveal", { path: detail.folder }).catch(errorToast) }, icon("folder"), "Show folder"),
        detail.artifacts.some((a) => a.key === "reoriented" && a.exists)
          ? h("button", { class: "btn", title: "Open the full-resolution scan in napari", onclick: () =>
            api.post("/api/napari", { sample: name, artifact: "reoriented" }).then(() => toast("Opening napari with the full-resolution scan…", "info")).catch(errorToast) },
          icon("external"), "napari") : null,
        h("button", { class: "btn ghost danger", onclick: remove }, icon("trash"), "Remove")));
  }

  // A flagged specimen says so at the top of its page, with why and a way back.
  function flagNotice(excluded) {
    if (!excluded) return null;
    return h("div", { class: "notice warning flag-note" }, icon("flag"),
      h("div", { class: "grow" },
        h("b", null, `Flagged and left out of ${exclusionText(excluded)}. `),
        excluded.note ? excluded.note : h("span", { class: "muted" }, "No note."),
        excluded.at ? h("span", { class: "muted" }, ` Flagged ${relTime(excluded.at)}.`) : null),
      h("button", { class: "btn small", onclick: () => flag(excluded) }, "Edit"),
      h("button", { class: "btn small", onclick: () => api.post("/api/flags", { names: [name], flag: null })
        .then(() => { toast(`${name} is back in the results.`); refresh(true); }).catch(errorToast) }, "Include again"));
  }

  // Folders are the library's: moving one changes nothing on disk.
  async function moveTo() {
    const here = detail.summary.folder || "";
    const dest = await pickFolder({ title: `Move ${name}`, current: here });
    if (dest == null || dest === here) return;
    try {
      const res = await api.post("/api/folders/move", { names: [name], folder: dest });
      toast(`Moved ${name} to ${res.folder ? leafOf(res.folder) : "the top of the library"}.`);
      refresh(true);
    } catch (err) { errorToast(err); }
  }

  async function flag(current) {
    const outcome = await flagDialog([name], current);
    if (outcome === "flagged") toast(`${name} is flagged and left out of the results.`);
    else if (outcome === "included") toast(`${name} is back in the results.`);
    if (outcome) refresh(true);
  }

  // Where the large files (the converted scan, the reoriented mandible) are
  // kept: beside a linked scan, on its own drive, rather than in the workspace.
  function storageLine() {
    const st = detail.storage;
    if (!st) return null;
    const shown = `…/${st.path.split("/").slice(-3).join("/")}`;
    return h("div", { class: "stack", style: { gap: "6px", marginTop: "6px" } },
      h("div", { class: "sample-meta", style: { marginTop: 0 } },
        h("span", null, "Large files beside the scan, in ", h("code", { title: st.path }, shown),
          st.exists ? [" ", h("button", { class: "btn small ghost", style: { verticalAlign: "middle" }, title: st.path,
            onclick: () => api.post("/api/reveal", { path: st.path }).catch(errorToast) }, icon("folder"), "Show")]
            : " (made on the first conversion or segmentation)")),
      st.synced ? h("div", { class: "notice warning" },
        `This folder syncs to ${st.synced}, so the large files written here upload to it too.`) : null);
  }

  async function editDetails() {
    // Suggest the values the other specimens use, as the import form does.
    const suggestions = await api.get("/api/meta/values").catch(() => ({}));
    const nameInput = h("input", { type: "text", value: name });
    // The name the details give, offered rather than applied: renaming also
    // renames its files. A specimen named before names carried the sex, say.
    const offer = h("div", { class: "name-offer" });
    const suggest = debounce(async () => {
      try {
        const res = await api.post("/api/names/compose", { meta: form.get(), found: name, current: name });
        replace(offer, res.name && res.from === "details" && res.name !== nameInput.value.trim() ? [
          h("span", { class: "muted" }, "Named from these details: "), h("b", { class: "specimen" }, res.name),
          h("button", { type: "button", class: "btn small", onclick: () => { nameInput.value = res.name; replace(offer); } }, "Use this name"),
        ] : null);
      } catch { replace(offer); }
    }, 200);
    const form = metaForm(detail.record.meta || {}, suggestions, suggest);
    nameInput.addEventListener("input", suggest);
    suggest();
    modal({
      title: "Edit details", wide: true,
      body: h("div", { class: "stack" },
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Name"), nameInput,
          h("span", { class: "help" }, "Renaming also renames every file the pipeline wrote for this specimen.")),
        offer,
        form.el),
      actions: [
        { label: "Cancel" },
        { label: "Save", primary: true, onClick: async () => {
          try {
            const res = await api.patch(`/api/samples/${enc(name)}`, { meta: form.get(), name: nameInput.value.trim() });
            const newName = res.record.name;
            toast("Saved.");
            if (newName !== name) go(`#/sample/${encodeURIComponent(newName)}`); else refresh(true);
          } catch (err) { errorToast(err); return false; }
          return true;
        } },
      ],
    });
  }

  async function remove() {
    const beside = detail.storage?.exists ? detail.storage.path : null;
    const ok = await confirmDialog(`Remove ${name}?`,
      "Its folder moves to the workspace's trash folder, so it can be recovered. Linked data outside the workspace is not touched." +
      (beside ? ` The large files beside the scan stay where they are, in ${beside}; delete them there if you no longer need them.` : ""),
      "Remove", { danger: true });
    if (!ok) return;
    try {
      const res = await api.del(`/api/samples/${enc(name)}`);
      toast(`Moved ${name} to the trash folder.` + (res.left_beside_scan ? " Its large files are still beside the scan." : ""));
      go(folderHref(detail.summary.folder || ""));
    } catch (err) { errorToast(err); }
  }

  // ------------------------------------------------------------ pipeline
  async function run(stepIds, params = {}) {
    try {
      const res = await api.post("/api/run", { samples: [name], steps: stepIds, params });
      if (res.skipped.length) toast(res.skipped[0].reason, "warning");
      else toast(`Queued ${res.jobs.map((j) => j.title).join(", ")}.`, "good", { action: { label: "Open queue", onClick: () => go("#/queue") } });
    } catch (err) { errorToast(err); }
  }

  function stepRow(p) {
    const spec = stepSpec(p.id);
    if (p.status === "disabled") {
      return h("li", { class: "disabled", "aria-disabled": "true" },
        h("div", { class: "step-top" }, h("h3", null, p.label), statusBadge(p.status)),
        h("div", { class: "step-summary" }, spec?.summary),
        h("div", { class: "step-body" }, h("span", { class: "off-note" }, p.reason)));
    }
    const src = detail.summary.source || {};
    // The converter cuts the crop box out of every format but NRRD, which is
    // segmented as it is and cropped as it loads.
    const cropStep = src.needs_conversion ? "convert" : src.format === "nrrd" ? "segment" : null;
    const rec = (detail.record.steps || {})[p.id] || {};
    // A run again starts from the defaults as they are now, plus whatever the
    // last run changed from them; not from every value the last run used.
    const values = { ...(state.app.workspace.settings.step_defaults?.[p.id] || {}), ...(rec.custom || {}) };
    if (!forms[p.id] && spec?.params?.length) forms[p.id] = paramsForm(spec.params, values);
    const active = ["running", "queued", "waiting"].includes(p.status);
    const canRun = !active && !["blocked", "disabled", "skipped"].includes(p.status);
    const label = p.status === "done" || p.status === "stale" || p.status === "failed" ? "Run again" : "Run";
    const meta = [];
    if (p.finished && !active) meta.push(`finished ${relTime(p.finished)}`);
    if (p.duration_s && !active) meta.push(`took ${fmtDuration(p.duration_s)}`);
    if (p.reason) meta.push(p.reason);

    const actions = h("div", { class: "step-actions" },
      spec?.params?.length ? h("button", { class: "btn small ghost", "aria-expanded": openOptions.has(p.id) ? "true" : "false",
        onclick: () => { openOptions.has(p.id) ? openOptions.delete(p.id) : openOptions.add(p.id); renderSteps(); } }, "Options") : null,
      active ? h("button", { class: "btn small", onclick: () => api.post(`/api/jobs/${p.job}/cancel`).catch(errorToast) }, icon("stop"), "Cancel")
        : h("button", { class: `btn small ${p.status === "ready" && !p.optional ? "primary" : ""}`, disabled: !canRun,
          onclick: () => run([p.id], forms[p.id] ? { [p.id]: forms[p.id].get() } : {}) }, icon("play"), label));

    const body = h("div", { class: "step-body" });
    if (p.status === "running") {
      body.appendChild(h("div", { class: "stack", style: { gap: "4px" } },
        h("div", { class: `progress ${p.progress == null ? "indeterminate" : ""}` }, h("i", { style: { width: `${((p.progress || 0) * 100).toFixed(1)}%` } })),
        h("span", { class: "muted", style: { fontSize: "12.5px" } }, p.stage || "Working…")));
    }
    if (p.status === "failed" && p.error) body.appendChild(h("div", { class: "step-error" }, p.error));
    if (p.id === cropStep) {
      const box = detail.record.crop_box;
      body.appendChild(h("div", { class: "row wrap" },
        h("span", { class: "soft" }, box ? `Crop box z ${box[0]}–${box[1]}, y ${box[2]}–${box[3]}, x ${box[4]}–${box[5]}` : "No crop box drawn"),
        h("a", { class: "btn small", href: `#/crop/${enc(name)}` }, icon("crop"), box ? "Edit crop box" : "Draw crop box")));
    }
    if (p.id === "landmarks") {
      // Found automatically when the step runs; a landmark set by hand in the viewer is kept.
      const mapped = marks?.mapped || [];
      const where = (r) => (r.kind === "span" ? `${r.landmark} ${sig(r.start, 3)}–${sig(r.end, 3)}%` : `${r.landmark} ${sig(r.start, 3)}%`)
        + (r.source === "manual" ? " (by hand)" : "");
      body.appendChild(h("div", { class: "row wrap" },
        h("a", { class: "btn small", href: `#/viewer/${enc(name)}/mandible?mode=landmarks`,
          title: "Move a landmark by hand where it was found wrong; a mark set by hand is kept when the step runs again" },
        icon("target"), marks?.marks?.length ? "Adjust in the viewer" : "Mark in the viewer"),
        mapped.length ? h("span", { class: "soft" }, mapped.map(where).join(", ")) : null));
      if (marks?.warnings?.length) body.appendChild(h("div", { class: "muted", style: { fontSize: "12.5px", marginTop: "6px" } }, marks.warnings.join(" ")));
    }
    if (openOptions.has(p.id) && forms[p.id]) {
      // Say which options this run would change from the last one.
      const now = forms[p.id].get();
      const changed = spec.params.filter((q) => rec.params && q.key in rec.params
        && JSON.stringify(rec.params[q.key] ?? null) !== JSON.stringify(now[q.key] ?? null)).map((q) => q.label);
      body.appendChild(h("div", { class: "params-box" },
        changed.length ? h("p", { class: "muted", style: { margin: "0 0 10px", fontSize: "12.5px" } },
          `The last run used other values for ${changed.length > 4 ? `${changed.slice(0, 4).join(", ")} and ${changed.length - 4} more`
            : changed.join(", ")}.`) : null,
        forms[p.id].el,
        h("div", { class: "row", style: { marginTop: "10px" } },
          h("button", { class: "btn small primary", disabled: !canRun, onclick: () => run([p.id], { [p.id]: forms[p.id].get() }) }, icon("play"), `${label} with these options`),
          h("button", { class: "btn small ghost", onclick: async () => {
            try {
              await api.patch("/api/settings", { step_defaults: { [p.id]: forms[p.id].get() } });
              toast(`Saved as the default for ${spec.label}.`);
            } catch (err) { errorToast(err); }
          } }, "Make these the defaults"))));
    }
    return h("li", { class: p.status },
      h("div", { class: "step-top" }, h("h3", null, p.label),
        statusBadge(p.status, p.status === "running" && p.progress != null ? `Running ${Math.round(p.progress * 100)}%` : null, p.progress),
        actions),
      h("div", { class: "step-summary" }, [spec?.summary,
        p.id === "segment" && src.format === "nrrd" ? "The NRRD is read as it is, with no conversion." : null,
        meta.length ? `${meta.join(", ").replace(/^./, (c) => c.toUpperCase())}.` : null].filter(Boolean).join(" ")),
      body);
  }

  const stepsList = h("ol", { class: "steps" });
  function renderSteps() {
    const pipeline = livePipeline(detail.summary).filter((p) => p.status !== "skipped");
    replace(stepsList, pipeline.map(stepRow));
  }

  // ------------------------------------------------------------ results
  function tileGroup(label, ...tiles) {
    return h("div", { class: "tile-group" }, h("h3", null, label), h("div", { class: "tiles" }, tiles));
  }

  function tile(k, v, unit, d) {
    return h("div", { class: "tile" }, h("div", { class: "k" }, k),
      h("div", { class: "v" }, v, unit ? h("small", null, unit) : null), d ? h("div", { class: "d" }, d) : null);
  }

  function molarSection() {
    const sm = molar?.summary;
    if (!sm) return null;
    const color = slotColor(detail.summary.slot);
    const teeth = sm.teeth || [];
    const profile = molar.profile || [];
    const hist = sm.histogram;
    // Enamel density: the mean, with the lower and upper percentile beneath it.
    const pct = sm.density_percentiles;
    const spread = (lo, hi) => (lo != null && hi != null
      ? `${pct ? `${ordinal(pct[0])}–${ordinal(pct[1])} percentile` : "Percentiles"}: ${sig(lo, 4)}–${sig(hi, 4)}` : null);
    const calibrated = sm.enamel_mean_mgHA != null;
    const cal = sm.calibration;
    const calLine = cal ? h("div", { class: "muted", style: { fontSize: "12.5px", marginTop: "6px" } },
      `Density calibration from ${basename(cal.source)}${cal.description ? ` (${cal.description})` : ""}: ${calibrationFormula(cal)}`) : null;
    return h("section", { class: "section" },
      h("div", { class: "section-head" }, h("h2", null, "Molar enamel"),
        h("span", { class: "sub" }, sm.enamel_mask === "local-halfway"
          ? `Enamel between its halfway surfaces${sm.junction_level ? `; junction at ${sig(sm.junction_level.p5, 3)}–${sig(sm.junction_level.p95, 3)} along the tooth` : ""}`
          : `Threshold ${sig(sm.threshold, 4)} (${sm.threshold_method === "fixed" ? "fixed" : sm.threshold_method === "otsu" ? "Otsu" : sm.threshold_method === "knee" ? "log-histogram knee" : "log-histogram valley"})`),
        h("a", { class: "btn small", href: `#/viewer/${enc(name)}/molar` }, icon("viewer"), "View as heatmap"),
        h("a", { class: "btn small", href: `#/viewer/${enc(name)}/molar?plane=3d` }, icon("viewer"), "View in 3-D")),
      // The same three groups as the results page: density, volume, thickness.
      h("div", { class: "tile-groups" },
        tileGroup("Density",
          tile("Enamel density", calibrated ? sig(sm.enamel_mean_mgHA, 4) : "–", calibrated ? " mg HA/cm³" : null,
            calibrated ? `mean; ${spread(sm.enamel_low_mgHA, sm.enamel_high_mgHA)}` : "no HA calibration for this scan"),
          tile("Enamel intensity", sm.enamel_mean_raw != null ? sig(sm.enamel_mean_raw, 4) : "–", null,
            `raw mean${spread(sm.enamel_low_raw, sm.enamel_high_raw) ? `; ${spread(sm.enamel_low_raw, sm.enamel_high_raw)}` : ""}`),
          tile("Enamel/dentin density", sm.density_ratio != null ? sig(sm.density_ratio, 3) : "–", null, "median enamel over the dentin peak")),
        tileGroup("Volume",
          tile("Enamel share of molar volume", fmtPct(sm.enamel_fraction)),
          tile("Enamel volume", sig(sm.enamel_mm3, 3), " mm³"),
          sm.dentin_mm3 != null ? tile("Dentin volume", sig(sm.dentin_mm3, 3), " mm³") : null,
          tile("Molar volume", sig(sm.total_mm3, 3), " mm³")),
        tileGroup("Thickness",
          tile("Enamel thickness", sm.thickness_um != null ? sig(sm.thickness_um, 3) : "–", " µm",
            sm.thickness_p25_um != null
              ? `median, enamel–dentin junction to outer surface; middle half ${sig(sm.thickness_p25_um, 3)}–${sig(sm.thickness_p75_um, 3)} µm`
              : "median, by the earlier method: run Molar enamel again to update it"),
          sm.thickness_mean_um != null ? tile("Mean thickness", sig(sm.thickness_mean_um, 3), " µm",
            sm.thickness_sd_um != null ? `± ${sig(sm.thickness_sd_um, 3)} µm SD across the junction's points` : "across the junction's points") : null,
          sm.junction_area_mm2 != null ? tile("Enamel–dentin junction", sig(sm.junction_area_mm2, 3), " mm²", "surface area") : null,
          sm.thickness_um_per_mm2 != null ? tile("Thickness ÷ junction area", sig(sm.thickness_um_per_mm2, 3), " µm/mm²",
            "median thickness divided by the junction's area") : null,
          sm.thickness_mean_per_sqrt_junction != null ? tile("Mean thickness ÷ √junction area", fmtPct(sm.thickness_mean_per_sqrt_junction, 2), null,
            "both in µm: the enamel's mean thickness as a share of the junction's linear size") : null),
        // Patches of enamel thinner, or less dense, than the enamel around them (summaries from version 6).
        sm.thin_patch_share != null ? tileGroup("Patches",
          tile("Thin patches", fmtPct(sm.thin_patch_share, 2), null,
            `of the judged enamel–dentin junction${sm.thin_patches_per_mm2 != null ? `; ${sig(sm.thin_patches_per_mm2, 3)} per mm²` : ""}${sm.thin_patch_depth != null ? `, ${fmtPct(sm.thin_patch_depth, 0)} thinner on average` : ""}`),
          sm.low_density_patch_share != null ? tile("Less dense patches", fmtPct(sm.low_density_patch_share, 2), null,
            `of the judged junction${sm.low_density_patches_per_mm2 != null ? `; ${sig(sm.low_density_patches_per_mm2, 3)} per mm²` : ""}${sm.low_density_patch_depth != null ? `, ${fmtPct(sm.low_density_patch_depth, 1)} less mineral on average` : ""}`) : null,
          sm.core_density_mgHA != null ? tile("Enamel core density", sig(sm.core_density_mgHA, 4), " mg HA/cm³",
            "mean over the junction, read clear of the blurred faces")
            : sm.core_density_raw != null ? tile("Enamel core intensity", sig(sm.core_density_raw, 4), null,
              "raw mean over the junction, read clear of the blurred faces") : null) : null),
      calLine,
      (sm.warnings || []).length ? h("div", { class: "notice warning", style: { marginTop: "10px" } }, sm.warnings.join(" ")) : null,
      h("div", { class: "charts-2", style: { marginTop: "14px" } },
        teeth.length ? chartCard({
          title: "Per tooth", sub: "enamel volume / tooth volume",
          chart: barChart({ categories: teeth.map((t) => t.name), series: [{ label: name, color, values: teeth.map((t) => t.enamel_fraction) }],
            yLabel: "Enamel share", fy: pctFmt, height: 220 }),
          table: () => dataTable([
            { key: "name", label: "Tooth" }, { key: "enamel_fraction", label: "Enamel share", numeric: true, format: (v) => fmtPct(v) },
            { key: "enamel_mm3", label: "Enamel mm³", numeric: true, format: (v) => sig(v, 3) },
            { key: "volume_mm3", label: "Tooth mm³", numeric: true, format: (v) => sig(v, 3) },
            ...(teeth.some((t) => t.thickness_um != null)
              ? [{ key: "thickness_um", label: "Thickness µm", numeric: true, format: (v) => (v == null ? "–" : sig(v, 3)) }] : []),
            ...(teeth.some((t) => t.thickness_mean_um != null)
              ? [{ key: "thickness_mean_um", label: "Mean µm", numeric: true, format: (v) => (v == null ? "–" : sig(v, 3)) },
                { key: "thickness_sd_um", label: "SD µm", numeric: true, format: (v) => (v == null ? "–" : sig(v, 3)) }] : []),
            ...(teeth.some((t) => t.junction_area_mm2 != null)
              ? [{ key: "junction_area_mm2", label: "Junction mm²", numeric: true, format: (v) => (v == null ? "–" : sig(v, 3)) }] : []),
            ...(teeth.some((t) => t.thickness_um_per_mm2 != null)
              ? [{ key: "thickness_um_per_mm2", label: "Thickness ÷ area µm/mm²", numeric: true, format: (v) => (v == null ? "–" : sig(v, 3)) }] : []),
            ...(teeth.some((t) => t.thickness_mean_per_sqrt_junction != null)
              ? [{ key: "thickness_mean_per_sqrt_junction", label: "Mean ÷ √area", numeric: true, format: (v) => (v == null ? "–" : fmtPct(v, 2)) }] : []),
            ...(teeth.some((t) => t.thin_patch_share != null)
              ? [{ key: "thin_patch_share", label: "Thin patches", numeric: true, format: (v) => fmtPct(v, 2) }] : []),
            ...(teeth.some((t) => t.low_density_patch_share != null)
              ? [{ key: "low_density_patch_share", label: "Less dense patches", numeric: true, format: (v) => fmtPct(v, 2) }] : [])], teeth),
        }) : null,
        profile.length ? chartCard({
          title: "Along the molar row", sub: "share of each slice that is enamel",
          chart: lineChart({ series: [{ id: "p", label: "enamel share", color, points: profile.map((r) => [r.z_mm, r.fraction]) }],
            xLabel: "Position along the jaw (mm)", yLabel: "Enamel share", fy: pctFmt, height: 220 }),
          table: () => dataTable([{ key: "z", label: "Slice", numeric: true }, { key: "z_mm", label: "mm", numeric: true, format: (v) => sig(v, 4) },
            { key: "tooth", label: "Tooth" }, { key: "fraction", label: "Enamel share", numeric: true, format: (v) => fmtPct(v) }], profile),
        }) : null),
      thicknessMapCard(),
      hist ? h("div", { style: { marginTop: "14px" } }, chartCard({
        title: "Where enamel starts", sub: sm.junction_level
          ? `molar intensity histogram, log scale; the junction's level runs ${sig(sm.junction_level.p5, 3)}–${sig(sm.junction_level.p95, 3)} along the tooth`
          : "molar intensity histogram, log scale",
        chart: histogramChart({ edges: hist.edges, counts: hist.counts, color,
          markers: [sm.junction_level ? { x: sm.junction_level.median, label: "junction level (median)" } : { x: sm.threshold, label: "enamel threshold" },
            ...(sm.dentin_mode != null ? [{ x: sm.dentin_mode, label: "dentin peak" }] : [])] }),
      })) : null);
  }

  // The 3-D thickness map (the molar step's self-contained viewer) loads on
  // request: it is several megabytes, and turning it is not what every visit
  // to the page is for. Once opened it stays open across re-renders.
  let mapOpen = false;
  function thicknessMapCard() {
    const has = (key) => detail.artifacts.some((a) => a.key === key && a.exists);
    if (!has("thickness_map")) return null;
    const url = `/api/samples/${enc(name)}/files/thickness_map`;
    const body = h("div");
    const open = () => {
      mapOpen = true;
      replace(body, h("iframe", { class: "thickness-frame", src: url, title: `Enamel thickness map of ${name}, in 3-D` }));
    };
    if (mapOpen) open();
    else {
      replace(body, h("div", { class: "thickness-preview" },
        has("thickness_plot") ? h("img", { src: `/api/samples/${enc(name)}/files/thickness_plot`,
          alt: `Enamel thickness map of ${name}, seen from above and from the side` }) : null,
        h("button", { class: "btn primary", onclick: open }, icon("viewer"), "Turn it in 3-D")));
    }
    return h("div", { style: { marginTop: "14px" } }, h("div", { class: "chart-card" },
      h("div", { class: "cc-head" }, h("h3", null, "Enamel thickness map"),
        h("span", { class: "sub" }, "the enamel coloured by its thickness, in 3-D"),
        h("a", { class: "btn small ghost", href: url, target: "_blank", rel: "noopener" }, icon("external"), "Open in a new tab")),
      body));
  }

  function incisorSection() {
    const prof = incisor?.profile;
    if (!prof) return null;
    const color = slotColor(detail.summary.slot);
    // The slices the cleanup kept, with their running median, and the outliers it left out.
    const kept = [], outliers = [], rows = [];
    const counts = {};
    prof.z_percent.forEach((z, i) => {
      counts[prof.status[i]] = (counts[prof.status[i]] || 0) + 1;
      if (z == null || prof.ratio[i] == null || !MEASURABLE.includes(prof.status[i])) return;
      if (prof.kept[i]) kept.push([z, prof.ratio[i], prof.z[i], prof.ratio_med[i]]);
      else if (prof.outlier[i]) outliers.push([z, prof.ratio[i]]);
      rows.push({ z, r: prof.ratio[i], med: prof.kept[i] ? prof.ratio_med[i] : null,
        out: prof.kept[i] ? "" : prof.outlier[i] ? "outlier" : "filtered" });
    });
    kept.sort((a, b) => a[0] - b[0]);
    outliers.sort((a, b) => a[0] - b[0]);
    rows.sort((a, b) => a.z - b.z);
    const k = medianWindow(sliceStep(kept.map((p) => p[2]).filter((z) => z != null)));
    const m = incisor.metrics || {};
    const left = [m.outlier_slices ? `${m.outlier_slices} outlier${m.outlier_slices === 1 ? "" : "s"}` : null,
      m.filtered_slices ? `${m.filtered_slices} filtered` : null].filter(Boolean);
    const spans = (incisor.landmarks || []).map((r) => ({ x0: r.start, x1: r.kind === "span" ? r.end : null, label: r.landmark,
      tone: landmarkTone(r.landmark) }));
    return h("section", { class: "section" },
      h("div", { class: "section-head" }, h("h2", null, "Incisor enamel"),
        h("span", { class: "sub" }, `${incisor.run?.dentin_baseline?.source === "molar" ? "Molar dentin baseline" : "Traced dentin baseline"}`),
        h("a", { class: "btn small", href: `#/viewer/${enc(name)}/cmpr` }, icon("viewer"), "View straightened incisor")),
      prof.stale ? h("div", { class: "notice warning", style: { marginBottom: "10px" } },
        "This profile was written by an older build of Incisor enamel, before its current handling of where enamel starts, so it can open high and drop. Run Incisor enamel again.") : null,
      // The plateau and the peak each in mg HA/cm³, as stored and as the ratio, as the molar's density tiles are.
      h("div", { class: "tile-groups" },
        tileGroup("Along the tooth",
          tile("Ratio crosses 1.0 at", m.crossover_z != null ? sig(m.crossover_z, 3) : "–", m.crossover_z != null ? "%" : "",
            "where it last rises through dentin density, from the cervical loop"),
          tile("Slices kept", (m.kept_slices ?? 0).toLocaleString(), null,
            `of ${prof.z.length.toLocaleString()}${left.length ? `; ${left.join(" and ")} left out` : ""}`)),
        ...[["plateau", "Mature plateau", "mean 70–95% along the tooth"], ["peak", "Peak mineralization", "highest along the tooth"]]
          .map(([key, label, read]) => tileGroup(label,
            tile("Enamel density", m[`${key}_mgHA`] != null ? sig(m[`${key}_mgHA`], 4) : "–", m[`${key}_mgHA`] != null ? " mg HA/cm³" : null,
              incisor.calibration ? `${read}; dentin reference ${m.dentin_mgHA != null ? sig(m.dentin_mgHA, 4) : "–"}` : "no HA calibration for this scan"),
            tile("Enamel intensity", m[`${key}_raw`] != null ? sig(m[`${key}_raw`], 4) : "–", null,
              `raw, ${read}; dentin reference ${m.dentin_raw != null ? sig(m.dentin_raw, 4) : "–"}`),
            tile("Enamel/dentin ratio", m[`${key}_ratio`] != null ? sig(m[`${key}_ratio`], 3) : "–", null, read)))),
      h("div", { style: { marginTop: "14px" } }, chartCard({
        title: "Mineralization along the tooth",
        sub: "enamel/dentin density ratio, cervical loop (0%) to tip (100%), cleaned as the R report cleans it",
        legendItems: [{ label: "Kept slice", color, kind: "dot" },
          ...(outliers.length ? [{ label: "Outlier, left out", color: "var(--ink-3)", kind: "dot" }] : []),
          { label: `Running median (21 slices${k === 21 ? "" : `, ${k} points`})`, color }],
        chart: lineChart({
          series: [
            { id: "raw", label: "slice", color, points: kept.map((p) => [p[0], p[1]]), width: 0, markers: true, opacity: 0.35, tip: false },
            ...(outliers.length ? [{ id: "outliers", label: "outlier", color: "var(--ink-3)", points: outliers, width: 0, markers: true, opacity: 0.8, tip: false }] : []),
            { id: "smooth", label: "running median", color, points: kept.map((p) => [p[0], p[3]]) },
          ],
          rules: [{ y: 1, label: "dentin density" }], spans,
          xLabel: "Position along the incisor (%)", yLabel: "Enamel / dentin", xDomain: [0, 100], height: 300,
          fx: (v) => `${Math.round(v)}`, tipHead: (x) => `${x.toFixed(1)}% along the tooth`,
        }),
        table: () => dataTable([{ key: "z", label: "Position (%)", numeric: true, format: (v) => sig(v, 3) },
          { key: "r", label: "Ratio", numeric: true, format: (v) => sig(v, 3) },
          { key: "med", label: "Running median", numeric: true, format: (v) => (v == null ? "–" : sig(v, 3)) },
          { key: "out", label: "Left out" }], rows),
        extra: h("div", { class: "row wrap", style: { marginTop: "8px", fontSize: "12.5px" } },
          Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([st, n]) =>
            h("span", { class: "tag" }, `${n} ${STATUS_WORDS[st] || st}`)),
          detail.artifacts.some((a) => a.key === "incisor_plot_ratio" && a.exists)
            ? h("a", { href: `/api/samples/${enc(name)}/files/incisor_plot_ratio`, target: "_blank", rel: "noopener" }, "Pipeline figure") : null),
      })),
      densityChart(prof, color, spans));
  }

  // The same profile in density rather than as a ratio: converted to mg HA/cm³
  // with the scan's HA calibration, or as stored, with the dentin beside it.
  function densityChart(prof, color, spans) {
    const host = h("div", { style: { marginTop: "14px" } });
    const cal = incisor.calibration;
    const draw = () => {
      const mg = Boolean(cal) && (densityView.units || "mgHA") === "mgHA";
      const conv = (v) => (v == null ? null : mg ? toMgHA(v, cal) : v);
      // The enamel over the kept slices with the server's running median, the dentin smoothed the same way here.
      const enamel = [], enamelLine = [], dentin = [];
      prof.z_percent.forEach((z, i) => {
        if (z == null) return;
        if (prof.kept[i] && prof.density[i] != null) {
          enamel.push([z, conv(prof.density[i]), prof.z[i]]);
          enamelLine.push([z, conv(prof.density_med[i])]);
        }
        if (DENTIN_MEASURED.includes(prof.status[i]) && prof.dentin_mode?.[i] != null) dentin.push([z, conv(prof.dentin_mode[i]), prof.z[i]]);
      });
      enamel.sort((a, b) => a[0] - b[0]);
      enamelLine.sort((a, b) => a[0] - b[0]);
      const dentinLine = densityView.dentin ? smoothProfile(dentin, 21) : [];
      const units = mg ? "mg HA/cm³" : "raw";
      const fy = numFmt;
      const seg = h("div", { class: "seg", role: "group", "aria-label": "Density units" },
        [["mgHA", "mg HA/cm³"], ["raw", "Raw intensity"]].map(([v, l]) => h("button", {
          "aria-pressed": (mg ? "mgHA" : "raw") === v ? "true" : "false", disabled: v === "mgHA" && !cal,
          title: v === "mgHA" && !cal ? "No HA calibration for this scan" : null,
          onclick: () => { densityView.units = v; draw(); },
        }, l)));
      const check = h("label", { class: "check" }, h("input", { type: "checkbox", checked: densityView.dentin,
        onchange: (e) => { densityView.dentin = e.target.checked; draw(); } }), "Dentin line");
      const series = [
        { id: "slices", label: "slice", color, points: enamel.map((p) => [p[0], p[1]]), width: 0, markers: true, opacity: 0.35, tip: false },
        { id: "enamel", label: "enamel, running median", color, points: enamelLine },
      ];
      if (dentinLine.length) series.push({ id: "dentin", label: "dentin, running median", color: "var(--ink-2)", points: dentinLine, width: 1.5, dash: "6 4" });
      const notes = [];
      if (mg) notes.push(`Density calibration from ${cal.source ? basename(cal.source) : "the Molar enamel settings"}${cal.description ? ` (${cal.description})` : ""}: ${calibrationFormula(cal)}.`);
      else if (!cal) notes.push("No HA calibration for this scan, so only its stored values. The converter copies the calibration from a Scanco ISQ; for other scans, enter one in the Molar enamel step's Density calibration settings.");
      if (dentinLine.length) notes.push("Dentin is the dentin peak fitted in each slice; toward the tip, where little dentin is left, it drifts up toward the enamel.");
      replace(host, chartCard({
        title: "Density along the tooth",
        sub: mg ? "enamel converted to mg HA/cm³, cervical loop (0%) to tip (100%)" : "enamel as stored in the scan, cervical loop (0%) to tip (100%)",
        controls: [seg, check],
        legendItems: [{ label: "Enamel", color }, ...(dentinLine.length ? [{ label: "Dentin", color: "var(--ink-2)", kind: "dash" }] : [])],
        chart: enamel.length ? lineChart({
          series, spans, xLabel: "Position along the incisor (%)", yLabel: mg ? "Density (mg HA/cm³)" : "Intensity (raw)",
          xDomain: [0, 100], height: 300, fy, fx: (v) => `${Math.round(v)}`, tipHead: (x) => `${x.toFixed(1)}% along the tooth`,
        }) : h("div", { class: "empty" }, h("span", null, "No measured slices.")),
        table: () => {
          const dentinAt = new Map(dentin.map((q) => [q[2], q[1]]));
          return dataTable([{ key: "z", label: "Position (%)", numeric: true, format: (v) => sig(v, 3) },
            { key: "e", label: `Enamel (${units})`, numeric: true, format: (v) => (v == null ? "–" : fy(v)) },
            { key: "d", label: `Dentin (${units})`, numeric: true, format: (v) => (v == null ? "–" : fy(v)) }],
          enamel.map((p) => ({ z: p[0], e: p[1], d: dentinAt.get(p[2]) })));
        },
        extra: notes.length ? h("div", { class: "muted", style: { fontSize: "12.5px", marginTop: "6px" } }, notes.join(" ")) : null,
      }));
    };
    draw();
    return host;
  }

  function checksSection() {
    const v = detail.record.validation;
    if (!v) return null;
    return h("section", { class: "section" },
      h("div", { class: "section-head" }, h("h2", null, "Segmentation checks"), statusBadge(v.status),
        h("span", { class: "sub" }, `${v.checks} checks`)),
      v.problems?.length ? h("ul", { class: "problems" }, v.problems.map((p) =>
        h("li", null, h("b", null, `${p.stage}: `), p.message))) : h("p", { class: "soft" }, "Every plausibility check passed."));
  }

  // ------------------------------------------------------------ side column
  function viewsPanel() {
    const cards = detail.views.map((v) => {
      const job = previewJob(name, v.view);
      const note = job ? `Building ${job.progress != null ? Math.round(job.progress * 100) + "%" : "…"}`
        : !v.built ? "Preview not built" : !v.current ? "Outdated preview" : `${v.shape?.join(" × ")}`;
      return h("a", { class: "view-card", href: `#/viewer/${enc(name)}/${v.view}` },
        h("div", { class: "th" }, v.thumb ? h("img", { src: `/api/samples/${enc(name)}/views/${v.view}/file/thumb.png?v=${v.current ? 1 : 0}-${(v.shape || []).join("x")}`, alt: "" }) : h("span", null, job ? "Building…" : "No preview")),
        h("div", { class: "cap" }, v.label, h("small", null, note)));
    });
    return h("div", { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", null, "Views")),
      h("div", { class: "panel-body" }, cards.length ? h("div", { class: "views-grid" }, cards)
        : h("p", { class: "muted" }, "Views appear once there is a scan or a segmentation to show.")));
  }

  function filesPanel() {
    const rows = detail.artifacts.filter((a) => a.exists).sort((a, b) => a.label.localeCompare(b.label));
    return h("div", { class: "panel", style: { marginTop: "16px" } },
      h("div", { class: "panel-head" }, h("h2", { class: "grow" }, "Files"),
        h("button", { class: "btn small ghost", onclick: () => api.post("/api/reveal", { path: detail.folder }).catch(errorToast) }, "Show folder")),
      h("div", { class: "panel-body files" }, rows.length ? rows.map((a) => h("div", { class: "file" },
        h("div", { style: { minWidth: 0 } }, h("div", null, a.label,
          a.where === "linked" ? h("span", { class: "tag", style: { marginLeft: "6px" } }, "linked")
            : a.where === "storage" ? h("span", { class: "tag accent", style: { marginLeft: "6px" }, title: "Written beside the scan, on its own drive" }, "beside the scan") : null),
          h("div", { class: "p", title: a.path }, `\u200E${a.path}`)),
        h("div", { class: "row" }, a.size != null ? h("span", { class: "muted num", style: { fontSize: "12px" } }, fmtBytes(a.size)) : null,
          /\.(csv|json|png)$/i.test(a.path) ? h("a", { class: "btn small ghost icon", title: "Download", "aria-label": `Download ${a.label}`,
            href: `/api/samples/${enc(name)}/files/${a.key}?download=1` }, icon("download")) : null))) : h("p", { class: "muted" }, "No files yet.")));
  }

  function jobsPanel() {
    const jobs = [...state.jobs.values()].filter((j) => j.sample === name).sort((a, b) => (a.created < b.created ? 1 : -1)).slice(0, 8);
    if (!jobs.length) return null;
    return h("div", { class: "panel", style: { marginTop: "16px" } },
      h("div", { class: "panel-head" }, h("h2", { class: "grow" }, "Recent jobs"), h("a", { class: "btn small ghost", href: "#/queue" }, "Queue")),
      h("div", { class: "panel-body files" }, jobs.map((j) => h("div", { class: "file" },
        h("a", { href: `#/queue/${j.id}` }, j.title),
        h("span", null, statusBadge(j.status, null, j.progress), h("span", { class: "muted", style: { fontSize: "12px", marginLeft: "8px" } },
          j.status === "running" ? fmtDuration(elapsedSince(j.started)) : relTime(j.finished || j.created)))))));
  }

  // ------------------------------------------------------------ assemble
  const sideSlot = h("div");
  function render() {
    replace(page,
      head(),
      h("div", { class: "sample-grid" },
        h("div", { style: { minWidth: 0 } },
          h("div", { class: "panel" },
            h("div", { class: "panel-head" }, h("h2", { class: "grow" }, "Pipeline"),
              detail.default_run?.length ? h("button", { class: "btn small primary", onclick: () => run("default") }, icon("play"),
                `Run the missing ${detail.default_run.length === 1 ? "step" : `${detail.default_run.length} steps`}`) : null),
            stepsList),
          molarSection(), incisorSection(), checksSection()),
        sideSlot));
    renderSteps();
    renderSide();
  }
  function renderSide() {
    replace(sideSlot, viewsPanel(), filesPanel(), jobsPanel());
  }

  const refresh = debounce(async (full) => {
    if (await load()) (full ? render() : (renderSteps(), renderSide()));
  }, 250);

  if (await load()) render();
  let lastFinished = "";
  const offs = [
    on("jobs", (job) => {
      if (!detail || (job && job.sample !== name)) return;
      renderSteps();
      if (job && ["done", "failed", "cancelled"].includes(job.status) && job.id !== lastFinished) {
        lastFinished = job.id;
        refresh(true);
      } else renderSide();
    }),
    on("samples", (s) => { if (detail && s && s.name === name) { detail.summary = s; renderSteps(); } }),
  ];
  return () => offs.forEach((f) => f());
}

export { statusInfo };
