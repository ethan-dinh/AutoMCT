// Import: bring a scan in (upload it, or point at it where it already is), say
// what it is, and let the studio name, file and process it.

import { h, icon, replace, fmtBytes, debounce, errorToast, toast, basename } from "../lib.js";
import { api, enc, uploadFiles, filesFromDrop, filesFromInput } from "../api.js";
import { metaForm, browser, pickPath, META_LABELS, folderSelect, folderHref } from "../components.js";
import { state } from "../store.js";
import { go, pending } from "../app.js";
import { foundSheet } from "./found.js";

const KIND_LABEL = {
  isq: "Scanco ISQ scan", rsq: "Scanco RSQ projections", dicom: "DICOM series", bmp: "BMP slice stack",
  volume: "Volume (.nrrd/.tif)", segmentation: "Segmentation results folder",
  collection: "Folder of several samples", unknown: "Unrecognized", results: "Existing results",
};
const FORMAT_LABEL = { nrrd: "NRRD volume", tiff: "TIFF volume" };

// Formats converted to NRRD before segmentation; an NRRD is segmented as it is.
const CONVERTED = ["isq", "dicom", "bmp", "tiff"];

function formatOf(info) {
  if (!info) return null;
  if (info.format) return info.format;
  if (info.kind !== "volume") return info.kind;
  return /\.nrrd$/i.test(info.path || "") ? "nrrd" : "tiff";
}

function kindLabel(info) {
  return (info.kind === "volume" && FORMAT_LABEL[formatOf(info)]) || KIND_LABEL[info.kind] || info.kind;
}

// What linking in place means for where files go: the scan stays put, and its
// large files are written beside it so they stay on that drive.
function linkHelp(info) {
  if (!info.storage) return "Nothing is copied; the scan stays where it is.";
  const big = formatOf(info) === "nrrd" ? "Its reoriented mandible is" : "Its converted scan and reoriented mandible are";
  return `Nothing is copied. ${big} written beside it, in MicroCT-Studio/<specimen> on the same drive; the smaller results go in the workspace.`;
}

function pipelineHelp(format) {
  const rest = "straighten the incisor, then measure incisor and molar enamel. It runs in the background; you can keep working.";
  if (format === "nrrd") return `An NRRD needs no conversion: segment, ${rest}`;
  if (CONVERTED.includes(format)) return `Convert to NRRD, segment, ${rest}`;
  return `Convert to NRRD (an NRRD skips this), segment, ${rest}`;
}

const ARTIFACT_SHORT = {
  reoriented: "Scan", incisor_volume: "Incisor", molar_volume: "Molars", bone_volume: "Bone",
  cmpr_volume: "CMPR", cmpr_transform: "Transform", incisor_profile: "Density profile",
  landmarks: "Landmarks", molar_quant: "Molar enamel", validation: "Checks",
};

// Details that describe a group or a batch rather than one animal, which a
// new specimen most often shares with one already imported.
const SHARED_DETAILS = ["age", "line", "genotype", "study", "scanner"];

// What a found specimen is, in a line under its name: the scan file, and the
// results found for it.
function sourceLine(c) {
  const src = c.source || {};
  const parts = [];
  if (src.kind && src.kind !== "results" && src.path) {
    parts.push(["dicom", "bmp"].includes(src.kind) ? `${src.kind === "dicom" ? "DICOM" : "BMP"} in ${basename(src.path)}` : basename(src.path));
  }
  const arts = Object.keys(c.artifacts || {}).filter((k) => ARTIFACT_SHORT[k]).map((k) => ARTIFACT_SHORT[k]);
  if (arts.length) parts.push(arts.length > 4 ? `${arts.slice(0, 3).join(", ")} and ${arts.length - 3} more` : arts.join(", "));
  const title = [src.kind && src.kind !== "results" ? kindLabel(src) : null, src.path,
    arts.length ? `Found: ${arts.join(", ")}` : null,
    c.recorded?.scanner_name ? `Named at the scanner: ${c.recorded.scanner_name}` : null].filter(Boolean).join("\n");
  return { text: parts.join(" + ") || c.name, title };
}

export function mount(root, _params, query) {
  const ui = {
    mode: query.mode || "upload", files: [], uploadId: null, uploading: false, sent: 0, total: 0,
    abort: null, info: null, path: null, pathInfo: null, link: "link", candidates: [],
    // The library folder the specimens are filed in: the one Import was opened from.
    folder: query.folder || "",
  };
  const page = h("div", { class: "page" });
  root.appendChild(page);
  page.appendChild(h("div", { class: "page-head" },
    h("div", null, h("h1", null, "Import"),
      h("div", { class: "sub" }, "Raw scans, converted volumes, or results you already have. Nothing is renamed or moved without asking."))));

  const tabs = h("div", { class: "seg", role: "tablist" });
  page.appendChild(h("div", { class: "row", style: { marginBottom: "16px" } }, tabs));
  const content = h("div");
  page.appendChild(content);

  let suggestions = {};
  api.get("/api/meta/values").then((v) => { suggestions = v; if (ui.mode !== "scan") render(); }).catch(() => {});

  // ------------------------------------------------ the right-hand form
  const nameEl = h("div", { class: "name-preview empty-name" }, "Name appears as you fill in the details");
  const nameNote = h("div", { class: "muted", style: { fontSize: "12.5px" } });
  const nameOverride = h("input", { type: "text", placeholder: "Use the composed name", "aria-label": "Custom name" });
  // The name the scan came with, which it keeps until its details can name it.
  const foundName = () => (ui.mode === "upload" ? ui.info : ui.pathInfo)?.found || "";
  const compose = debounce(async () => {
    nameNote.textContent = "";
    if (nameOverride.value.trim()) { nameEl.textContent = nameOverride.value.trim(); nameEl.classList.remove("empty-name"); return; }
    try {
      const res = await api.post("/api/names/compose", { meta: meta.get(), found: foundName() });
      nameEl.textContent = res.name || "Name appears as you fill in the details";
      nameEl.classList.toggle("empty-name", !res.name);
      if (res.name && res.from === "found") nameNote.textContent = "Its own name, until it has an age and a genotype.";
    } catch { /* offline: leave the last name */ }
  }, 150);
  nameOverride.addEventListener("input", compose);
  let meta = metaForm({}, {}, compose);

  const cropSel = h("select", null,
    h("option", { value: "auto" }, "Crop to the specimen automatically"),
    h("option", { value: "draw" }, "Let me draw a crop box first"),
    h("option", { value: "none" }, "Keep the whole scan"));
  const runBox = h("input", { type: "checkbox", checked: true });
  const submitBtn = h("button", { class: "btn primary", onclick: () => submit() }, "Import specimen");
  const status = h("div", { class: "muted" });

  // Fill the empty fields from a specimen already imported: the details it
  // most likely shares with this one.
  function copyFrom() {
    const recent = [...state.samples.values()]
      .sort((a, b) => String(b.created || "").localeCompare(String(a.created || ""))).slice(0, 20);
    if (!recent.length) return null;
    const sel = h("select", { class: "copy-from", "aria-label": "Copy details from a specimen" },
      h("option", { value: "" }, "Copy details from…"), recent.map((s) => h("option", { value: s.name }, s.name)));
    sel.addEventListener("change", () => {
      const s = state.samples.get(sel.value);
      sel.value = "";
      if (!s) return;
      const values = Object.fromEntries(SHARED_DETAILS.filter((k) => s.meta?.[k]).map((k) => [k, s.meta[k]]));
      const from = Object.fromEntries(Object.keys(values).map((k) => [k, { from: `From ${s.name}`, guess: false }]));
      const filled = meta.set(values, from);
      compose();
      toast(filled.length ? `Filled ${filled.map((k) => META_LABELS[k].toLowerCase()).join(", ")} from ${s.name}.`
        : "Nothing to fill in: those details are already filled in.", filled.length ? "good" : "info");
    });
    return sel;
  }

  // Where in the library the import is filed. Nothing moves on disk for it.
  const folderField = (label) => h("label", { class: "field" }, h("span", { class: "lbl" }, label),
    folderSelect(ui.folder, { onchange: (e) => { ui.folder = e.target.value; } }),
    h("span", { class: "help" }, "A folder of the library, to keep studies apart. Nothing moves on disk."));

  function formPanel() {
    meta = metaForm(meta.get(), suggestions, compose, { sources: meta.sources(), recorded: meta.recorded() });
    compose();
    const format = formatOf(ui.mode === "upload" ? ui.info : ui.pathInfo);
    return h("div", { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", { class: "grow" }, "About this specimen"), copyFrom()),
      h("div", { class: "panel-body stack" },
        h("div", null, h("div", { class: "muted", style: { fontSize: "12.5px" } }, "It will be filed as"), nameEl, nameNote),
        meta.el,
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Custom name"), nameOverride,
          h("span", { class: "help" }, "Only if the composed name is wrong. Letters, digits and . _ + - only.")),
        folderField("Folder"),
        h("div", { class: "form-group stack" },
          CONVERTED.includes(format) ? h("label", { class: "field" }, h("span", { class: "lbl" }, "Cropping"), cropSel,
            h("span", { class: "help" }, "Cropping before conversion makes every later step faster.")) : null,
          h("label", { class: "check" }, runBox, h("span", null, "Run the pipeline after import")),
          h("span", { class: "help muted" }, pipelineHelp(format))),
        h("div", { class: "row" }, submitBtn, status)));
  }

  async function submit() {
    const body = { meta: meta.get(), run: null, params: {}, folder: ui.folder };
    if (nameOverride.value.trim()) body.name = nameOverride.value.trim();
    if (ui.mode === "upload") {
      if (!ui.files.length) return errorToast(new Error("Add a file or folder first"));
      if (ui.uploading) {
        submitBtn.disabled = true;
        status.textContent = "Importing as soon as the upload finishes…";
        await ui.uploadDone;
      }
      if (!ui.uploadId) return errorToast(new Error("The upload did not finish"));
      body.source = { upload: ui.uploadId };
    } else {
      if (!ui.path) return errorToast(new Error("Choose a file or folder first"));
      body.source = { path: ui.path, mode: ui.link };
    }
    // Read once the upload has finished, which is when its format is known.
    const format = formatOf(ui.mode === "upload" ? ui.info : ui.pathInfo);
    const crop = CONVERTED.includes(format) ? cropSel.value : "none";
    if (runBox.checked && crop !== "draw") {
      body.run = "default";
      body.params = { convert: { autocrop: crop === "auto" } };
    }
    submitBtn.disabled = true;
    try {
      const res = await api.post("/api/samples", body);
      const name = res.sample.name;
      ui.uploadId = null;
      toast(`Imported ${name}.`);
      go(crop === "draw" ? `#/crop/${encodeURIComponent(name)}` : `#/sample/${encodeURIComponent(name)}`);
    } catch (err) {
      errorToast(err);
      submitBtn.disabled = false;
      status.textContent = "";
    }
  }

  // ------------------------------------------------ upload
  async function startUpload(files) {
    if (!files.length) return;
    if (ui.abort) ui.abort.abort();
    if (ui.uploadId) api.del(`/api/uploads/${ui.uploadId}`).catch(() => {});
    ui.files = files; ui.uploadId = null; ui.info = null; ui.sent = 0;
    ui.total = files.reduce((s, f) => s + f.file.size, 0);
    ui.uploading = true;
    ui.abort = new AbortController();
    let resolveDone;
    ui.uploadDone = new Promise((r) => { resolveDone = r; });
    render();
    try {
      ui.uploadId = await uploadFiles(files, (sent, total) => { ui.sent = sent; ui.total = total; drawProgress(); }, ui.abort.signal);
      ui.info = await api.get(`/api/uploads/${ui.uploadId}`);
      meta.set(ui.info.meta || {}, ui.info.sources || {}, ui.info.recorded || {});
    } catch (err) {
      if (!ui.abort.signal.aborted) errorToast(err);
    } finally {
      ui.uploading = false;
      resolveDone();
      render();
    }
  }

  const progressBar = h("div", { class: "progress" }, h("i", { style: { width: "0%" } }));
  const progressText = h("span", { class: "muted num" });
  function drawProgress() {
    const f = ui.total ? ui.sent / ui.total : 0;
    progressBar.firstChild.style.width = `${(f * 100).toFixed(1)}%`;
    progressText.textContent = `${fmtBytes(ui.sent)} of ${fmtBytes(ui.total)}`;
  }

  function uploadPanel() {
    const fileInput = h("input", { type: "file", multiple: true, style: { display: "none" },
      onchange: (e) => startUpload(filesFromInput(e.target)) });
    const dirInput = h("input", { type: "file", webkitdirectory: true, style: { display: "none" },
      onchange: (e) => startUpload(filesFromInput(e.target)) });
    const zone = h("div", { class: "dropzone", tabindex: 0, role: "button",
      onclick: () => fileInput.click(),
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } },
      ondragover: (e) => { e.preventDefault(); zone.classList.add("over"); },
      ondragleave: () => zone.classList.remove("over"),
      ondrop: async (e) => { e.preventDefault(); zone.classList.remove("over"); startUpload(await filesFromDrop(e.dataTransfer)); },
    }, icon("upload"), h("h3", null, "Drop a scan file or folder here"),
      h("span", null, "ISQ, DICOM or BMP folders, NRRD or TIFF. Uploading starts right away, so fill in the details while it runs."));
    const list = ui.files.length ? h("div", { class: "filelist" },
      h("div", { class: "row" },
        h("b", null, ui.files.length === 1 ? basename(ui.files[0].path) : `${ui.files.length} files from ${ui.files[0].path.split("/")[0]}`),
        h("span", { class: "spacer" }),
        ui.uploading ? h("button", { class: "btn small ghost", onclick: () => { ui.abort.abort(); ui.files = []; render(); } }, "Cancel") : null),
      progressBar, h("div", { class: "row" }, progressText, h("span", { class: "spacer" }),
        ui.info ? h("span", { class: "tag accent" }, kindLabel(ui.info)) : ui.uploading ? h("span", { class: "muted" }, "Uploading…") : null),
      ui.info && ["collection", "unknown"].includes(ui.info.kind)
        ? h("div", { class: "notice warning" }, ui.info.kind === "collection"
          ? "This looks like several samples. Use Scan folders to import them together."
          : "This is not a format the pipeline reads. Supported: ISQ, DICOM, BMP stacks, NRRD, TIFF.") : null) : null;
    setTimeout(drawProgress);
    return h("div", { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", null, "Scan files")),
      h("div", { class: "panel-body" }, zone,
        h("div", { class: "row", style: { marginTop: "10px" } },
          h("button", { class: "btn", onclick: () => fileInput.click() }, icon("file"), "Choose files"),
          h("button", { class: "btn", onclick: () => dirInput.click() }, icon("folder"), "Choose a folder"),
          fileInput, dirInput),
        h("p", { class: "help muted", style: { marginTop: "8px" } },
          "Uploading copies the scan into the workspace. To leave a scan on its own drive, with its large files beside it, use From this computer."),
        list));
  }

  // ------------------------------------------------ from this computer
  function pathPanel() {
    const info = h("div", { class: "stack", style: { marginTop: "10px" } });
    const showInfo = () => {
      if (!ui.pathInfo) return replace(info, h("span", { class: "muted" }, "Select a scan file or a sample folder."));
      const p = ui.pathInfo;
      replace(info,
        h("div", { class: "row wrap" }, h("span", { class: "tag accent" }, kindLabel(p)),
          h("span", { class: "muted num" }, `${fmtBytes(p.bytes)} in ${p.files.toLocaleString()} file${p.files === 1 ? "" : "s"}`)),
        h("code", { class: "soft" }, p.path),
        p.kind === "collection" ? h("div", { class: "notice" }, "A folder of several samples. ",
          h("button", { class: "btn small", onclick: () => { ui.mode = "scan"; ui.scanPaths = [p.path]; render(); } }, "Scan it for specimens")) : null,
        p.kind === "segmentation" ? h("div", { class: "row wrap" }, "Found:",
          Object.keys(p.artifacts || {}).filter((k) => ARTIFACT_SHORT[k]).map((k) => h("span", { class: "tag" }, ARTIFACT_SHORT[k]))) : null,
        ["isq", "dicom", "bmp", "volume", "rsq"].includes(p.kind) ? h("div", { class: "seg" },
          h("button", { "aria-pressed": ui.link === "link" ? "true" : "false", onclick: () => { ui.link = "link"; showInfo(); } }, "Use it where it is"),
          h("button", { "aria-pressed": ui.link === "copy" ? "true" : "false", onclick: () => { ui.link = "copy"; showInfo(); } }, "Copy into the workspace")) : null,
        ["isq", "dicom", "bmp", "volume", "rsq"].includes(p.kind) ? h("span", { class: "help muted" }, ui.link === "link"
          ? linkHelp(p)
          : "A copy is made in the workspace in the background, so the original can be unplugged afterwards. Its large files stay in the workspace too.") : null,
        ui.link === "link" && p.storage?.synced ? h("div", { class: "notice warning" },
          `That folder syncs to ${p.storage.synced}, so the large files written beside the scan upload to it too.`) : null);
    };
    const b = browser({
      start: ui.path || "",
      onSelect: async (e) => {
        ui.path = e.path;
        try {
          ui.pathInfo = await api.get(`/api/fs/inspect?path=${enc(e.path)}`);
          meta.set(ui.pathInfo.meta || {}, ui.pathInfo.sources || {}, ui.pathInfo.recorded || {});
        }
        catch (err) { errorToast(err); ui.pathInfo = null; }
        showInfo();
        replace(formSlot, formPanel());
      },
    });
    showInfo();
    return h("div", { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", null, "Choose from this computer")),
      h("div", { class: "panel-body" }, b.el, info));
  }

  // ------------------------------------------------ scan folders
  function scanPanel() {
    const paths = h("textarea", { rows: 4, class: "path", placeholder: "/Volumes/Z Drive/Incisor Enamel Analysis/SEG DATA\n/Volumes/Z Drive/Incisor Enamel Analysis/CMPR" });
    paths.value = (ui.scanPaths || []).join("\n");
    const results = h("div", { class: "found" });
    const scan = async () => {
      ui.scanPaths = paths.value.split("\n").map((p) => p.trim()).filter(Boolean);
      if (!ui.scanPaths.length) return errorToast(new Error("Add at least one folder"));
      replace(results, h("div", { class: "stack", style: { gap: "8px" } }, h("div", { class: "progress indeterminate" }, h("i")),
        h("span", { class: "muted" }, "Finding specimens and reading what their scans record…")));
      try {
        const res = await api.post("/api/import/scan", { paths: ui.scanPaths });
        const rows = res.samples.map((c) => {
          // One already imported shows the details the library holds for it.
          const held = c.existing ? state.samples.get(c.existing)?.meta : null;
          return {
            ...c, key: c.name, typed: "", meta: { ...(held || c.meta || {}) }, sources: held ? {} : { ...(c.sources || {}) },
            recorded: c.recorded || {}, include: !c.exists && (Object.keys(c.artifacts).length > 0 || c.source?.kind !== "results"),
          };
        });
        ui.candidates = [...rows.filter((c) => !c.exists), ...rows.filter((c) => c.exists)];
        drawCandidates();
      } catch (err) { errorToast(err); replace(results); }
    };
    const drawCandidates = () => {
      if (!ui.candidates.length) return replace(results, h("div", { class: "empty" }, h("h3", null, "No specimens found there"),
        h("span", null, "Point at a folder that holds one folder per specimen, or loose scans.")));
      replace(results, foundSheet({
        rows: ui.candidates, suggestions, sourceLine,
        commit: async (samples, run) => {
          const res = await api.post("/api/import/commit", { samples, run, folder: ui.folder });
          toast(`Imported ${res.created.length} specimen${res.created.length === 1 ? "" : "s"}. Viewer previews are building in the background.`);
          if (res.failed.length) toast(res.failed.map((f) => `${f.sample}: ${f.reason}`).join("  "), "warning", { timeout: 10000 });
          ui.candidates = [];
          go(folderHref(ui.folder));
        },
      }));
    };
    if (ui.candidates.length) setTimeout(drawCandidates);
    return h("div", { class: "panel" },
      h("div", { class: "panel-head" }, h("h2", null, "Scan folders for specimens")),
      h("div", { class: "panel-body stack" },
        h("p", { class: "soft" }, "Give the folders your data lives in: a segmentation root with one folder per specimen, a CMPR folder, a RESULTS folder, or folders of raw scans. Everything found is matched up by specimen name and linked in place, and each specimen's details are filled in from what its scan, its name and its folders say."),
        paths,
        h("div", { class: "row" },
          h("button", { class: "btn", onclick: async () => {
            const picked = await pickPath({ title: "Add a folder", mode: "dir" });
            if (picked) paths.value = (paths.value.trim() ? paths.value.trim() + "\n" : "") + picked;
          } }, icon("folder"), "Add a folder…"),
          h("button", { class: "btn primary", onclick: scan }, icon("search"), "Scan")),
        h("div", { style: { maxWidth: "420px" } }, folderField("File them in")),
        results));
  }

  // ------------------------------------------------ layout
  const formSlot = h("div");
  function render() {
    replace(tabs, [["upload", "Upload files"], ["path", "From this computer"], ["scan", "Scan folders"]].map(([id, label]) =>
      h("button", { role: "tab", "aria-selected": ui.mode === id ? "true" : "false", "aria-pressed": ui.mode === id ? "true" : "false",
        onclick: () => { ui.mode = id; render(); } }, label)));
    if (ui.mode === "scan") {
      replace(content, scanPanel());
      return;
    }
    replace(formSlot, formPanel());
    replace(content, h("div", { class: "import-grid" }, ui.mode === "upload" ? uploadPanel() : pathPanel(), formSlot));
  }

  render();
  if (pending.files && pending.files.length) {
    const files = pending.files;
    pending.files = null;
    ui.mode = "upload";
    startUpload(files);
  }
  return () => { if (ui.uploading && ui.abort) ui.abort.abort(); };
}
