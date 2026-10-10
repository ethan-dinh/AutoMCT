// Library: every specimen, its group, and how far through the pipeline it is,
// sorted into the library's own folders (which move nothing on disk).

import { h, icon, replace, fmtPct, sig, relTime, statusBadge, rafThrottle, errorToast, toast, basename, slotColor, confirmDialog } from "../lib.js";
import { state, on, livePipeline, libraryFolders } from "../store.js";
import { api } from "../api.js";
import {
  track, groupLabel, flagDialog, flagTag, folderHref, folderCrumbs, pickFolder, nameFolder, parentOf, leafOf, inFolder,
} from "../components.js";
import { go } from "../app.js";

const CORE = ["segment", "cmpr", "incisor_quant", "molar_quant"];

const KIND_LABEL = {
  isq: "Scanco ISQ", rsq: "Scanco RSQ", dicom: "DICOM series", bmp: "BMP stack", volume: "Volume file",
  results: "Existing results",
};

const FORMAT_LABEL = { nrrd: "NRRD file", tiff: "TIFF file" };

// What a drag carries: specimen names, or one folder's path.
const DRAG_SPECIMENS = "application/x-studio-specimens";
const DRAG_FOLDER = "application/x-studio-folder";

function sourceText(s) {
  const kind = (s.source?.kind === "volume" && FORMAT_LABEL[s.source?.format]) || KIND_LABEL[s.source?.kind] || "No raw scan";
  const where = s.source?.mode === "link" ? "linked" : s.source?.mode === "upload" ? "uploaded" : s.source?.mode === "copy" ? "copied" : "";
  return [kind, where].filter(Boolean).join(", ");
}

function matchesStatus(s, filter) {
  const steps = livePipeline(s);
  const by = Object.fromEntries(steps.map((p) => [p.id, p.status]));
  if (filter === "running") return steps.some((p) => ["running", "queued", "waiting"].includes(p.status));
  if (filter === "failed") return steps.some((p) => p.status === "failed") || s.validation === "FAIL";
  if (filter === "excluded") return Boolean(s.excluded);
  if (filter === "complete") return CORE.every((id) => by[id] === "done");
  if (filter === "todo") return CORE.some((id) => by[id] !== "done");
  return true;
}

const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

export function mount(root, _params, query = {}) {
  const ui = { q: "", groups: new Set(), status: "all", selected: new Set(), sort: ["name", 1], dragging: false, leaving: false };
  // The folder shown: its specimens and the folders in it. "" is the top of the library.
  const folder = String(query.folder || "").split("/").map((p) => p.trim()).filter(Boolean).join("/");
  const page = h("div", { class: "page" });
  root.appendChild(page);

  const crumbs = h("div");
  const title = h("h1");
  const sub = h("div", { class: "sub" });
  const runAllBtn = h("button", { class: "btn", title: folder ? "Every specimen shown, in this folder and the folders inside it" : null,
    onclick: () => runSamples(scope().map((s) => s.name), "default") }, icon("play"), "Run missing steps");
  const folderTools = h("span", { class: "row", style: { gap: "8px" } });
  page.appendChild(h("div", { class: "page-head" },
    h("div", null, crumbs, title, sub),
    h("div", { class: "actions" }, folderTools, runAllBtn,
      h("button", { class: "btn", onclick: newFolder, title: folder ? `A new folder inside ${leafOf(folder)}` : "A new folder at the top of the library" },
        icon("plus"), "New folder"),
      h("a", { class: "btn primary", href: folder ? `#/import?folder=${encodeURIComponent(folder)}` : "#/import" }, icon("import"), "Import"))));

  const search = h("input", { type: "search", placeholder: folder ? "Search this folder  ( / )" : "Search specimens  ( / )", "aria-label": "Search specimens",
    oninput: (e) => { ui.q = e.target.value.toLowerCase(); refresh(); } });
  const groupChips = h("div", { class: "row wrap" });
  const statusSel = h("select", { "aria-label": "Filter by progress", onchange: (e) => { ui.status = e.target.value; refresh(); } },
    [["all", "All specimens"], ["todo", "Needs work"], ["running", "In the queue"], ["failed", "Failed a step or check"],
      ["complete", "Complete"], ["excluded", "Flagged and excluded"]]
      .map(([v, l]) => h("option", { value: v }, l)));
  page.appendChild(h("div", { class: "filters" }, search, statusSel, groupChips));

  const bulk = h("div");
  page.appendChild(bulk);
  const body = h("div");
  page.appendChild(body);

  const all = () => [...state.samples.values()];
  // Everything under this folder, however deep: what the counts, the group
  // chips, a search and "Run missing steps" cover.
  const under = () => all().filter((s) => inFolder(s.folder || "", folder));
  const filtering = () => Boolean(ui.q || ui.groups.size || ui.status !== "all");
  const scope = () => {
    const [key, dir] = ui.sort;
    return under().filter((s) =>
      (!ui.q || [s.name, s.group, s.folder, s.meta?.notes, s.meta?.study, s.meta?.line, s.excluded?.note].join(" ").toLowerCase().includes(ui.q)) &&
      (!ui.groups.size || ui.groups.has(s.group)) &&
      matchesStatus(s, ui.status))
      .sort((a, b) => {
        const va = key === "enamel" ? (a.metrics?.enamel_fraction ?? -1) : key === "updated" ? (a.updated || "") : key === "group" ? a.group : a.name;
        const vb = key === "enamel" ? (b.metrics?.enamel_fraction ?? -1) : key === "updated" ? (b.updated || "") : key === "group" ? b.group : b.name;
        return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
      });
  };
  // A search or filter looks through the folders inside this one too; with
  // none, the table holds this folder's own specimens under its folders.
  const visible = () => (filtering() ? scope() : scope().filter((s) => (s.folder || "") === folder));
  const childFolders = () => libraryFolders().filter((f) => parentOf(f) === folder && f !== folder);

  async function runSamples(names, steps) {
    if (!names.length) return;
    try {
      const res = await api.post("/api/run", { samples: names, steps });
      const n = res.jobs.length;
      if (n) toast(`Queued ${n} job${n === 1 ? "" : "s"} for ${names.length} specimen${names.length === 1 ? "" : "s"}.`, "good",
        { action: { label: "Open queue", onClick: () => go("#/queue") } });
      if (res.skipped.length) {
        toast(res.skipped.slice(0, 4).map((s) => `${s.sample}: ${s.reason}`).join("  "), n ? "warning" : "info", { timeout: 8000 });
      }
    } catch (err) { errorToast(err); }
  }

  // ------------------------------------------------------------ folders
  async function newFolder() {
    const name = await nameFolder({ title: folder ? `New folder in ${leafOf(folder)}` : "New folder", confirm: "Make folder",
      help: "Folders sort the library. Specimens filed in one stay where they are on disk." });
    if (!name) return;
    try {
      await api.post("/api/folders", { folder: [folder, name].filter(Boolean).join("/") });
      toast(`Made the folder ${name}. Drag specimens onto it, or select some and choose Move to.`);
    } catch (err) { errorToast(err); }
  }

  async function renameFolder(path) {
    const name = await nameFolder({ title: `Rename ${leafOf(path)}`, value: leafOf(path), confirm: "Rename" });
    if (!name || name === leafOf(path)) return;
    await refileFolder(path, [parentOf(path), name].filter(Boolean).join("/"), `Renamed ${leafOf(path)} to ${name}.`);
  }

  async function moveFolder(path) {
    const dest = await pickFolder({ title: `Move ${leafOf(path)}`, current: parentOf(path), exclude: path,
      help: "The folder moves with everything in it." });
    if (dest == null || dest === parentOf(path)) return;
    await refileFolder(path, [dest, leafOf(path)].filter(Boolean).join("/"),
      `Moved ${leafOf(path)} into ${dest ? leafOf(dest) : "the top of the library"}.`);
  }

  // Rename or move a folder; the page follows it when it is the one open.
  async function refileFolder(from, to, message) {
    const open = Boolean(folder) && inFolder(folder, from);
    ui.leaving = open;
    try {
      const res = await api.post("/api/folders/rename", { from, to });
      toast(message);
      if (open) go(folderHref(res.folder + folder.slice(from.length)));
    } catch (err) { ui.leaving = false; errorToast(err); }
  }

  async function removeFolder(path) {
    const inside = all().filter((s) => inFolder(s.folder || "", path)).length;
    const folders = libraryFolders().filter((f) => f !== path && inFolder(f, path)).length;
    const up = parentOf(path) ? leafOf(parentOf(path)) : "the top of the library";
    const held = [inside ? plural(inside, "specimen") : null, folders ? plural(folders, "folder") : null].filter(Boolean).join(" and ");
    const ok = await confirmDialog(`Remove the folder ${leafOf(path)}?`,
      held ? `The ${held} in it move up to ${up}. No specimen and no file is removed.`
        : "It is empty. Nothing else changes.", "Remove folder");
    if (!ok) return;
    const open = Boolean(folder) && inFolder(folder, path);
    ui.leaving = open;
    try {
      await api.post("/api/folders/remove", { folder: path });
      toast(`Removed the folder ${leafOf(path)}.`);
      if (open) go(folderHref(parentOf(path)));
    } catch (err) { ui.leaving = false; errorToast(err); }
  }

  async function moveSpecimens(names, dest) {
    if (!names.length) return;
    try {
      const res = await api.post("/api/folders/move", { names, folder: dest });
      const where = res.folder ? leafOf(res.folder) : "the top of the library";
      toast(`Moved ${names.length === 1 ? names[0] : plural(names.length, "specimen")} to ${where}.`, "good",
        { action: res.folder !== folder ? { label: "Open", onClick: () => go(folderHref(res.folder)) } : null });
      for (const n of names) ui.selected.delete(n);
      refresh();
    } catch (err) { errorToast(err); }
  }

  async function moveSelected() {
    const names = [...ui.selected];
    const here = new Set(names.map((n) => state.samples.get(n)?.folder || ""));
    const dest = await pickFolder({ title: `Move ${names.length === 1 ? names[0] : plural(names.length, "specimen")}`,
      current: here.size === 1 ? [...here][0] : folder });
    if (dest != null) moveSpecimens(names, dest);
  }

  // ------------------------------------------------------------ drag and drop
  // Specimen rows and folder rows can be dragged onto a folder row or a level
  // of the path above the table.
  function dragFrom(e, kind, value) {
    ui.dragging = true;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData(kind, JSON.stringify(value));
    e.dataTransfer.setData("text/plain", Array.isArray(value) ? value.join("\n") : value);
  }
  function dragEnd() {
    ui.dragging = false;
    page.querySelectorAll(".drop-over").forEach((el) => el.classList.remove("drop-over"));
    refresh();
  }
  function dropTarget(el, dest) {
    const accepts = (e) => {
      const types = [...(e.dataTransfer?.types || [])];
      return types.includes(DRAG_SPECIMENS) || types.includes(DRAG_FOLDER);
    };
    el.addEventListener("dragover", (e) => {
      if (!accepts(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      el.classList.add("drop-over");
    });
    el.addEventListener("dragleave", (e) => { if (!el.contains(e.relatedTarget)) el.classList.remove("drop-over"); });
    el.addEventListener("drop", (e) => {
      if (!accepts(e)) return;
      e.preventDefault();
      e.stopPropagation();
      const names = e.dataTransfer.getData(DRAG_SPECIMENS);
      const moving = e.dataTransfer.getData(DRAG_FOLDER);
      dragEnd();
      if (names) {
        const list = JSON.parse(names).filter((n) => (state.samples.get(n)?.folder || "") !== dest);
        if (list.length) moveSpecimens(list, dest);
      } else if (moving) {
        const path = JSON.parse(moving);
        if (inFolder(dest, path) || parentOf(path) === dest) return;
        refileFolder(path, [dest, leafOf(path)].filter(Boolean).join("/"),
          `Moved ${leafOf(path)} into ${dest ? leafOf(dest) : "the top of the library"}.`);
      }
    });
  }

  const header = (key, label, cls = "") => {
    if (!key) return h("th", { class: cls }, label);
    const [k, d] = ui.sort;
    const arrow = k === key ? (d > 0 ? " ▲" : " ▼") : "";
    return h("th", { class: cls, "aria-sort": k === key ? (d > 0 ? "ascending" : "descending") : null },
      h("button", { class: "btn ghost small", style: { padding: 0, height: "auto", fontWeight: 600 },
        onclick: () => { ui.sort = [key, k === key ? -d : 1]; refresh(); } }, label + arrow));
  };

  // One folder as a row of the table: what it holds, its groups and how far they are.
  const folderRow = (path, samples) => {
    const inside = samples.filter((s) => inFolder(s.folder || "", path));
    const folders = libraryFolders().filter((f) => parentOf(f) === path).length;
    const complete = inside.filter((s) => matchesStatus(s, "complete")).length;
    const running = inside.filter((s) => matchesStatus(s, "running")).length;
    const groups = [...new Map(inside.map((s) => [s.group, s])).values()].sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99));
    const latest = inside.reduce((m, s) => (s.updated && s.updated > m ? s.updated : m), "");
    const row = h("tr", { class: "clickable folder-row", draggable: "true", onclick: () => go(folderHref(path)),
      ondragstart: (e) => dragFrom(e, DRAG_FOLDER, path), ondragend: dragEnd },
      h("td", { class: "folder-icon" }, icon("folder")),
      h("td", null, h("div", { class: "folder-name" }, leafOf(path)),
        h("div", { class: "specimen-sub" }, [inside.length ? plural(inside.length, "specimen") : "Empty",
          folders ? plural(folders, "folder") : null].filter(Boolean).join(", "))),
      h("td", null, h("span", { class: "folder-groups", title: groups.map((g) => g.group).join(", ") },
        groups.slice(0, 6).map((g) => h("i", { class: "sw", style: { background: slotColor(g.slot) } })),
        groups.length > 6 ? h("span", { class: "muted" }, `+${groups.length - 6}`) : null)),
      h("td", { class: "muted" }, inside.length ? [`${complete} of ${inside.length} complete`,
        running ? h("span", { class: "tag accent", style: { marginLeft: "8px" } }, `${running} in the queue`) : null] : null),
      h("td", { class: "n" }), h("td", { class: "n hide-sm" }), h("td", { class: "hide-sm" }),
      h("td", { class: "hide-sm muted nowrap" }, relTime(latest)));
    dropTarget(row, path);
    return row;
  };

  const refresh = rafThrottle(() => {
    // A drag keeps the rows it started on: redrawn, they would drop it. And a
    // folder this page is following to its new name is not drawn gone meanwhile.
    if (ui.dragging || ui.leaving) return;
    const everything = all();
    const samples = under();
    const folders = libraryFolders();
    // The folder was renamed or removed elsewhere: show the nearest one still there.
    if (folder && !folders.includes(folder) && !samples.length) {
      let up = parentOf(folder);
      while (up && !folders.includes(up)) up = parentOf(up);
      go(folderHref(up));
      return;
    }
    replace(crumbs, folder ? folderCrumbs(parentOf(folder), dropTarget) : null);
    title.textContent = folder ? leafOf(folder) : "Library";
    const kids = childFolders();
    const groups = [...new Map(samples.map((s) => [s.group, s])).values()].sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99));
    sub.textContent = [
      samples.length ? `${plural(samples.length, "specimen")} in ${plural(groups.length, "group")}` : folder ? "No specimens in this folder yet" : "No specimens yet",
      kids.length ? plural(kids.length, "folder") : null,
    ].filter(Boolean).join(", ");
    runAllBtn.disabled = !samples.length;
    replace(folderTools, folder ? [
      h("button", { class: "btn ghost", onclick: () => renameFolder(folder), title: "Rename this folder" }, icon("edit"), "Rename"),
      h("button", { class: "btn ghost", onclick: () => moveFolder(folder), title: "Move this folder into another" }, "Move…"),
      h("button", { class: "btn ghost danger", onclick: () => removeFolder(folder), title: "Remove the folder; what it holds moves up a level" },
        icon("trash"), "Remove folder"),
    ] : null);
    replace(groupChips, groups.map((g) => h("button", {
      class: "chip", "aria-pressed": ui.groups.has(g.group) ? "true" : "false",
      onclick: () => { ui.groups.has(g.group) ? ui.groups.delete(g.group) : ui.groups.add(g.group); refresh(); },
    }, h("i", { class: "sw", style: { background: slotColor(g.slot) } }), g.group)));

    for (const name of [...ui.selected]) if (!state.samples.has(name)) ui.selected.delete(name);
    if (ui.selected.size) {
      const chosen = [...ui.selected];
      const flagged = chosen.filter((n) => state.samples.get(n)?.excluded);
      const flag = async () => {
        // One flagged specimen's flag is edited as it is; a mixed selection gets a fresh one.
        const current = chosen.length === 1 ? state.samples.get(chosen[0])?.excluded : null;
        const who = chosen.length === 1 ? chosen[0] : `${chosen.length} specimens`;
        const outcome = await flagDialog(chosen, current);
        if (outcome === "flagged") toast(`Flagged ${who}: left out of the results.`);
        else if (outcome === "included") toast(`${who} back in the results.`);
      };
      const include = async () => {
        try {
          await api.post("/api/flags", { names: flagged, flag: null });
          toast(`${flagged.length === 1 ? flagged[0] : `${flagged.length} specimens`} back in the results.`);
        } catch (err) { errorToast(err); }
      };
      const stepSel = h("select", { "aria-label": "Step to run" },
        (state.app?.steps || []).map((s) =>
          h("option", { value: s.id, disabled: Boolean(s.disabled) }, s.disabled ? `${s.label} (turned off)` : s.label)));
      replace(bulk, h("div", { class: "bulkbar" },
        h("b", null, `${ui.selected.size} selected`),
        h("button", { class: "btn small primary", onclick: () => runSamples([...ui.selected], "default") }, "Run missing steps"),
        h("span", { class: "muted" }, "or"), stepSel,
        h("button", { class: "btn small", onclick: () => runSamples([...ui.selected], [stepSel.value]) }, "Run"),
        h("span", { class: "divider", "aria-hidden": "true" }),
        h("button", { class: "btn small", onclick: moveSelected, title: "File them in a folder of the library" }, icon("folder"), "Move to…"),
        h("button", { class: "btn small", onclick: flag, title: "Leave them out of the results, with a note" }, icon("flag"), "Flag…"),
        flagged.length ? h("button", { class: "btn small", onclick: include }, `Include ${flagged.length === chosen.length ? "" : `${flagged.length} `}again`) : null,
        h("span", { class: "divider", "aria-hidden": "true" }),
        h("button", { class: "btn small", disabled: chosen.length > 4,
          title: chosen.length > 4 ? "Four at most" : "Side by side in 3-D, turning together, on one heatmap scale",
          onclick: () => go(`#/compare?s=${chosen.map(encodeURIComponent).join(",")}`) }, icon("compare"), "Compare in 3-D"),
        h("button", { class: "btn small", title: "The Results page's charts and statistics for these specimens alone",
          onclick: () => go(`#/results/all?s=${chosen.map(encodeURIComponent).join(",")}`) }, icon("results"), "Show in Results"),
        h("span", { class: "spacer" }),
        h("button", { class: "btn small ghost", onclick: () => { ui.selected.clear(); refresh(); } }, "Clear selection")));
    } else replace(bulk);

    if (!everything.length && !folders.length) {
      replace(body, h("a", { class: "dropzone", href: "#/import" },
        icon("upload"), h("h3", null, "Drop a scan anywhere on this page"),
        h("span", null, "An ISQ or NRRD file, a DICOM or BMP folder, or an existing segmentation folder."),
        h("span", { class: "btn primary" }, "Import specimens")));
      return;
    }
    const rows = visible();
    const shownFolders = filtering() ? [] : kids;
    const rowOf = (s) => {
      // A specimen found by a search in a folder inside this one says which.
      const elsewhere = (s.folder || "") !== folder ? (s.folder || "").slice(folder ? folder.length + 1 : 0) : "";
      return h("tr", {
        class: `clickable ${ui.selected.has(s.name) ? "selected" : ""}`, draggable: "true",
        onclick: () => go(`#/sample/${encodeURIComponent(s.name)}`),
        // Dragging a selected row takes the whole selection with it.
        ondragstart: (e) => dragFrom(e, DRAG_SPECIMENS, ui.selected.has(s.name) ? [...ui.selected] : [s.name]),
        ondragend: dragEnd,
      },
        h("td", { onclick: (e) => e.stopPropagation() }, h("input", { type: "checkbox", "aria-label": `Select ${s.name}`, checked: ui.selected.has(s.name),
          onchange: (e) => { e.target.checked ? ui.selected.add(s.name) : ui.selected.delete(s.name); refresh(); } })),
        h("td", null, h("div", { class: "specimen-line" }, h("span", { class: "specimen" }, s.name), flagTag(s.excluded)),
          h("div", { class: "specimen-sub", title: s.excluded?.note || s.source?.path || "" },
            elsewhere ? h("span", { class: "in-folder" }, icon("folder"), elsewhere.split("/").join(" › ")) : null,
            s.excluded ? (s.excluded.note || "Flagged, with no note") : [sourceText(s), s.source?.path ? `: ${basename(s.source.path)}` : ""])),
        h("td", null, groupLabel(s)),
        h("td", null, track(s)),
        h("td", { class: "n" }, s.metrics?.enamel_fraction != null ? fmtPct(s.metrics.enamel_fraction) : h("span", { class: "muted" }, "–")),
        h("td", { class: "n hide-sm" }, s.metrics?.peak_ratio != null
          ? h("span", { title: s.metrics.crossover_z != null ? `Crosses 1.0 at ${sig(s.metrics.crossover_z, 3)}% along the tooth` : "" },
            `${sig(s.metrics.peak_ratio, 3)} peak`)
          : h("span", { class: "muted" }, "–")),
        h("td", { class: "hide-sm" }, s.validation ? statusBadge(s.validation) : h("span", { class: "muted" }, "–")),
        h("td", { class: "hide-sm muted nowrap" }, relTime(s.updated)));
    };
    // Flagged specimens below the rest, under a heading of their own, in the same columns.
    const kept = rows.filter((s) => !s.excluded);
    const flagged = rows.filter((s) => s.excluded);
    const allChecked = rows.length && rows.every((s) => ui.selected.has(s.name));
    const table = h("table", { class: "data" },
      h("thead", null, h("tr", null,
        h("th", { style: { width: "32px" } }, h("input", { type: "checkbox", "aria-label": "Select all shown", checked: allChecked, disabled: !rows.length,
          onchange: (e) => { rows.forEach((s) => e.target.checked ? ui.selected.add(s.name) : ui.selected.delete(s.name)); refresh(); } })),
        header("name", "Specimen"), header("group", "Group"), header(null, "Pipeline"),
        header("enamel", "Molar enamel", "n"), header(null, "Incisor ratio", "n hide-sm"),
        header(null, "Checks", "hide-sm"), header("updated", "Updated", "hide-sm"))),
      shownFolders.length ? h("tbody", { class: "folders" }, shownFolders.map((f) => folderRow(f, samples))) : null,
      h("tbody", null, kept.map(rowOf)),
      flagged.length ? h("tbody", { class: "flagged-group" },
        h("tr", { class: "group-head" }, h("th", { colspan: 8, scope: "rowgroup" },
          h("span", { class: "group-title" }, icon("flag"), "Flagged"),
          h("span", { class: "muted" }, `${flagged.length} specimen${flagged.length === 1 ? "" : "s"} left out of the results. Select one and choose Include again to bring it back.`))),
        flagged.map(rowOf)) : null);
    if (rows.length || shownFolders.length) {
      replace(body, h("div", { class: "table-wrap" }, table),
        filtering() && kids.length ? h("p", { class: "muted", style: { fontSize: "12.5px", marginTop: "8px" } },
          folder ? "Searching this folder and the folders inside it." : "Searching every folder of the library.") : null);
    } else if (filtering()) {
      replace(body, h("div", { class: "empty" }, h("h3", null, folder ? "Nothing in this folder matches these filters" : "Nothing matches these filters"),
        h("button", { class: "btn", onclick: () => { ui.q = ""; search.value = ""; ui.groups.clear(); ui.status = "all"; statusSel.value = "all"; refresh(); } }, "Clear filters")));
    } else {
      const empty = h("div", { class: "empty" }, h("h3", null, "This folder is empty"),
        h("span", null, "Drag specimens onto it from another folder, select some and choose Move to, or import straight into it."),
        h("div", { class: "row" },
          h("a", { class: "btn primary", href: `#/import?folder=${encodeURIComponent(folder)}` }, icon("import"), "Import into this folder"),
          h("a", { class: "btn", href: folderHref(parentOf(folder)) }, icon("back"), parentOf(folder) ? `Back to ${leafOf(parentOf(folder))}` : "Back to the library")));
      replace(body, empty);
    }
  });

  const offs = [on("samples", refresh), on("jobs", refresh), on("app", refresh), on("folders", refresh)];
  refresh();
  const timer = setInterval(refresh, 60_000);  // keep "updated" times honest
  return () => { offs.forEach((f) => f()); clearInterval(timer); };
}
