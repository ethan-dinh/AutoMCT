// Workspaces: choose the folder the studio works in, when it starts and
// whenever you switch. A workspace on an external exFAT drive opens on a Mac
// and on a PC; the studio finds it on either, wherever the drive is mounted.

import { h, icon, replace, errorToast, toast, relTime, modal, confirmDialog, basename } from "../lib.js";
import { api, enc } from "../api.js";
import { state, loadAll } from "../store.js";
import { pickPath } from "../components.js";
import { go } from "../app.js";

const DEFAULT_NAME = "MicroCT-Studio";

function joinPath(parent, name) {
  const sep = parent.includes("\\") && !parent.includes("/") ? "\\" : "/";
  return parent.endsWith(sep) ? parent + name : parent + sep + name;
}

// What a drive's format means for opening the workspace on both a Mac and a PC.
function sharingNote(note) {
  if (!note?.text || note.tone === "good") return null;
  return h("div", { class: `ws-note ${note.tone}` }, icon(note.tone === "warning" ? "alert" : "info"), h("span", null, note.text));
}

function sharingTag(ws) {
  if (ws.sharing?.tone === "good") return h("span", { class: "tag accent", title: ws.sharing.text }, "Opens on Mac and PC");
  return null;
}

export async function mount(root) {
  const current = state.app?.workspace;
  const bare = !current?.available;
  const page = h("div", { class: `page ws-page${bare ? " ws-bare" : ""}` });
  root.appendChild(page);
  page.appendChild(h("header", { class: "ws-head" },
    bare ? h("div", { class: "ws-brand" },
      h("svg:svg", { viewBox: "0 0 32 32", "aria-hidden": "true" }, h("svg:rect", { width: 32, height: 32, rx: 7 }),
        h("svg:path", { d: "M9 23c1-8 4-13 8-14 3-1 5 1 6 4" })), "MicroCT Studio") : null,
    h("h1", null, bare ? "Choose a workspace" : "Workspaces"),
    h("p", { class: "sub" }, "A workspace is the folder that holds your specimens, their results and the studio's settings. "
      + "Keep it on an external drive formatted as exFAT to open it from both a Mac and a PC.")));
  if (current && !current.available) {
    page.appendChild(h("div", { class: "notice warning" },
      `The drive that holds ${current.name} is not connected. Plug it in and open it again, or choose another workspace.`));
  }
  const body = h("div", { class: "ws-body" },
    h("div", { class: "ws-loading" }, h("div", { class: "progress indeterminate" }, h("i")),
      h("span", { class: "muted" }, "Looking for workspaces on this computer and its drives…")));
  page.appendChild(body);
  let data = null;
  // The workspace being opened. Opening takes a moment (the server reads the
  // whole workspace), so the list waits: one open at a time, and the row says so.
  let opening = null;

  async function load() {
    try {
      data = await api.get("/api/workspaces");
      render();
    } catch (err) { errorToast(err); }
  }

  function setOpening(path) {
    opening = path;
    if (data) render();
  }

  async function open(path, how = {}) {
    if (opening) return;
    setOpening(path);
    try {
      await api.post("/api/workspace", { path, ...how });
      await loadAll();
      toast(`Opened ${basename(path)}.`);
      go("#/");
    } catch (err) {
      setOpening(null);
      if (err.data?.code === "not_workspace") return offerToCreate(path, err.data);
      errorToast(err);
    }
  }

  async function offerToCreate(path, info) {
    const inside = !info.empty;
    const ok = await confirmDialog("That folder is not a workspace",
      inside ? `${path} holds other files. Create a new workspace inside it, named ${DEFAULT_NAME}?`
        : `Make ${path} a new, empty workspace?`, "Create workspace");
    if (!ok || opening) return;
    setOpening(path);
    try {
      await api.post("/api/workspace", inside ? { parent: path, name: DEFAULT_NAME } : { path, create: true });
      await loadAll();
      toast("Workspace created.");
      go("#/");
    } catch (err) { setOpening(null); errorToast(err); }
  }

  // Off the list only: the folder and everything in it stay where they are.
  async function remove(ws) {
    try {
      const undo = await api.post("/api/workspaces/remove", { path: ws.path, id: ws.id });
      toast(`Removed ${ws.name} from the list. Its folder is untouched.`, "good", { action: { label: "Undo", onClick: async () => {
        try { await api.post("/api/workspaces/restore", undo); load(); } catch (err) { errorToast(err); }
      } } });
      load();
    } catch (err) { errorToast(err); }
  }

  async function openOther() {
    if (opening) return;
    const path = await pickPath({ title: "Open a workspace", mode: "dir", start: data?.places?.[0]?.path || "",
      help: "Choose the workspace folder itself, the one that holds studio.json and a samples folder. Workspaces are tagged in the list." });
    if (path) open(path);
  }

  function createDialog() {
    const places = data?.places || [];
    let parent = (places.find((p) => p.label !== "Home folder") || places[0])?.path || "";
    let note = (places.find((p) => p.path === parent) || {}).sharing || null;
    const nameInput = h("input", { type: "text", value: DEFAULT_NAME, "aria-label": "Workspace name" });
    const where = h("div", { class: "stack", style: { gap: "10px" } });
    const draw = () => {
      replace(where,
        h("div", { class: "row wrap" },
          places.map((p) => h("button", { type: "button", class: "chip", "aria-pressed": p.path === parent ? "true" : "false",
            onclick: () => { parent = p.path; note = p.sharing; draw(); } },
          icon(p.label === "Home folder" ? "folder" : "drive"), p.label)),
          h("button", { type: "button", class: "btn small ghost", onclick: async () => {
            const picked = await pickPath({ title: "Create the workspace in…", mode: "dir", start: parent,
              help: "Choose the folder to create the workspace in." });
            if (!picked) return;
            parent = picked;
            try { note = (await api.get(`/api/fs/inspect?path=${enc(picked)}`)).sharing || null; } catch { note = null; }
            draw();
          } }, "Another folder…")),
        h("div", { class: "soft" }, "Creates ", h("code", null, joinPath(parent, nameInput.value.trim() || DEFAULT_NAME))),
        sharingNote(note) || (note?.tone === "good" ? h("div", { class: "ws-note good" }, icon("checkCircle"), h("span", null, note.text)) : null));
    };
    nameInput.addEventListener("input", draw);
    modal({
      title: "Create a workspace", wide: true,
      body: h("div", { class: "stack" },
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Name"), nameInput),
        h("div", { class: "field" }, h("span", { class: "lbl" }, "Where"), where)),
      actions: [
        { label: "Cancel" },
        { label: "Create and open", primary: true, onClick: async () => {
          if (opening) return false;
          setOpening(joinPath(parent, nameInput.value.trim() || DEFAULT_NAME));
          try {
            await api.post("/api/workspace", { parent, name: nameInput.value.trim() || DEFAULT_NAME });
            await loadAll();
            toast("Workspace created.");
            go("#/");
          } catch (err) { setOpening(null); errorToast(err); return false; }
          return true;
        } },
      ],
    });
    draw();
  }

  function row(ws, primary) {
    const facts = [
      h("span", { class: "tag" }, icon(ws.drive ? "drive" : "folder"), ws.drive || "This computer"),
      ws.available ? h("span", null, `${ws.samples} specimen${ws.samples === 1 ? "" : "s"}`) : h("span", null, "Not connected"),
      ws.opened ? h("span", { class: "muted" }, `Opened ${relTime(ws.opened)}`) : null,
      ws.available ? sharingTag(ws) : null,
    ];
    return h("li", { class: `ws-row${ws.available ? "" : " off"}${ws.current ? " current" : ""}` },
      h("div", { class: "ws-main" },
        h("div", { class: "ws-name" }, ws.name, ws.current ? h("span", { class: "tag accent" }, "Open now") : null),
        h("code", { class: "ws-path", title: ws.path }, `‎${ws.path}`),
        h("div", { class: "ws-facts" }, facts),
        ws.available ? sharingNote(ws.sharing) : null),
      h("div", { class: "ws-act" },
        ws.current ? h("a", { class: "btn", href: "#/" }, "Back to it")
          : opening === ws.path
            ? h("button", { class: `btn busy${primary ? " primary" : ""}`, disabled: true, "aria-busy": "true" },
              h("span", { class: "spinner", "aria-hidden": "true" }), "Opening…")
            : h("button", { class: `btn${primary ? " primary" : ""}`, disabled: !ws.available || Boolean(opening),
              onclick: () => open(ws.path) }, ws.available ? "Open" : "Not connected"),
        ws.current ? null : h("button", { class: "btn ghost icon", title: "Remove from the list", disabled: Boolean(opening),
          "aria-label": `Remove ${ws.name} from the list`, onclick: () => remove(ws) }, icon("x"))));
  }

  function render() {
    const recent = data.recent || [];
    const found = data.found || [];
    const firstOpenable = [...recent, ...found].find((w) => w.available && !w.current);
    const section = (title, list) => list.length ? h("section", { class: "ws-section" }, h("h2", null, title),
      h("ul", { class: "ws-list" }, list.map((w) => row(w, w === firstOpenable)))) : null;
    const wait = Boolean(opening);
    const actions = h("div", { class: "ws-actions" },
      h("button", { class: `btn${firstOpenable ? "" : " primary"}`, disabled: wait, onclick: createDialog }, icon("plus"), "Create a workspace"),
      h("button", { class: "btn", disabled: wait, onclick: openOther }, icon("folder"), "Open another folder"),
      h("button", { class: "btn ghost", disabled: wait, onclick: () => { replace(body, h("div", { class: "ws-loading" },
        h("div", { class: "progress indeterminate" }, h("i")))); load(); } }, icon("refresh"), "Look again"));
    if (!recent.length && !found.length) {
      replace(body, h("div", { class: "ws-empty" },
        h("h2", null, "No workspace yet"),
        h("p", null, "Create one to get started. To use it from both your Mac and your PC, put it on an external drive "
          + "formatted as exFAT; plug the drive into either computer and choose it here."),
        actions));
      return;
    }
    // Opened from "Open another folder" or a create dialog, the folder has no row of its own.
    const listed = [...recent, ...found].some((w) => w.path === opening);
    const status = opening && !listed ? h("div", { class: "ws-opening", role: "status" },
      h("span", { class: "spinner", "aria-hidden": "true" }), h("span", null, `Opening ${basename(opening)}…`)) : null;
    replace(body, status, section("Recent on this computer", recent), section("Found on connected drives", found), actions);
  }

  load();
  return null;
}
