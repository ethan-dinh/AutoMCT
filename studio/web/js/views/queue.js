// Queue: what is running, what is waiting, what finished, and each job's log.

import { h, icon, replace, statusBadge, fmtDuration, relTime, elapsedSince, errorToast, toast, rafThrottle, confirmDialog } from "../lib.js";
import { api } from "../api.js";
import { state, on } from "../store.js";
import { go } from "../app.js";

const LANE_LABEL = { heavy: "Heavy", preview: "Preview", light: "Light" };

export function mount(root, [selectedId]) {
  const page = h("div", { class: "page" });
  root.appendChild(page);
  const sub = h("div", { class: "sub" });
  const lanes = h("div", { class: "row wrap" });
  page.appendChild(h("div", { class: "page-head" },
    h("div", null, h("h1", null, "Queue"), sub),
    h("div", { class: "actions" }, lanes,
      h("button", { class: "btn", onclick: async () => {
        if (!(await confirmDialog("Clear finished jobs?", "Their records and logs are deleted. Results they produced are kept.", "Clear"))) return;
        try { const r = await api.post("/api/jobs/clear"); toast(`Cleared ${r.cleared} job${r.cleared === 1 ? "" : "s"}.`); }
        catch (err) { errorToast(err); }
      } }, "Clear finished"))));

  const list = h("div");
  const logPanel = h("div");
  page.appendChild(h("div", { class: "queue-grid" }, list, logPanel));

  let selected = selectedId || null;

  function jobRow(j) {
    const active = ["running", "queued", "waiting"].includes(j.status);
    const timing = j.status === "running" ? `running for ${fmtDuration(elapsedSince(j.started))}`
      : j.finished && j.started ? `took ${fmtDuration((new Date(j.finished) - new Date(j.started)) / 1000)}, ${relTime(j.finished)}`
        : j.status === "waiting" ? "waiting for the steps before it" : `queued ${relTime(j.created)}`;
    return h("div", { class: `job ${selected === j.id ? "selected" : ""}` },
      h("div", null,
        h("div", { class: "title" }, h("a", { href: `#/queue/${j.id}`, onclick: (e) => { e.preventDefault(); select(j.id); } }, j.title)),
        h("div", { class: "who" }, j.sample ? h("a", { href: `#/sample/${encodeURIComponent(j.sample)}` }, j.sample) : "Tool",
          `, ${LANE_LABEL[j.lane] || j.lane} lane`)),
      h("div", { class: "stack", style: { gap: "4px", minWidth: 0 } },
        statusBadge(j.status, j.status === "running" && j.progress != null ? `${Math.round(j.progress * 100)}%` : null, j.progress),
        j.status === "running" ? h("div", { class: `progress ${j.progress == null ? "indeterminate" : ""}` },
          h("i", { style: { width: `${((j.progress || 0) * 100).toFixed(1)}%` } })) : null,
        h("div", { class: "when" }, j.status === "running" && j.stage ? `${j.stage}, ` : "", timing)),
      h("div", { class: "row" },
        active ? h("button", { class: "btn small", onclick: () => api.post(`/api/jobs/${j.id}/cancel`).catch(errorToast) }, "Cancel") : null,
        ["failed", "cancelled"].includes(j.status) ? h("button", { class: "btn small", onclick: async () => {
          try { const nj = await api.post(`/api/jobs/${j.id}/retry`); select(nj.id); } catch (err) { errorToast(err); }
        } }, icon("refresh"), "Retry") : null,
        h("button", { class: "btn small ghost", onclick: () => select(j.id) }, "Log")),
      j.error && j.status !== "done" ? h("div", { class: "err" }, j.error) : null);
  }

  const renderList = rafThrottle(() => {
    const jobs = [...state.jobs.values()].sort((a, b) => (a.created < b.created ? 1 : -1));
    const running = jobs.filter((j) => j.status === "running");
    const waiting = jobs.filter((j) => ["queued", "waiting"].includes(j.status)).reverse();
    const done = jobs.filter((j) => ["done", "failed", "cancelled"].includes(j.status)).slice(0, 120);
    sub.textContent = `${running.length} running, ${waiting.length} waiting`;
    const settings = state.app?.workspace?.settings?.lanes || {};
    replace(lanes, h("span", { class: "muted" }, "At once:"), ["heavy", "preview"].map((lane) =>
      h("label", { class: "row", title: lane === "heavy" ? "Conversion, segmentation, CMPR and quantification hold whole scans in memory." : "Viewer previews" },
        h("span", null, LANE_LABEL[lane]),
        h("select", { onchange: async (e) => {
          try { await api.patch("/api/settings", { lanes: { [lane]: Number(e.target.value) } }); } catch (err) { errorToast(err); }
        } }, [1, 2, 3, 4].map((n) => h("option", { value: n, selected: Number(settings[lane] || 1) === n }, n))))));
    const section = (title, items, empty) => h("div", { class: "section", style: { marginTop: "0", marginBottom: "18px" } },
      h("div", { class: "section-head" }, h("h2", null, title), h("span", { class: "sub" }, items.length ? `${items.length}` : "")),
      items.length ? h("div", { class: "panel jobs" }, items.map(jobRow)) : h("p", { class: "muted" }, empty));
    replace(list,
      section("Running", running, "Nothing is running."),
      section("Waiting", waiting, "Nothing is waiting."),
      section("Finished", done, "No finished jobs yet."));
  });

  // ------------------------------------------------------------ log
  let logTimer = null, logOffset = 0, logJob = null;
  const pre = h("pre", { class: "log", "aria-live": "off" });
  function lineClass(line) {
    if (line.startsWith("[studio]")) return "l-studio";
    if (/\b(ERROR|CRITICAL|Traceback|Exception)\b/.test(line)) return "l-err";
    if (/\bWARN(ING)?\b/.test(line)) return "l-warn";
    return "";
  }
  async function pollLog() {
    if (!logJob) return;
    try {
      const res = await api.get(`/api/jobs/${logJob}/log?offset=${logOffset}`);
      if (res.text) {
        const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
        for (const line of res.text.split("\n")) {
          if (!line) continue;
          const span = document.createElement("span");
          const cls = lineClass(line);
          if (cls) span.className = cls;
          span.textContent = line + "\n";
          pre.appendChild(span);
        }
        logOffset = res.offset;
        if (atBottom) pre.scrollTop = pre.scrollHeight;
      }
    } catch { /* job cleared */ }
    const job = state.jobs.get(logJob);
    const live = job && ["running", "queued", "waiting"].includes(job.status);
    logTimer = setTimeout(pollLog, live ? 700 : 4000);
  }
  function select(id) {
    selected = id;
    if (location.hash !== `#/queue/${id}`) history.replaceState(null, "", `#/queue/${id}`);
    renderList();
    openLog(id);
  }
  function openLog(id) {
    clearTimeout(logTimer);
    logJob = id; logOffset = 0;
    replace(pre);
    const job = state.jobs.get(id);
    if (!job) { replace(logPanel, h("p", { class: "muted" }, "Select a job to see its log.")); return; }
    replace(logPanel, h("div", { class: "stack" },
      h("div", { class: "row" }, h("h2", { class: "grow" }, job.title),
        job.sample ? h("button", { class: "btn small", onclick: () => go(`#/sample/${encodeURIComponent(job.sample)}`) }, job.sample) : null),
      job.argv?.length ? h("details", null, h("summary", { class: "muted" }, "Command"), h("code", { class: "soft", style: { wordBreak: "break-all" } }, job.argv.join(" "))) : null,
      pre));
    pollLog();
  }

  const offs = [on("jobs", renderList), on("settings", renderList)];
  renderList();
  if (selected) openLog(selected);
  else {
    const first = [...state.jobs.values()].find((j) => j.status === "running");
    if (first) select(first.id); else replace(logPanel, h("p", { class: "muted" }, "Select a job to see its log."));
  }
  const tick = setInterval(renderList, 5000);
  return () => { offs.forEach((f) => f()); clearTimeout(logTimer); clearInterval(tick); };
}
