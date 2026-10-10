// Client state, kept current by the server's event stream. Views subscribe to
// topics ("samples", "jobs", "settings", "folders") and re-render only what changed.

import { api, connectEvents } from "./api.js";

export const state = {
  app: null,            // /api/state: steps, tools, workspace, labels
  samples: new Map(),   // name -> summary
  jobs: new Map(),      // id -> job
  connected: false,
};

const subs = new Map();

export function on(topic, fn) {
  if (!subs.has(topic)) subs.set(topic, new Set());
  subs.get(topic).add(fn);
  return () => subs.get(topic).delete(fn);
}

export function emit(topic, detail) {
  for (const fn of subs.get(topic) || []) {
    try { fn(detail); } catch (err) { console.error(err); }
  }
}

export async function loadAll() {
  const app = await api.get("/api/state");
  let samples = [], jobs = [];
  // With no workspace open (or its drive unplugged) there is nothing else to load.
  if (app.workspace?.available) [samples, jobs] = await Promise.all([api.get("/api/samples"), api.get("/api/jobs")]);
  state.app = app;
  state.samples = new Map(samples.map((s) => [s.name, s]));
  state.jobs = new Map(jobs.map((j) => [j.id, j]));
  emit("app"); emit("samples"); emit("jobs");
}

export const hasWorkspace = () => Boolean(state.app?.workspace?.available);

export function start() {
  connectEvents({
    job: (job) => { state.jobs.set(job.id, job); emit("jobs", job); },
    jobs: async () => { const jobs = await api.get("/api/jobs"); state.jobs = new Map(jobs.map((j) => [j.id, j])); emit("jobs"); },
    sample: (summary) => { state.samples.set(summary.name, summary); emit("samples", summary); },
    removed: ({ name }) => { state.samples.delete(name); emit("samples"); },
    renamed: ({ from }) => { state.samples.delete(from); emit("samples"); },
    settings: (settings) => { if (state.app) state.app.workspace.settings = settings; emit("settings"); },
    folders: ({ folders }) => { if (state.app?.workspace) state.app.workspace.folders = folders; emit("folders"); },
    workspace: () => loadAll(),
  }, (up) => {
    const was = state.connected;
    state.connected = up;
    if (up !== was) emit("connection", up);
    if (up && !was && state.app) loadAll();
  });
}

export const step = (id) => state.app?.steps.find((s) => s.id === id);

export function activeJobsFor(sample) {
  return [...state.jobs.values()].filter((j) => j.sample === sample && ["queued", "waiting", "running"].includes(j.status));
}

export function runningCount() {
  let n = 0;
  for (const j of state.jobs.values()) if (["queued", "waiting", "running"].includes(j.status)) n++;
  return n;
}

// The live status of each pipeline step: the server's summary, with any
// running job's progress laid over it as events arrive.
export function livePipeline(summary) {
  const active = activeJobsFor(summary.name);
  return (summary.pipeline || []).map((p) => {
    const job = active.find((j) => j.step === p.id);
    return job ? { ...p, status: job.status, progress: job.progress, stage: job.stage, job: job.id } : p;
  });
}

// The library's folders, as full paths ("Odam study/Batch 2"), parents before what they hold.
export const libraryFolders = () => state.app?.workspace?.folders || [];

export function groupColorSlot(group) {
  const slots = state.app?.workspace?.settings?.group_colors || {};
  return slots[group];
}

export function previewJob(sample, view) {
  return [...state.jobs.values()].find((j) => j.step === "preview" && j.sample === sample &&
    j.params?.view === view && ["queued", "waiting", "running"].includes(j.status));
}
