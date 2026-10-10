// The server API, the live event stream, and file uploads.

export class ApiError extends Error {
  constructor(message, status, data = null) {
    super(message);
    this.status = status;
    this.data = data;  // the server's reply, e.g. { code: "not_workspace" }
  }
}

async function request(method, url, body) {
  const init = { method, headers: {} };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(url, init);
  } catch (err) {
    throw new ApiError("The studio server is not responding. Is it still running?", 0);
  }
  const type = response.headers.get("content-type") || "";
  const data = type.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const code = data && typeof data === "object" ? data.code : null;
    // The workspace closed or its drive was unplugged: the app switches to the chooser.
    if (code === "no_workspace" || code === "workspace_missing") window.dispatchEvent(new CustomEvent("studio:workspace-lost"));
    throw new ApiError((data && data.error) || response.statusText, response.status, typeof data === "object" ? data : null);
  }
  return data;
}

export const api = {
  get: (url) => request("GET", url),
  post: (url, body = {}) => request("POST", url, body),
  patch: (url, body = {}) => request("PATCH", url, body),
  del: (url) => request("DELETE", url),
};

export const enc = encodeURIComponent;

export async function fetchBinary(url, Type) {
  const response = await fetch(url);
  if (!response.ok) throw new ApiError(`Could not load ${url} (${response.status})`, response.status);
  return new Type(await response.arrayBuffer());
}

// ---------------------------------------------------------------- events

export function connectEvents(handlers, onState) {
  let source = null;
  const open = () => {
    source = new EventSource("/api/events");
    source.addEventListener("hello", () => onState(true));
    source.onerror = () => onState(false);
    for (const [kind, fn] of Object.entries(handlers)) {
      source.addEventListener(kind, (e) => {
        try { fn(JSON.parse(e.data)); } catch (err) { console.error(kind, err); }
      });
    }
  };
  open();
  return () => source && source.close();
}

// ---------------------------------------------------------------- uploads

// Files collected from a drop or a picker, each with the path it had inside
// the dropped folder, so a DICOM or BMP folder arrives with its structure.
export async function filesFromDrop(dataTransfer) {
  const out = [];
  const items = [...(dataTransfer.items || [])].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  if (!items.length) {
    for (const f of dataTransfer.files || []) out.push({ file: f, path: f.name });
    return out;
  }
  const walk = async (entry, prefix) => {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      if (!file.name.startsWith(".")) out.push({ file, path: prefix + file.name });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
      } while (batch.length);
    }
  };
  for (const entry of items) await walk(entry, "");
  return out;
}

export function filesFromInput(input) {
  return [...input.files].filter((f) => !f.name.startsWith(".")).map((f) => ({ file: f, path: f.webkitRelativePath || f.name }));
}

export async function uploadFiles(files, onProgress, signal) {
  const { id } = await api.post("/api/uploads");
  const total = files.reduce((s, f) => s + f.file.size, 0) || 1;
  const sent = new Map();
  const report = () => onProgress([...sent.values()].reduce((a, b) => a + b, 0), total);
  const putOne = ({ file, path }) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const url = `/api/uploads/${id}/${path.split("/").map(encodeURIComponent).join("/")}`;
    xhr.open("PUT", url);
    xhr.upload.onprogress = (e) => { sent.set(path, e.loaded); report(); };
    xhr.onload = () => (xhr.status < 300 ? (sent.set(path, file.size), report(), resolve()) :
      reject(new ApiError(`Upload of ${path} failed (${xhr.status})`, xhr.status)));
    xhr.onerror = () => reject(new ApiError(`Upload of ${path} failed`, 0));
    if (signal) signal.addEventListener("abort", () => xhr.abort());
    xhr.send(file);
  });
  // Big files go one at a time (bandwidth, not latency, is the limit); small
  // ones -- a DICOM series is thousands -- go several at once.
  const queue = [...files];
  const lanes = files.length > 8 ? 6 : 1;
  const worker = async () => { while (queue.length) { if (signal?.aborted) return; await putOne(queue.shift()); } };
  await Promise.all(Array.from({ length: lanes }, worker));
  return id;
}
