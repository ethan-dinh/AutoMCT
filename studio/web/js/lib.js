// DOM building, formatting, icons, toasts and dialogs shared by every view.
// Untrusted text (sample names, paths, log lines) only ever goes in through
// textContent, never innerHTML.

export function h(tag, props, ...children) {
  const el = tag.startsWith("svg:")
    ? document.createElementNS("http://www.w3.org/2000/svg", tag.slice(4))
    : document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") el.setAttribute("class", value);
      else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
      else if (key === "dataset") Object.assign(el.dataset, value);
      else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2).toLowerCase(), value);
      else if (key === "text") el.textContent = value;
      else if (key === "value" && "value" in el) el.value = value;
      else if (key === "checked" && "checked" in el) el.checked = !!value;
      else if (value === true) el.setAttribute(key, "");
      else el.setAttribute(key, value);
    }
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function replace(el, ...children) {
  clear(el);
  return append(el, children);
}

// ---------------------------------------------------------------- icons

const ICONS = {
  library: "M4 5h16M4 12h16M4 19h10",
  import: "M12 4v11m0 0-4-4m4 4 4-4M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3",
  viewer: "M3 12s3.5-6 9-6 9 6 9 6-3.5 6-9 6-9-6-9-6Zm9 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z",
  results: "M4 20V10m6 10V4m6 16v-7m4 7H3",
  compare: "M4 5h5a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Zm11 0h5a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Z",
  queue: "M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  tools: "M14.7 6.3a4 4 0 0 0-5.4 5.4L4 17l3 3 5.3-5.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.1-.4-.4-2.1 2.5-2.5Z",
  settings: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm7.4-3a7.4 7.4 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7.3 7.3 0 0 0-2-1.2L14.5 3h-5l-.4 2.6a7.3 7.3 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 0 0 0 2.4l-2 1.6 2 3.4 2.4-1a7.3 7.3 0 0 0 2 1.2l.4 2.6h5l.4-2.6a7.3 7.3 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2Z",
  check: "M5 12.5 10 17l9-10",
  checkCircle: "M8 12.5 11 15.5 16.5 9.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  x: "M6 6l12 12M18 6 6 18",
  xCircle: "M9 9l6 6m0-6-6 6m12-3a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  alert: "M12 9v4m0 4h.01M10.3 3.9 2.4 17.5A2 2 0 0 0 4.1 20.5h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z",
  info: "M12 11v5m0-8h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  clock: "M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  circle: "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  dash: "M7 12h10M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  slash: "M5.6 5.6l12.8 12.8M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  play: "M7 5v14l11-7L7 5Z",
  stop: "M7 7h10v10H7z",
  refresh: "M20 11a8 8 0 0 0-14.3-4.9L4 8m0-4v4h4m-4 5a8 8 0 0 0 14.3 4.9L20 16m0 4v-4h-4",
  folder: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z",
  file: "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Zm0 0v5h5",
  cube: "M12 3 4 7.5v9L12 21l8-4.5v-9L12 3Zm0 0v18m8-13.5-8 4.5-8-4.5",
  chevron: "M9 6l6 6-6 6",
  back: "M15 6l-6 6 6 6",
  external: "M14 4h6v6m0-6-9 9M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4",
  download: "M12 4v11m0 0-4-4m4 4 4-4M5 20h14",
  trash: "M4 7h16M10 11v6m4-6v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3",
  edit: "M4 20h4L19 9l-4-4L4 16v4Zm9-13 4 4",
  crop: "M6 2v14a2 2 0 0 0 2 2h14M2 6h14a2 2 0 0 1 2 2v14",
  target: "M12 3v3m0 12v3M3 12h3m12 0h3m-3 0a6 6 0 1 1-12 0 6 6 0 0 1 12 0Z",
  layers: "m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5m-18 4 9 5 9-5",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14Zm5-2 4 4",
  grid: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
  list: "M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01",
  plus: "M12 5v14M5 12h14",
  upload: "M12 20V9m0 0-4 4m4-4 4 4M5 4h14",
  drive: "M3 14h18v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4Zm0 0 3-8a2 2 0 0 1 1.9-1.3h8.2A2 2 0 0 1 18 6l3 8M7 17h.01",
  flag: "M5 21V4m0 0h12.5l-2.8 4.5 2.8 4.5H5",
  move: "M12 3v18M3 12h18M12 3 9.5 5.5M12 3l2.5 2.5M12 21l-2.5-2.5M12 21l2.5-2.5M3 12l2.5-2.5M3 12l2.5 2.5M21 12l-2.5-2.5M21 12l-2.5 2.5",
  palette: "M12 3a9 9 0 0 0 0 18c1.2 0 1.8-.8 1.8-1.7 0-1-.9-1.4-.9-2.4 0-.9.8-1.6 1.7-1.6H17a4 4 0 0 0 4-4c0-4.6-4-8.3-9-8.3ZM7.5 12h.01M9 7.5h.01M14.5 7h.01M17.5 10.5h.01",
};

export function icon(name, cls) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.8");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  if (cls) svg.setAttribute("class", cls);
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", ICONS[name] || ICONS.circle);
  svg.appendChild(path);
  return svg;
}

export function spinner(fraction) {
  // A ring that fills with progress; spins when progress is unknown.
  const svg = h("svg:svg", { viewBox: "0 0 20 20", "aria-hidden": "true" });
  const r = 7.5, c = 2 * Math.PI * r;
  svg.appendChild(h("svg:circle", { class: "bg", cx: 10, cy: 10, r, fill: "none", "stroke-width": 2.5 }));
  const fg = h("svg:circle", {
    class: "fg", cx: 10, cy: 10, r, fill: "none", "stroke-width": 2.5, "stroke-linecap": "round",
    transform: "rotate(-90 10 10)",
    "stroke-dasharray": `${c}`, "stroke-dashoffset": `${c * (1 - Math.max(0.08, Math.min(1, fraction ?? 0.25)))}`,
  });
  if (fraction == null) {
    fg.appendChild(h("svg:animateTransform", {
      attributeName: "transform", type: "rotate", from: "0 10 10", to: "360 10 10", dur: "1s", repeatCount: "indefinite",
    }));
  }
  svg.appendChild(fg);
  return svg;
}

// ---------------------------------------------------------------- status

const STATUS = {
  done: ["good", "Done", "checkCircle"],
  stale: ["warning", "Out of date", "alert"],
  failed: ["critical", "Failed", "xCircle"],
  running: ["active", "Running", "clock"],
  queued: ["active", "Queued", "clock"],
  waiting: ["active", "Waiting", "clock"],
  cancelled: ["neutral", "Cancelled", "slash"],
  ready: ["neutral", "Not run", "circle"],
  blocked: ["neutral", "Not ready", "dash"],
  optional: ["neutral", "Optional", "dash"],
  skipped: ["neutral", "Not needed", "dash"],
  disabled: ["neutral", "Turned off", "slash"],
  OK: ["good", "Checks pass", "checkCircle"],
  WARN: ["warning", "Warnings", "alert"],
  FAIL: ["critical", "Checks failed", "xCircle"],
};

export function statusInfo(status) {
  const [tone, label, ic] = STATUS[status] || ["neutral", status || "Unknown", "circle"];
  return { tone, label, icon: ic };
}

export function statusBadge(status, labelOverride, progress) {
  const info = statusInfo(status);
  const mark = status === "running" ? spinner(progress) : icon(info.icon);
  return h("span", { class: `status ${info.tone}` }, mark, labelOverride || info.label);
}

// ---------------------------------------------------------------- formatting

export function fmtBytes(n) {
  if (n == null) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = n;
  while (v >= 1000 && i < units.length - 1) { v /= 1000; i++; }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export function sig(v, digits = 3) {
  if (v == null || !Number.isFinite(v)) return "–";
  if (v === 0) return "0";
  const mag = Math.floor(Math.log10(Math.abs(v)));
  const decimals = Math.max(0, digits - 1 - mag);
  return v.toLocaleString(undefined, { minimumFractionDigits: Math.min(decimals, 6), maximumFractionDigits: Math.min(decimals, 6) });
}

// 5 -> "5th", 1 -> "1st", 2.5 -> "2.5th": how a percentile is named.
export function ordinal(p) {
  if (!Number.isInteger(p)) return `${p}th`;
  const tens = p % 100;
  const suffix = tens >= 10 && tens <= 20 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[p % 10] || "th");
  return `${p}${suffix}`;
}

export function fmtNum(v, decimals = 0) {
  if (v == null || !Number.isFinite(v)) return "–";
  return v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

export function fmtPct(v, decimals = 1) {
  if (v == null || !Number.isFinite(v)) return "–";
  return `${(v * 100).toFixed(decimals)}%`;
}

// Percent with as many decimals as a tick step needs to tell ticks apart:
// ticks 0.2% apart get one decimal, 1% apart none. Without a step, one decimal.
export function pctFmt(v, step) {
  if (step == null) return fmtPct(v, 1);
  const decimals = Math.max(0, Math.min(3, Math.ceil(-Math.log10(Math.abs(step) * 100) - 1e-9)));
  return fmtPct(v, decimals);
}

// A stored scan value in mg HA/cm³, as tools/scanco_calibration.to_density
// converts it with a calibration the server found for the scan.
export function toMgHA(value, cal) {
  return (value + (cal.offset || 0)) / cal.divisor * cal.slope + cal.intercept;
}

// "mg HA/cm³ = value / 4096 × 375.5 − 197.3": a calibration, readably.
export function calibrationFormula(cal) {
  return `mg HA/cm³ = ${cal.offset ? `(value + ${sig(cal.offset, 4)})` : "value"} / ${sig(cal.divisor, 4)} × ${sig(cal.slope, 4)} `
    + `${cal.intercept < 0 ? "−" : "+"} ${sig(Math.abs(cal.intercept), 4)}`;
}

// A measurement with as many decimals as a tick step needs; without a step,
// whole numbers from 100 up and four significant digits below.
export function numFmt(v, step) {
  if (v == null || !Number.isFinite(v)) return "–";
  if (step != null) return fmtNum(v, Math.max(0, Math.min(4, Math.ceil(-Math.log10(Math.abs(step)) - 1e-9))));
  return Math.abs(v) >= 100 ? fmtNum(v, 0) : sig(v, 4);
}

export function fmtDuration(s) {
  if (s == null || !Number.isFinite(s)) return "";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60), rest = Math.round(s % 60);
  if (m < 60) return `${m} min${rest ? ` ${rest} s` : ""}`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function relTime(iso) {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  const s = (Date.now() - t) / 1000;
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  if (s < 86400 * 14) return `${Math.round(s / 86400)} d ago`;
  return new Date(iso).toLocaleDateString();
}

export function elapsedSince(iso) {
  if (!iso) return null;
  return (Date.now() - new Date(iso).getTime()) / 1000;
}

export function basename(path) {
  if (!path) return "";
  const parts = String(path).split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || path;
}

export function slotColor(slot) {
  if (slot == null) return "var(--ink-3)";
  return `var(--s${(slot % 8) + 1})`;
}

// Resolved CSS colour, for canvas drawing.
export function cssColor(value, el = document.documentElement) {
  const m = /^var\((--[\w-]+)\)$/.exec(value);
  return m ? getComputedStyle(el).getPropertyValue(m[1]).trim() : value;
}

export function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

export function rafThrottle(fn) {
  let queued = false;
  return () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; fn(); });
  };
}

export function download(url) {
  const a = h("a", { href: url, download: "" });
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ---------------------------------------------------------------- toasts

export function toast(message, tone = "good", { action, timeout = 5000 } = {}) {
  const root = document.getElementById("toasts");
  const ic = { good: "checkCircle", critical: "xCircle", warning: "alert", info: "info" }[tone] || "info";
  const el = h("div", { class: `toast ${tone}` }, icon(ic), h("div", { class: "grow" }, message));
  if (action) el.appendChild(h("button", { class: "btn small ghost", onclick: () => { action.onClick(); el.remove(); } }, action.label));
  el.appendChild(h("button", { class: "btn small ghost icon", "aria-label": "Dismiss", onclick: () => el.remove() }, icon("x")));
  root.appendChild(el);
  if (timeout) setTimeout(() => el.remove(), timeout);
  return el;
}

export function errorToast(err) {
  return toast(err?.message || String(err), "critical", { timeout: 9000 });
}

// ---------------------------------------------------------------- dialogs

export function modal({ title, body, actions = [], wide = false, onClose }) {
  const root = document.getElementById("modal-root");
  const previous = document.activeElement;
  const close = () => {
    back.remove();
    document.removeEventListener("keydown", onKey);
    if (onClose) onClose();
    if (previous && previous.focus) previous.focus();
  };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  const footer = h("footer");
  for (const a of actions) {
    footer.appendChild(h("button", {
      class: `btn ${a.primary ? "primary" : ""} ${a.danger ? "danger" : ""}`,
      onclick: async () => { if (a.onClick) { const keep = await a.onClick(close); if (keep === false) return; } close(); },
    }, a.label));
  }
  const dialog = h("div", { class: `modal ${wide ? "wide" : ""}`, role: "dialog", "aria-modal": "true", "aria-label": title },
    h("header", null, h("h2", { class: "grow" }, title),
      h("button", { class: "btn ghost icon", "aria-label": "Close", onclick: close }, icon("x"))),
    h("div", { class: "body" }, body),
    actions.length ? footer : null);
  const back = h("div", { class: "modal-back", onmousedown: (e) => { if (e.target === back) close(); } }, dialog);
  root.appendChild(back);
  document.addEventListener("keydown", onKey);
  const focusable = dialog.querySelector("input, select, textarea, button.primary") || dialog.querySelector("button");
  if (focusable) setTimeout(() => focusable.focus(), 0);
  return { close, dialog };
}

export function confirmDialog(title, text, confirmLabel = "Confirm", { danger = false } = {}) {
  return new Promise((resolve) => {
    let answered = false;
    modal({
      title,
      body: h("p", null, text),
      onClose: () => { if (!answered) resolve(false); },
      actions: [
        { label: "Cancel", onClick: () => { answered = true; resolve(false); } },
        { label: confirmLabel, primary: !danger, danger, onClick: () => { answered = true; resolve(true); } },
      ],
    });
  });
}

// ---------------------------------------------------------------- tooltip

let tipEl = null;
export function showTip(x, y, content) {
  if (!tipEl) {
    tipEl = h("div", { class: "tooltip", role: "tooltip" });
    document.body.appendChild(tipEl);
  }
  replace(tipEl, content);
  tipEl.style.display = "block";
  const pad = 14, r = tipEl.getBoundingClientRect();
  let left = x + pad, top = y + pad;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - pad;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - pad;
  tipEl.style.left = `${Math.max(8, left)}px`;
  tipEl.style.top = `${Math.max(8, top)}px`;
}

export function hideTip() {
  if (tipEl) tipEl.style.display = "none";
}

export function field(label, control, help) {
  return h("label", { class: "field" }, h("span", { class: "lbl" }, label), control, help ? h("span", { class: "help" }, help) : null);
}
