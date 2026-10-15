"""
The studio's HTTP server: the JSON API, uploads, the event stream, and the UI.

Standard library only (``http.server``), so the studio runs in the existing
``microct-analysis`` environment with nothing more to install.

It listens on localhost and is meant for the person at the machine (or, with
``--tailscale``, for their own devices on their tailnet: see remote.py). Two checks
keep other websites out of it anyway, since a page in any browser tab can
send requests to localhost: the Host header must name this server, which
defeats DNS rebinding, and a request that changes anything must come from this
server's own origin, which defeats cross-site form posts.
"""

from __future__ import annotations

import gzip
import json
import math
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from . import __version__, drives, results, steps
from .jobs import ACTIVE, JobManager
from .paths import PYTHON, REPO, SCRIPTS, WEB
from .scaninfo import describe, describe_all
from .volumes import read_gz, suggest_crop
from .workspace import (
    ARTIFACT_LABELS,
    META_FIELDS,
    FolderExists,
    ProjectExists,
    Workspace,
    artifacts_in,
    detect_kind,
    directory_size,
    find_workspaces,
    folder_path,
    group_of,
    is_workspace,
    now_iso,
    parse_name,
    read_json,
    safe_name,
    scan_paths,
    source_format,
    storage_root,
    synced_service,
    valid_name,
    workspace_summary,
    write_json,
)

PREFS = Path.home() / ".microct-studio.json"
VIEW_FILE = re.compile(r"^(base\.u16|unwrap\.u16|ov_[a-z0-9_]+\.u8)\.gz$|^thumb\.png$")
UPLOAD_ID = re.compile(r"^[a-f0-9]{8,40}$")
# Unread request body a reply reads and discards to keep its connection
# usable; a reply that would leave more than this closes the connection.
DRAIN_LIMIT = 1 << 20


def _folder_or_400(value: Any) -> str:
    """The library folder a request names, or a 400 saying what is wrong with it."""
    try:
        return folder_path(value)
    except ValueError as exc:
        raise HttpError(400, str(exc)) from exc


def clean(obj: Any) -> Any:
    """JSON-safe copy: NaN and infinity become null, which the browser can parse."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


class HttpError(Exception):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra  # more for the UI to act on, e.g. {"code": "not_workspace"}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventBus:
    """Fan-out of server events to every open event stream."""

    def __init__(self):
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, kind: str, data: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait((kind, data))
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------

def _load_prefs(path: Path = PREFS) -> dict:
    return read_json(path, {}) or {}


_FOLDER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]{0,79}$")


def _keys(path: str | None, ws_id: str | None) -> set[str]:
    """What identifies a workspace in the prefs: its id, and its folder for one without."""
    return {k for k in (ws_id, os.path.realpath(path) if path else None) if k}


def _can_list(folder: Path) -> bool:
    """Whether a drive can be browsed at all (a Time Machine backup disk cannot)."""
    try:
        with os.scandir(folder) as entries:
            next(entries, None)
        return True
    except OSError:
        return False


class Studio:
    """The server's state: the open workspace (if any), its jobs, and the event bus.

    The studio can start with no workspace open, so the person at the
    computer chooses one first: on a Mac and a PC sharing an external drive
    the same workspace sits at a different path on each, and the internal
    drive is not a good default for scans many gigabytes big.
    """

    def __init__(self, root: Path | None, remember: bool = True, prefs: Path = PREFS):
        self.bus = EventBus()
        # Whether to record the workspaces opened here, per computer, in ``prefs``.
        self.remember = remember
        self.prefs = prefs
        self.ws: Workspace | None = None
        self.jobs: JobManager | None = None
        if root is not None:
            self._attach(Workspace(root))

    # -- workspace --------------------------------------------------------

    def _attach(self, ws: Workspace) -> None:
        self.ws = ws
        self.jobs = JobManager(
            ws,
            prepare=lambda job: steps.prepare(ws, job),
            finish=self._finish,
            publish=self.bus.publish,
            failed=self._failed,
        )
        if self.remember:
            self._remember(ws)

    def require(self) -> None:
        """Refuse a request that needs a workspace when none is open or its drive is gone."""
        if self.ws is None:
            raise HttpError(409, "No workspace is open. Choose one first.", code="no_workspace")
        if not self.ws.root.is_dir():
            raise HttpError(409, f"The drive that holds {self.ws.root} is not connected.",
                            code="workspace_missing")

    def switch(self, root: Path) -> None:
        if self.jobs is not None and self.ws is not None and self.ws.root.is_dir() and \
                any(j.status in ACTIVE for j in self.jobs.jobs.values()):
            raise HttpError(409, "Jobs are still queued or running in this workspace. Cancel them first.")
        try:
            ws = Workspace(root)
        except OSError as exc:
            raise HttpError(400, f"Cannot open {root} as a workspace: {exc.strerror or exc}") from exc
        if self.jobs is not None:
            self.jobs.shutdown()
        self._attach(ws)
        self.bus.publish("workspace", {"root": str(ws.root)})

    # -- workspaces this computer knows --------------------------------------

    def _remember(self, ws: Workspace) -> None:
        """Record a workspace as recently opened on this computer, with its drive's label.

        The label is what finds it again when the drive comes back under a
        new letter (Windows) or a numbered name (macOS).
        """
        prefs = _load_prefs(self.prefs)
        vol = drives.volume_of(ws.root)
        entry = {"path": str(ws.root), "id": ws.id, "drive": vol[0] if vol else None,
                 "on_drive": ws.root.relative_to(vol[1]).as_posix() if vol else None, "opened": now_iso()}
        recent = [entry] + [e for e in self._recent_entries(prefs)
                            if e.get("path") != entry["path"] and (not entry["id"] or e.get("id") != entry["id"])]
        # Opening a workspace that was removed from the list puts it back.
        keys = _keys(entry["path"], entry["id"])
        prefs.update({"last": str(ws.root), "recent": recent[:12],
                      "hidden": [k for k in prefs.get("hidden", []) if k not in keys]})
        try:
            write_json(self.prefs, prefs)
        except OSError:
            pass

    def _recent_entries(self, prefs: dict | None = None) -> list[dict]:
        prefs = _load_prefs(self.prefs) if prefs is None else prefs
        return [e if isinstance(e, dict) else {"path": str(e)} for e in prefs.get("recent", [])]

    def forget(self, path: str, ws_id: str | None) -> dict:
        """Take a workspace off this computer's lists; its folder is not touched.

        It is hidden from "Found on connected drives" too, or it would come
        straight back there. Returns what ``restore`` needs to undo it.
        """
        keys = _keys(path, ws_id)
        if self.ws is not None and keys & _keys(str(self.ws.root), self.ws.id):
            raise HttpError(409, "That workspace is open now. Open another one before removing it from the list.")
        prefs = _load_prefs(self.prefs)
        recent = self._recent_entries(prefs)
        at = [i for i, e in enumerate(recent) if _keys(e.get("path"), e.get("id")) & keys]
        removed = [recent[i] for i in at]
        hidden = list(prefs.get("hidden", []))
        added = sorted(keys - set(hidden))
        prefs.update({"recent": [e for e in recent if e not in removed], "hidden": (hidden + added)[-100:]})
        paths = keys | {k for e in removed for k in _keys(e.get("path"), None)}
        last = prefs.get("last")
        if last and _keys(last, None) & paths:
            del prefs["last"]  # --last would otherwise reopen it
        else:
            last = None
        write_json(self.prefs, prefs)
        return {"removed": removed, "at": at, "hidden": added, "last": last}

    def restore(self, removed: list[dict], at: list[int], hidden: list[str], last: str | None = None) -> None:
        """Undo ``forget``, putting each entry back where it was in the list."""
        prefs = _load_prefs(self.prefs)
        recent = self._recent_entries(prefs)
        for i, e in sorted(zip(at, removed), key=lambda pair: pair[0]):
            if isinstance(e, dict) and isinstance(e.get("path"), str) and \
                    not any(e["path"] == r.get("path") for r in recent):
                recent.insert(i, e)
        prefs.update({"recent": recent[:12], "hidden": [k for k in prefs.get("hidden", []) if k not in hidden]})
        if last and not prefs.get("last"):
            prefs["last"] = last
        write_json(self.prefs, prefs)

    @staticmethod
    def _locate(entry: dict) -> Path | None:
        """Where a remembered workspace is now, following its drive if it moved."""
        path = Path(entry.get("path") or "")
        if entry.get("path") and is_workspace(path):
            return path
        if entry.get("drive"):
            root = drives.find_volume(entry["drive"])
            if root is not None:
                candidate = Path(root, *[p for p in (entry.get("on_drive") or "").split("/") if p])
                if is_workspace(candidate):
                    return candidate
        return None

    def workspaces(self) -> dict:
        """Recent workspaces, those found on connected drives, and places to make a new one."""
        drives.forget()  # a drive plugged in since the last look
        current = self.ws.root if self.ws is not None else None
        seen: set[str] = set()

        def item(path: Path, extra: dict) -> dict | None:
            info = {**workspace_summary(path), **extra}
            key = info.get("id") or os.path.realpath(path)
            if key in seen:
                return None
            seen.add(key)
            info["current"] = current is not None and os.path.realpath(path) == os.path.realpath(current)
            return info

        recent = []
        for entry in self._recent_entries():
            path = self._locate(entry)
            if path is not None:
                row = item(path, {"available": True, "opened": entry.get("opened")})
            else:
                key = entry.get("id") or entry.get("path")
                row = None if key in seen else {
                    "path": entry.get("path"), "name": Path(entry.get("path") or "").name, "id": entry.get("id"),
                    "drive": entry.get("drive"), "samples": None, "available": False,
                    "opened": entry.get("opened"), "current": False, "sharing": None}
                seen.add(key)
            if row:
                recent.append(row)
        hidden = set(_load_prefs(self.prefs).get("hidden", []))
        found = [row for row in (item(p, {"available": True}) for p in find_workspaces()) if row
                 and not _keys(row["path"], row.get("id")) & hidden]
        places = [{"label": label, "path": str(root), "sharing": drives.sharing_note(root)}
                  for label, root in drives.mounted_volumes().items()
                  if os.path.realpath(root) != os.sep and _can_list(root)]
        # A drive both systems can write comes first, so it is the one offered.
        places.sort(key=lambda p: p["sharing"]["tone"] != "good")
        home = Path.home()
        places.append({"label": "Home folder", "path": str(home), "sharing": drives.sharing_note(home)})
        return {"current": str(current) if current else None, "recent": recent, "found": found, "places": places}

    def _finish(self, job) -> None:
        steps.finish(self.ws, job, self.submit_previews)
        if job.sample:
            self.publish_sample(job.sample)

    def _failed(self, job) -> None:
        steps.record_failure(self.ws, job)
        if job.sample:
            self.publish_sample(job.sample)

    # -- summaries --------------------------------------------------------

    def summary(self, record: dict) -> dict:
        ws = self.ws
        active = self.jobs.active_for(record["name"])
        src = record.get("source") or {}
        scan = ws.source_path(record)
        group = group_of(record.get("meta") or {})
        mq = results.molar(ws, record)
        prof = results.incisor_profile(ws, record)
        thumb_view = next((v for v in ("mandible", "incisor", "converted", "source")
                           if (steps.view_dir(ws, record["name"], v) / "thumb.png").exists()), None)
        return {
            "name": record["name"],
            "folder": record.get("folder") or "",
            "meta": record.get("meta") or {},
            "group": group,
            "slot": ws.group_slot(group),
            "created": record.get("created"),
            "updated": record.get("updated"),
            "source": {"kind": src.get("kind"), "mode": src.get("mode"),
                       "path": str(scan) if scan is not None else src.get("path"),
                       # False while the drive holding a linked scan is unplugged.
                       "connected": scan is not None and scan.exists(),
                       "format": steps.raw_format(record),
                       "needs_conversion": steps.needs_conversion(record)},
            "crop_box": record.get("crop_box"),
            "pipeline": steps.pipeline_status(ws, record, active),
            "validation": (record.get("validation") or {}).get("status"),
            "excluded": results.exclusion(record),
            "metrics": {
                "enamel_fraction": mq["enamel_fraction"] if mq else None,
                "molar_mm3": mq["total_mm3"] if mq else None,
                **results.incisor_metrics(prof),
            },
            "thumb": f"/api/samples/{record['name']}/views/{thumb_view}/file/thumb.png" if thumb_view else None,
            "busy": bool(active),
        }

    def publish_sample(self, name: str) -> None:
        record = self.ws.get(name)
        if record is not None:
            self.bus.publish("sample", clean(self.summary(record)))

    def publish_folders(self) -> None:
        self.bus.publish("folders", {"folders": self.ws.folders()})

    def detail(self, name: str) -> dict:
        record = self._record(name)
        ws = self.ws
        storage = ws.storage_dir(record)
        stored_storage = (record.get("storage") or "").rstrip("/")
        artifacts = []
        for key, value in (record.get("artifacts") or {}).items():
            path = ws.artifact(record, key)
            resolved = ws.resolve_path(record, value)
            # workspace: in the workspace; storage: a large file the studio wrote
            # beside the scan; linked: someone else's file, used where it is.
            where = "workspace" if ws.is_local(value) else \
                "storage" if stored_storage and value.startswith(stored_storage + "/") else "linked"
            artifacts.append({
                "key": key, "label": ARTIFACT_LABELS.get(key, key),
                "path": str(path or resolved or value), "exists": path is not None,
                "size": results.file_size(path) if path and path.is_file() else None,
                "linked": where == "linked", "where": where,
            })
        jobs = [j.public() for j in sorted(self.jobs.jobs.values(), key=lambda j: j.created, reverse=True)
                if j.sample == name][:40]
        return {
            "record": record,
            "summary": self.summary(record),
            "views": steps.views_for(ws, record),
            "artifacts": artifacts,
            "jobs": jobs,
            "folder": str(ws.sample_dir(name)),
            "storage": {"path": str(storage), "exists": storage.is_dir(), "connected": True,
                        "synced": synced_service(storage)} if storage is not None else
            {"path": record["storage"], "exists": False, "connected": False, "synced": None}
            if record.get("storage") else None,
            "default_run": steps.default_run(ws, record),
        }

    def _record(self, name: str) -> dict:
        record = self.ws.get(name)
        if record is None:
            raise HttpError(404, f"No sample named {name}")
        return record

    # -- jobs -------------------------------------------------------------

    def submit_previews(self, name: str, views: list[str], after: list[str] | None = None,
                        force: bool = False) -> list[str]:
        record = self.ws.get(name)
        if record is None:
            return []
        made = []
        available = {v["view"]: v for v in steps.views_for(self.ws, record)}
        for view in views:
            info = available.get(view)
            if info is None or (info["current"] and not force):
                continue
            if any(j.step == "preview" and j.sample == name and j.params.get("view") == view
                   and j.status in ACTIVE for j in self.jobs.jobs.values()):
                continue
            job = self.jobs.submit(step="preview", title=f"Preview: {steps.VIEWS[view]['label']}",
                                   sample=name, params={"view": view}, lane="preview", deps=after)
            made.append(job.id)
        return made

    def run(self, names: list[str], requested: list[str] | str, params: dict) -> dict:
        batch = uuid.uuid4().hex[:8]
        created, skipped = [], []
        for name in names:
            record = self.ws.get(name)
            if record is None:
                skipped.append({"sample": name, "reason": "No such sample"})
                continue
            wanted = steps.default_run(self.ws, record) if requested == "default" else list(requested)
            if not wanted:
                skipped.append({"sample": name, "reason": "Nothing left to run"})
                continue
            busy = {j.step for j in self.jobs.active_for(name)}
            clash = [s for s in wanted if s in busy]
            if clash:
                skipped.append({"sample": name, "reason": f"{steps.STEPS[clash[0]].label} is already queued"})
                continue
            try:
                plan = steps.plan(self.ws, record, wanted)
            except ValueError as exc:
                skipped.append({"sample": name, "reason": str(exc)})
                continue
            ids: dict[str, str] = {}
            for sid, deps in plan:
                step = steps.STEPS[sid]
                job = self.jobs.submit(step=sid, title=step.label, sample=name,
                                       params=dict(params.get(sid) or {}), lane=step.lane,
                                       deps=[ids[d] for d in deps], batch=batch)
                ids[sid] = job.id
                created.append(job.public())
            self.publish_sample(name)
        return {"jobs": created, "skipped": skipped, "batch": batch}

    # -- importing --------------------------------------------------------

    def create_sample(self, body: dict) -> dict:
        meta = dict(body.get("meta") or {})
        source = body.get("source") or {}
        name = (body.get("name") or "").strip() or None
        if name and not valid_name(name):
            raise HttpError(400, f"{name!r} is not a usable name: letters, digits, . _ + - only")
        shelf = _folder_or_400(body.get("folder"))  # the library folder; `folder` below is the upload's

        artifacts: dict[str, str] = {}
        src_record: dict = {}
        copy_from = None
        found = ""  # the name the scan came with, kept when its details are too few to name it
        if source.get("upload"):
            uid = source["upload"]
            if not UPLOAD_ID.match(uid):
                raise HttpError(400, "Bad upload id")
            folder = self.ws.uploads_dir / uid
            entries = [p for p in folder.iterdir() if not p.name.startswith(".")] if folder.exists() else []
            if not entries:
                raise HttpError(400, "The upload is empty")
            target = entries[0] if len(entries) == 1 else folder
            kind = detect_kind(target)
            src_record = {"kind": kind, "format": source_format(kind, target), "mode": "upload",
                          "path": str(target), "upload": uid}
            # Several loose files have no name of their own; the upload's folder is not one.
            found = "" if target == folder else (target.stem if target.is_file() else target.name)
        elif source.get("path"):
            path = Path(source["path"]).expanduser()
            if not path.exists():
                raise HttpError(400, f"{path} does not exist")
            kind = detect_kind(path)
            found = path.stem if path.is_file() else path.name
            if kind == "collection":
                raise HttpError(400, f"{path} holds several samples; use Scan a folder to import them together")
            if kind == "segmentation":
                artifacts = artifacts_in(path, path.name)
                src_record = {"kind": "results", "mode": "link", "path": str(path)}
            else:
                mode = source.get("mode") or "link"
                # The format is recorded now: a copy blanks the path until it lands.
                src_record = {"kind": kind, "format": source_format(kind, path), "mode": mode, "path": str(path)}
                if mode == "copy":
                    copy_from = path
        if not name:
            planned = self.ws.plan_names([{"meta": meta, "found": found}])[0]
            name = planned["name"]
            if planned["from"] == "details":
                meta["replicate"] = planned["replicate"]
        record = self.ws.create(meta, name=name, source=src_record if not copy_from else
                                {**src_record, "path": ""}, artifacts=artifacts, folder=shelf)
        name = record["name"]
        if shelf:
            self.publish_folders()

        if src_record.get("mode") == "upload":
            # Same filesystem, so this is a rename, not a copy.
            raw_dir = self.ws.sample_dir(name) / "raw"
            upload_dir = self.ws.uploads_dir / src_record["upload"]
            moved = raw_dir / Path(src_record["path"]).name if Path(src_record["path"]) != upload_dir else raw_dir
            if moved == raw_dir:
                os.replace(upload_dir, raw_dir)
            else:
                raw_dir.mkdir(parents=True, exist_ok=True)
                os.replace(src_record["path"], moved)
                shutil.rmtree(upload_dir, ignore_errors=True)

            def point(rec):
                rec["source"] = {"kind": src_record["kind"], "format": src_record["format"],
                                 "mode": "upload", "path": self.ws.store_path(rec, moved)}
                self.ws._set_artifact(rec, "raw", moved)  # noqa: SLF001
            record = self.ws.update(name, point)

        if copy_from is not None:
            copy = self.jobs.submit(step="import_copy", title="Copy raw data", sample=name,
                                    params={"src": str(copy_from)}, lane="light")
            after = [copy.id]
        else:
            after = None
            if self.ws.settings().get("auto_preview", True):
                self.submit_previews(name, ["source", "mandible", "incisor", "molar", "cmpr"])

        run = body.get("run")
        queued = None
        if run:
            if after:
                # The copy must land first; the rest of the pipeline waits on it.
                queued = self._run_after(name, run, body.get("params") or {}, after)
            else:
                queued = self.run([name], run, body.get("params") or {})
        self.publish_sample(name)
        return {"sample": self.summary(self.ws.get(name)), "run": queued}

    def _run_after(self, name: str, run, params: dict, after: list[str]) -> dict:
        wanted = ["convert", "segment", "cmpr", "incisor_quant", "molar_quant"] if run == "default" else list(run)
        record = self.ws.get(name)
        wanted = [s for s in wanted if not steps.STEPS[s].disabled
                  and (s != "convert" or steps.needs_conversion(record))]
        created, prev = [], list(after)
        ids: dict[str, str] = {}
        for sid in [s for s in steps.PIPELINE_ORDER if s in wanted]:
            deps = [ids[u] for u in ids if u in steps._upstream_closure(sid)] or prev  # noqa: SLF001
            step = steps.STEPS[sid]
            job = self.jobs.submit(step=sid, title=step.label, sample=name, params=params.get(sid) or {},
                                   lane=step.lane, deps=deps)
            ids[sid] = job.id
            created.append(job.public())
        return {"jobs": created, "skipped": []}

    def import_many(self, body: dict) -> dict:
        made, failed = [], []
        items = body.get("samples") or []
        folder = _folder_or_400(body.get("folder"))
        # Named exactly as the import table previewed them. An item without
        # the name it was found under comes from before names were planned,
        # and its name is taken as typed.
        plan = self.ws.plan_names([{"key": i, "meta": item.get("meta"), "found": item.get("found"),
                                    "name": item.get("name")} for i, item in enumerate(items)])
        for item, planned in zip(items, plan):
            try:
                if planned["error"]:
                    raise ValueError(planned["error"])
                artifacts = item.get("artifacts") or {}
                source = item.get("source") or {}
                meta = dict(item.get("meta") or {})
                if planned["from"] == "details":
                    meta["replicate"] = planned["replicate"]
                src = {"kind": source.get("kind"), "mode": "link", "path": source.get("path")}
                record = self.ws.create(meta, name=planned["name"], source=src, artifacts=artifacts, folder=folder)
                validation = steps.validation_summary(self.ws.artifact(record, "validation"), record["name"])
                if validation:
                    self.ws.update(record["name"], lambda r, v=validation: r.update(validation=v))
                made.append(record["name"])
            except (ValueError, OSError) as exc:
                failed.append({"sample": planned["name"] or item.get("found"), "reason": str(exc)})
        if self.ws.settings().get("auto_preview", True):
            for name in made:
                self.submit_previews(name, ["incisor", "molar", "cmpr", "mandible", "source"])
        run = None
        if body.get("run") and made:
            run = self.run(made, "default", body.get("params") or {})
        for name in made:
            self.publish_sample(name)
        if folder and made:
            self.publish_folders()
        return {"created": made, "failed": failed, "run": run}

    def imported_paths(self) -> dict[str, str]:
        """Every scan and linked output in the library, by path, with the specimen that has it.

        A results folder is not counted: loose outputs of many specimens (a
        CMPR folder, say) share one, so it does not tell them apart.
        """
        out: dict[str, str] = {}
        for record in self.ws.all():
            src = record.get("source") or {}
            values = list((record.get("artifacts") or {}).values())
            if src.get("kind") != "results":
                values.append(src.get("path"))
            for value in values:
                path = self.ws.resolve_path(record, value)
                if path is not None:
                    out[os.path.normcase(os.path.abspath(path))] = record["name"]
        return out

    # -- landmarks --------------------------------------------------------

    def save_marks(self, name: str, marks: list[dict], run: bool) -> dict:
        """Save the viewer's marks; with ``run``, map them as they are, without finding the landmarks again."""
        record = self._record(name)
        rows = []
        for mark in marks:
            label = str(mark.get("landmark") or "").strip()
            if not label or mark.get("z_start") is None:
                continue
            rows.append({"landmark": label, "z_start": float(mark["z_start"]),
                         "z_end": None if mark.get("z_end") is None else float(mark["z_end"]),
                         "source": "auto" if mark.get("source") in steps.AUTO_SOURCES else "manual"})
        path = steps.marks_path(self.ws, record)
        steps.write_marks(path, rows)
        self.ws.set_artifacts(name, {"landmark_marks": path})
        queued = self.run([name], ["landmarks"], {"landmarks": {"map_only": True}}) if run and rows else None
        self.publish_sample(name)
        return {"saved": len(rows), "run": queued, "path": str(path), "record": record["name"]}

    def read_marks(self, name: str) -> dict:
        """The marks, what they map to, and where the Landmarks step found each: what a hand mark goes back to."""
        record = self._record(name)
        path = self.ws.artifact(record, "landmark_marks")
        marks = steps.read_marks(path) if path and path.is_file() else []
        report_path = self.ws.artifact(record, "landmark_report")
        report = (read_json(report_path) if report_path else None) or {}
        return {"marks": marks, "mapped": results.landmarks(self.ws, record),
                "auto": steps.detected_marks(report), "warnings": report.get("warnings") or []}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

ROUTES: list[tuple[str, re.Pattern, str]] = []


def route(method: str, pattern: str):
    def register(fn):
        ROUTES.append((method, re.compile("^" + pattern + "$"), fn.__name__))
        return fn
    return register


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"MicroCTStudio/{__version__}"
    studio: Studio  # set on the class by serve()
    allowed_hosts: set[str] = set()
    remote = False  # served to other devices: Finder and napari would open on a screen nobody sees

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt, *args):  # noqa: D401 -- quiet: the UI shows what matters
        if os.environ.get("STUDIO_HTTP_LOG"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "").lower() in self.allowed_hosts

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return urlsplit(origin).netloc.lower() in self.allowed_hosts

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        # Request body bytes not yet read. The browser keeps one connection
        # open for many requests, and whatever a handler leaves unread would be
        # parsed as the start of the next request (http.server answers "{}PUT"
        # with 501), so send_response settles the body before any reply.
        try:
            self._unread = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            self._unread, self.close_connection = 0, True
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
        try:
            if not self._host_ok():
                raise HttpError(403, "Unrecognized Host header")
            if method != "GET" and not self._origin_ok():
                raise HttpError(403, "Cross-origin request refused")
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            self.query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
            if path.startswith("/api/"):
                for m, pattern, name in ROUTES:
                    if m != method:
                        continue
                    match = pattern.match(path)
                    if match:
                        if name not in WITHOUT_WORKSPACE:
                            self.st.require()
                        getattr(self, name)(*match.groups())
                        return
                raise HttpError(404, f"No endpoint {method} {path}")
            if method != "GET":
                raise HttpError(405, "Method not allowed")
            self._static(path)
        except HttpError as exc:
            self._error(exc.status, exc.message, exc.extra)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 -- report, keep serving
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")

    def send_response(self, code, message=None):
        self._settle_body()
        super().send_response(code, message)
        if getattr(self, "_unread", 0):
            self.send_header("Connection", "close")

    def _settle_body(self) -> None:
        """Read and drop what is left of the request body, if that is little."""
        left = getattr(self, "_unread", 0)
        if not 0 < left <= DRAIN_LIMIT:
            return
        while left > 0:
            block = self.rfile.read(min(left, 1 << 16))
            if not block:
                break
            left -= len(block)
        self._unread = left

    def _read(self, size: int) -> bytes:
        data = self.rfile.read(size) if size > 0 else b""
        self._unread -= len(data)
        return data

    def _error(self, status: int, message: str, extra: dict | None = None) -> None:
        try:
            self._json({"error": message, **(extra or {})}, status)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _body(self, limit: int = 20_000_000) -> dict:
        length = self._unread
        if length > limit:
            raise HttpError(413, "Request too large")
        raw = self._read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise HttpError(400, f"Body is not JSON: {exc}") from exc

    def _send(self, status: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(clean(payload), separators=(",", ":")).encode("utf-8")
        if len(body) > 4096 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
            self._send(status, gzip.compress(body, 5), "application/json",
                       {"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})
        else:
            self._send(status, body, "application/json")

    def _file(self, path: Path, ctype: str | None = None, *, encoding: str | None = None,
              download: str | None = None, cache: bool = False) -> None:
        try:
            st = path.stat()
        except OSError as exc:
            raise HttpError(404, f"{path.name} not found") from exc
        etag = f'"{st.st_size:x}-{int(st.st_mtime * 1000):x}"'
        if cache and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(st.st_size))
        self.send_header("Cache-Control", "no-cache" if cache else "no-store")
        self.send_header("ETag", etag)
        if encoding:
            self.send_header("Content-Encoding", encoding)
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as handle:
            shutil.copyfileobj(handle, self.wfile, 1 << 20)

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (WEB / rel).resolve()
        if WEB.resolve() not in target.parents and target != WEB.resolve():
            raise HttpError(404, "Not found")
        if not target.is_file():
            if Path(rel).suffix:
                raise HttpError(404, "Not found")
            # Client-side routes all serve the app shell.
            target = WEB / "index.html"
        ctype = {".js": "text/javascript", ".css": "text/css", ".html": "text/html",
                 ".svg": "image/svg+xml"}.get(target.suffix, None)
        self._file(target, (ctype + "; charset=utf-8") if ctype and ctype.startswith("text") else ctype,
                   cache=True)

    @property
    def st(self) -> Studio:
        return self.studio

    # -- state ------------------------------------------------------------

    @route("GET", r"/api/state")
    def api_state(self):
        ws = self.st.ws
        workspace = None
        if ws is not None:
            available = ws.root.is_dir()
            workspace = {"root": str(ws.root), "name": ws.root.name, "available": available,
                         "id": ws.id if available else None, "settings": ws.settings(),
                         "folders": ws.folders() if available else [],
                         "sharing": drives.sharing_note(ws.root) if available else None}
        self._json({
            "version": __version__,
            "workspace": workspace,
            "steps": [steps.STEPS[s].spec() for s in steps.PIPELINE_ORDER],
            "views": {k: v["label"] for k, v in steps.VIEWS.items()},
            "tools": [t.spec() for t in steps.TOOLS.values()],
            "artifact_labels": ARTIFACT_LABELS,
            "meta_fields": list(META_FIELDS),
            "python": PYTHON,
            "repo": str(REPO),
            "platform": sys.platform,
            "remote": self.remote,
        })

    @route("GET", r"/api/events")
    def api_events(self):
        q = self.st.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def chunk(text: str) -> None:
            data = text.encode("utf-8")
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            chunk("retry: 2000\nevent: hello\ndata: {}\n\n")
            while True:
                try:
                    kind, data = q.get(timeout=15)
                except queue.Empty:
                    chunk(": keep-alive\n\n")
                    continue
                chunk(f"event: {kind}\ndata: {json.dumps(clean(data), separators=(',', ':'))}\n\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.st.bus.unsubscribe(q)
            self.close_connection = True

    @route("GET", r"/api/settings")
    def api_settings(self):
        self._json(self.st.ws.settings())

    @route("PATCH", r"/api/settings")
    def api_settings_patch(self):
        body = self._body()
        allowed = {"lanes", "preview_budget_voxels", "auto_preview", "step_defaults", "group_colors", "results_style"}
        patch = {k: v for k, v in body.items() if k in allowed}
        # The Results page's colours and titles, replaced whole on each save.
        if "results_style" in patch and not isinstance(patch["results_style"], dict):
            raise HttpError(400, "results_style must be an object")
        settings = self.st.ws.update_settings(patch)
        self.st.bus.publish("settings", settings)
        self._json(settings)

    @route("GET", r"/api/workspaces")
    def api_workspaces(self):
        self._json(self.st.workspaces())

    @route("POST", r"/api/workspaces/remove")
    def api_workspace_remove(self):
        """Take a workspace off the lists: {path, id}. Its folder is left as it is."""
        body = self._body()
        path = str(body.get("path") or "").strip()
        if not path:
            raise HttpError(400, "Give the workspace to remove from the list")
        self._json(self.st.forget(path, body.get("id") or None))

    @route("POST", r"/api/workspaces/restore")
    def api_workspace_restore(self):
        """Undo a removal, with what /api/workspaces/remove returned."""
        body = self._body()
        removed, at, hidden = body.get("removed") or [], body.get("at") or [], body.get("hidden") or []
        if not all(isinstance(v, list) for v in (removed, at, hidden)) or \
                not all(isinstance(i, int) and i >= 0 for i in at) or len(at) != len(removed):
            raise HttpError(400, "removed, at and hidden must be what /api/workspaces/remove returned")
        self.st.restore(removed, at, [str(k) for k in hidden], str(body["last"]) if body.get("last") else None)
        self._json({"ok": True})

    @route("POST", r"/api/workspace")
    def api_workspace_open(self):
        """Open a workspace: {path} opens one, {path, create} makes that folder one,
        and {parent, name} makes a new folder inside parent."""
        body = self._body()
        if body.get("parent"):
            parent = Path(str(body["parent"])).expanduser()
            name = str(body.get("name") or "MicroCT-Studio").strip()
            if not _FOLDER_NAME.match(name) or name.endswith("."):
                raise HttpError(400, f"{name!r} is not a usable folder name: letters, digits, spaces and . _ + -")
            if not parent.is_dir():
                raise HttpError(404, f"{parent} is not a folder")
            root = parent / name
            if root.exists() and not is_workspace(root) and any(root.iterdir()):
                raise HttpError(409, f"{root} already exists and holds other files. Choose another name.")
        else:
            raw = str(body.get("path") or "").strip()
            if not raw:
                raise HttpError(400, "Give a folder for the workspace")
            root = Path(raw).expanduser()
            if root.exists() and not root.is_dir():
                raise HttpError(400, f"{root} is a file")
            if not body.get("create") and not is_workspace(root):
                empty = root.is_dir() and not any(p for p in root.iterdir() if not p.name.startswith("."))
                raise HttpError(404, f"{root} is not a MicroCT Studio workspace.", code="not_workspace",
                                empty=empty, exists=root.exists())
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HttpError(400, f"Cannot create {root}: {exc.strerror or exc}") from exc
        self.st.switch(root)
        self._json({"root": str(self.st.ws.root), "name": self.st.ws.root.name})

    # -- samples ----------------------------------------------------------

    @route("GET", r"/api/samples")
    def api_samples(self):
        self._json([self.st.summary(r) for r in self.st.ws.all()])

    @route("POST", r"/api/samples")
    def api_samples_create(self):
        body = self._body()
        try:
            self._json(self.st.create_sample(body), 201)
        except ValueError as exc:
            raise HttpError(400, str(exc)) from exc

    @route("GET", r"/api/samples/([^/]+)")
    def api_sample(self, name):
        self._json(self.st.detail(name))

    @route("PATCH", r"/api/samples/([^/]+)")
    def api_sample_patch(self, name):
        body = self._body()
        ws = self.st.ws
        self.st._record(name)  # noqa: SLF001 -- 404 if missing
        new_name = (body.get("name") or "").strip()
        if new_name and new_name != name:
            if self.st.jobs.active_for(name):
                raise HttpError(409, "Wait for this sample's jobs to finish before renaming it")
            try:
                ws.rename(name, new_name)
            except ValueError as exc:
                raise HttpError(400, str(exc)) from exc
            self.st.bus.publish("renamed", {"from": name, "to": new_name})
            # Its Results projects name it anew.
            self.st.bus.publish("settings", ws.settings())
            name = new_name
        if "meta" in body:
            meta = {k: body["meta"].get(k) for k in META_FIELDS if body["meta"].get(k) not in (None, "")}

            def apply(rec):
                rec["meta"] = meta
            ws.update(name, apply)
            ws.group_slot(group_of(meta))
        self.st.publish_sample(name)
        self._json(self.st.detail(name))

    @route("DELETE", r"/api/samples/([^/]+)")
    def api_sample_delete(self, name):
        if self.st.jobs.active_for(name):
            raise HttpError(409, "Cancel this sample's jobs before removing it")
        storage = self.st.ws.storage_dir(self.st._record(name))  # noqa: SLF001 -- 404 if missing
        try:
            dest = self.st.ws.archive(name)
        except KeyError as exc:
            raise HttpError(404, f"No sample named {name}") from exc
        self.st.bus.publish("removed", {"name": name})
        # The large files beside the scan stay there; say where, so they can be found.
        self._json({"archived_to": str(dest),
                    "left_beside_scan": str(storage) if storage is not None and storage.is_dir() else None})

    @route("POST", r"/api/flags")
    def api_flags(self):
        """Flag specimens -- left out of some or all analyses, with a note -- or, with no flag, include them again."""
        body = self._body()
        names = [str(n) for n in body.get("names") or []]
        if not names:
            raise HttpError(400, "Name the specimens to flag")
        flag = body.get("flag")
        if flag is not None:
            scopes = flag.get("from", "all")
            if scopes != "all":
                if not isinstance(scopes, list) or not scopes:
                    raise HttpError(400, "Pick at least one analysis to leave the specimen out of")
                unknown = [k for k in scopes if k not in results.EXCLUSION_SCOPES]
                if unknown:
                    raise HttpError(400, f"There is no analysis called {unknown[0]}")
                if set(scopes) >= set(results.EXCLUSION_SCOPES):
                    scopes = "all"
                else:
                    scopes = [k for k in results.EXCLUSION_SCOPES if k in scopes]
            note = " ".join(str(flag.get("note") or "").split())
        for name in names:
            self.st._record(name)  # noqa: SLF001 -- 404 before anything is written

        def apply(rec):
            if flag is None:
                rec.pop("excluded", None)
            else:
                # A flag keeps the day it was first raised when its note is edited.
                since = (rec.get("excluded") or {}).get("at") or now_iso()
                rec["excluded"] = {"from": scopes, "note": note, "at": since}
        for name in names:
            self.st.ws.update(name, apply)
            self.st.publish_sample(name)
        self._json({"names": names, "excluded": flag is not None})

    # -- library folders --------------------------------------------------
    # Folders are written with slashes ("Odam study/Batch 2"), so they travel in
    # the body rather than the URL.

    @route("GET", r"/api/folders")
    def api_folders(self):
        self._json({"folders": self.st.ws.folders()})

    @route("POST", r"/api/folders")
    def api_folder_add(self):
        try:
            folder = self.st.ws.add_folder(self._body().get("folder"))
        except ValueError as exc:
            raise HttpError(409 if isinstance(exc, FolderExists) else 400, str(exc)) from exc
        self.st.publish_folders()
        self._json({"folder": folder, "folders": self.st.ws.folders()}, 201)

    @route("POST", r"/api/folders/rename")
    def api_folder_rename(self):
        """Rename a folder, or move it into another: {from, to}, each a whole path."""
        body = self._body()
        try:
            folder, moved = self.st.ws.rename_folder(body.get("from"), body.get("to"))
        except KeyError as exc:
            raise HttpError(404, f"There is no folder called {body.get('from')}") from exc
        except ValueError as exc:
            raise HttpError(409 if isinstance(exc, FolderExists) else 400, str(exc)) from exc
        self._refiled(moved)
        self._json({"folder": folder, "moved": moved, "folders": self.st.ws.folders()})

    @route("POST", r"/api/folders/remove")
    def api_folder_remove(self):
        """Remove a folder; the specimens and folders in it move up a level. No specimen is removed."""
        body = self._body()
        try:
            moved = self.st.ws.remove_folder(body.get("folder"))
        except KeyError as exc:
            raise HttpError(404, f"There is no folder called {body.get('folder')}") from exc
        except ValueError as exc:
            raise HttpError(400, str(exc)) from exc
        self._refiled(moved)
        self._json({"moved": moved, "folders": self.st.ws.folders()})

    @route("POST", r"/api/folders/move")
    def api_folder_move(self):
        """File specimens in a folder: {names, folder}, "" for the top of the library."""
        body = self._body()
        names = [str(n) for n in body.get("names") or []]
        if not names:
            raise HttpError(400, "Name the specimens to move")
        try:
            folder = self.st.ws.move_to_folder(names, body.get("folder"))
        except KeyError as exc:
            raise HttpError(404, f"No sample named {exc.args[0]}") from exc
        except ValueError as exc:
            raise HttpError(400, str(exc)) from exc
        self._refiled(names)
        self._json({"names": names, "folder": folder, "folders": self.st.ws.folders()})

    def _refiled(self, names: list[str]) -> None:
        self.st.publish_folders()
        for name in names:
            self.st.publish_sample(name)

    @route("GET", r"/api/meta/values")
    def api_meta_values(self):
        self._json(self.st.ws.known_values())

    @route("POST", r"/api/names/compose")
    def api_compose(self):
        """The name one import would get, by the rule a batch is named by.

        With ``current``, the name an existing specimen's details give it:
        its own name does not count as taken.
        """
        body = self._body()
        meta = dict(body.get("meta") or {})
        current = str(body.get("current") or "")
        planned = self.st.ws.plan_names([{"meta": meta, "found": body.get("found") or ""}],
                                        ignore=[current] if current else [])[0]
        # With neither enough details nor a name of its own, there is no name yet.
        unnamed = planned["from"] == "found" and not safe_name(body.get("found") or "")
        self._json({"name": "" if unnamed else planned["name"], "replicate": planned["replicate"],
                    "from": planned["from"]})

    @route("POST", r"/api/names/plan")
    def api_names_plan(self):
        items = self._body().get("items") or []
        self._json({"names": self.st.ws.plan_names(items[:5000])})

    @route("POST", r"/api/samples/([^/]+)/crop")
    def api_crop(self, name):
        body = self._body()
        box = body.get("box")
        if box is not None:
            if len(box) != 6:
                raise HttpError(400, "A crop box is six numbers: z0 z1 y0 y1 x0 x1")
            box = [int(round(float(v))) for v in box]
            if any(box[i + 1] <= box[i] for i in (0, 2, 4)):
                raise HttpError(400, "Each crop range must end after it starts")
        self.st.ws.update(name, lambda r: r.update(crop_box=box))
        self.st.publish_sample(name)
        self._json({"crop_box": box})

    @route("POST", r"/api/samples/([^/]+)/crop/suggest")
    def api_crop_suggest(self, name):
        record = self.st._record(name)  # noqa: SLF001
        folder = steps.view_dir(self.st.ws, record["name"], "source")
        meta = read_json(folder / "view.json")
        if not meta:
            raise HttpError(409, "The raw scan preview is not built yet")
        q = read_gz(folder / "base.u16.gz", "<u2", meta["shape"])
        margin = float(self._body().get("margin_mm") or 0.5)
        self._json({"box": suggest_crop(q, meta, margin_mm=margin)})

    @route("GET", r"/api/samples/([^/]+)/marks")
    def api_marks(self, name):
        self._json(self.st.read_marks(name))

    @route("POST", r"/api/samples/([^/]+)/marks")
    def api_marks_save(self, name):
        body = self._body()
        self._json(self.st.save_marks(name, body.get("marks") or [], bool(body.get("run"))))

    @route("GET", r"/api/samples/([^/]+)/incisor")
    def api_incisor(self, name):
        record = self.st._record(name)  # noqa: SLF001
        profile = results.incisor_profile(self.st.ws, record)
        run = read_json(self.st.ws.artifact(record, "incisor_run")) if self.st.ws.artifact(record, "incisor_run") else None
        cal = results.ha_calibration(self.st.ws, record) if profile else None
        self._json({"profile": profile, "metrics": results.incisor_metrics(profile, cal), "calibration": cal,
                    "landmarks": results.landmarks(self.st.ws, record),
                    "run": {k: (run or {}).get(k) for k in ("orientation", "dentin_baseline", "status_counts")}})

    @route("GET", r"/api/samples/([^/]+)/molar")
    def api_molar(self, name):
        record = self.st._record(name)  # noqa: SLF001
        ws = self.st.ws
        profile_path = ws.artifact(record, "molar_profile")
        profile = results._read_rows(profile_path) if profile_path else []  # noqa: SLF001
        self._json({"summary": results.molar(ws, record),
                    "profile": [{"z": int(r["z"]), "z_mm": float(r["z_mm"]), "tooth": r.get("tooth"),
                                 "fraction": float(r["enamel_fraction"])} for r in profile]})

    @route("GET", r"/api/samples/([^/]+)/files/([a-z_]+)")
    def api_sample_file(self, name, key):
        record = self.st._record(name)  # noqa: SLF001
        path = self.st.ws.artifact(record, key)
        if path is None or not path.is_file():
            raise HttpError(404, f"{ARTIFACT_LABELS.get(key, key)} is not available")
        download = path.name if self.query.get("download") else None
        self._file(path, download=download)

    # -- views ------------------------------------------------------------

    @route("GET", r"/api/samples/([^/]+)/views")
    def api_views(self, name):
        self._json(steps.views_for(self.st.ws, self.st._record(name)))  # noqa: SLF001

    @route("POST", r"/api/samples/([^/]+)/views/([a-z]+)/build")
    def api_view_build(self, name, view):
        if view not in steps.VIEWS:
            raise HttpError(404, f"No view {view}")
        self.st._record(name)  # noqa: SLF001
        made = self.st.submit_previews(name, [view], force=bool(self._body().get("force")))
        self._json({"jobs": made})

    @route("GET", r"/api/samples/([^/]+)/views/([a-z]+)/meta")
    def api_view_meta(self, name, view):
        record = self.st._record(name)  # noqa: SLF001
        folder = steps.view_dir(self.st.ws, name, view)
        meta = read_json(folder / "view.json")
        if not meta:
            raise HttpError(404, "Not built yet")
        base, overlays = steps.view_inputs(self.st.ws, record, view)
        from .volumes import view_is_current  # noqa: PLC0415
        meta["current"] = view_is_current(meta, base, overlays)
        # What converts the preview's values to mg HA/cm³, so specimens can share one heatmap scale.
        meta["calibration"] = results.view_calibration(self.st.ws, record, view)
        self._json(meta)

    @route("GET", r"/api/samples/([^/]+)/views/([a-z]+)/file/([A-Za-z0-9_.]+)")
    def api_view_file(self, name, view, filename):
        if not VIEW_FILE.match(filename):
            raise HttpError(404, "Not a preview file")
        path = steps.view_dir(self.st.ws, name, view) / filename
        if filename.endswith(".gz"):
            self._file(path, "application/octet-stream", encoding="gzip", cache=True)
        else:
            self._file(path, "image/png", cache=True)

    # -- running ----------------------------------------------------------

    @route("POST", r"/api/run")
    def api_run(self):
        body = self._body()
        names = body.get("samples") or []
        requested = body.get("steps") or "default"
        if requested != "default":
            unknown = [s for s in requested if s not in steps.STEPS]
            if unknown:
                raise HttpError(400, f"Unknown step {unknown[0]}")
        self._json(self.st.run(names, requested, body.get("params") or {}))

    @route("GET", r"/api/jobs")
    def api_jobs(self):
        self._json(self.st.jobs.list())

    @route("GET", r"/api/jobs/([A-Za-z0-9-]+)")
    def api_job(self, job_id):
        job = self.st.jobs.jobs.get(job_id)
        if job is None:
            raise HttpError(404, "No such job")
        data = job.public()
        data["outputs"] = steps.tool_outputs(job)
        self._json(data)

    @route("GET", r"/api/jobs/([A-Za-z0-9-]+)/log")
    def api_job_log(self, job_id):
        offset = int(self.query.get("offset") or 0)
        text, end = self.st.jobs.read_log(job_id, offset)
        self._json({"text": text, "offset": end})

    @route("GET", r"/api/jobs/([A-Za-z0-9-]+)/output")
    def api_job_output(self, job_id):
        job = self.st.jobs.jobs.get(job_id)
        if job is None:
            raise HttpError(404, "No such job")
        wanted = self.query.get("name") or ""
        match = next((o for o in steps.tool_outputs(job) if o["name"] == wanted), None)
        if match is None:
            raise HttpError(404, "No such output")
        self._file(Path(match["path"]), download=Path(wanted).name if self.query.get("download") else None)

    @route("POST", r"/api/jobs/([A-Za-z0-9-]+)/cancel")
    def api_job_cancel(self, job_id):
        if job_id not in self.st.jobs.jobs:
            raise HttpError(404, "No such job")
        self._json(self.st.jobs.cancel(job_id).public())

    @route("POST", r"/api/jobs/([A-Za-z0-9-]+)/retry")
    def api_job_retry(self, job_id):
        if job_id not in self.st.jobs.jobs:
            raise HttpError(404, "No such job")
        self._json(self.st.jobs.retry(job_id).public())

    @route("POST", r"/api/jobs/clear")
    def api_jobs_clear(self):
        self._json({"cleared": self.st.jobs.clear_finished()})

    @route("POST", r"/api/tools/([a-z_]+)")
    def api_tool(self, tool_id):
        body = self._body()
        tool = steps.TOOLS.get("ml_train" if tool_id == "training_set" else tool_id)
        if tool is None:
            raise HttpError(404, f"No tool {tool_id}")
        if tool.disabled:
            raise HttpError(409, tool.disabled)
        if tool_id == "training_set":
            self._json(steps.build_training_set(self.st.ws))
            return
        job = self.st.jobs.submit(step=f"tool:{tool_id}", title=tool.label, params=body.get("params") or {},
                                  lane=tool.lane)
        self._json(job.public(), 201)

    # -- uploads ----------------------------------------------------------

    @route("POST", r"/api/uploads")
    def api_upload_new(self):
        uid = uuid.uuid4().hex
        (self.st.ws.uploads_dir / uid).mkdir(parents=True)
        self._json({"id": uid}, 201)

    @route("PUT", r"/api/uploads/([a-f0-9]+)/(.+)")
    def api_upload_put(self, uid, rel):
        if not UPLOAD_ID.match(uid):
            raise HttpError(400, "Bad upload id")
        root = (self.st.ws.uploads_dir / uid).resolve()
        if not root.is_dir():
            raise HttpError(404, "Unknown upload")
        parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".", "..")]
        if not parts:
            raise HttpError(400, "No file name")
        target = root.joinpath(*parts).resolve()
        if root not in target.parents:
            raise HttpError(400, "Bad file path")
        target.parent.mkdir(parents=True, exist_ok=True)
        length = self._unread
        remaining = length
        with open(target, "wb") as handle:
            while remaining > 0:
                block = self._read(min(remaining, 4 << 20))
                if not block:
                    break
                handle.write(block)
                remaining -= len(block)
        if remaining:
            target.unlink(missing_ok=True)
            raise HttpError(400, "The upload was cut short")
        self._json({"bytes": length})

    def _details(self, name: str, kind: str, path: Path, folders: list[str],
                 artifacts: dict | None = None) -> dict:
        """What one scan's name, header and folder say about it, as a scan import reads them."""
        known = self.st.ws.known_values() if self.st.ws is not None and self.st.ws.root.is_dir() else {}
        source = {"kind": "results", "path": str(path)} if kind == "segmentation" else {"kind": kind, "path": str(path)}
        cand = describe({"name": name, "meta": parse_name(name), "source": source,
                         "artifacts": artifacts or {}, "folders": folders}, known)
        return {"meta": cand["meta"], "sources": cand["sources"], "recorded": cand["recorded"]}

    @route("GET", r"/api/uploads/([a-f0-9]+)")
    def api_upload_info(self, uid):
        root = self.st.ws.uploads_dir / uid
        if not UPLOAD_ID.match(uid) or not root.is_dir():
            raise HttpError(404, "Unknown upload")
        entries = [p for p in root.iterdir() if not p.name.startswith(".")]
        target = entries[0] if len(entries) == 1 else root
        size, count = directory_size(root)
        name = target.stem if target.is_file() else target.name
        kind = detect_kind(target)
        info = {"kind": kind, "format": source_format(kind, target), "bytes": size, "files": count,
                "name": target.name, "found": "" if target == root else name}
        # An upload's folder is named for the upload, so it says nothing about the scan.
        info.update(self._details(name, kind, target, folders=[]))
        self._json(info)

    @route("DELETE", r"/api/uploads/([a-f0-9]+)")
    def api_upload_delete(self, uid):
        root = self.st.ws.uploads_dir / uid
        if UPLOAD_ID.match(uid) and root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
        self._json({"ok": True})

    # -- importing from disk ---------------------------------------------

    @route("GET", r"/api/fs")
    def api_fs(self):
        raw = self.query.get("path") or ""
        path = Path(raw).expanduser() if raw else Path.home()
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            raise HttpError(404, f"{path} is not a folder")
        entries = []
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.name.startswith("."):
                        continue
                    try:
                        is_dir = entry.is_dir()
                        size = None if is_dir else entry.stat().st_size
                    except OSError:
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    hint = {".isq": "isq", ".rsq": "rsq", ".nrrd": "volume", ".tif": "volume",
                            ".tiff": "volume", ".npz": "transform", ".csv": "table"}.get(ext)
                    if is_dir and os.path.isfile(os.path.join(entry.path, "studio.json")):
                        hint = "workspace"
                    entries.append({"name": entry.name, "path": entry.path, "dir": is_dir,
                                    "size": size, "hint": hint})
        except PermissionError as exc:
            raise HttpError(403, f"No permission to read {path}") from exc
        entries.sort(key=lambda e: (not e["dir"], e["name"].lower()))
        roots = [{"label": "Home", "path": str(Path.home())}]
        if self.st.ws is not None:
            roots.append({"label": "Workspace", "path": str(self.st.ws.root)})
        roots += [{"label": label, "path": str(root)} for label, root in drives.browse_roots()]
        crumbs = [{"label": p.name or str(p), "path": str(p)} for p in reversed([path, *path.parents])]
        self._json({"path": str(path), "parent": str(path.parent) if path.parent != path else None,
                    "crumbs": crumbs, "entries": entries[:5000], "truncated": len(entries) > 5000,
                    "roots": roots})

    @route("GET", r"/api/fs/inspect")
    def api_fs_inspect(self):
        path = Path(self.query.get("path") or "").expanduser()
        if not path.exists():
            raise HttpError(404, f"{path} does not exist")
        kind = detect_kind(path)
        size, count = directory_size(path, limit_files=20_000)
        name = path.stem if path.is_file() else path.name
        info = {"path": str(path), "kind": kind, "format": source_format(kind, path), "bytes": size,
                "files": count, "name": name, "found": name, "meta": parse_name(name)}
        if path.is_dir():
            info["workspace"] = is_workspace(path)
            info["sharing"] = drives.sharing_note(path)
        if kind == "segmentation":
            info["artifacts"] = artifacts_in(path, path.name)
        if kind in ("isq", "rsq", "dicom", "bmp", "volume", "segmentation"):
            info.update(self._details(name, kind, path, folders=[path.parent.name],
                                      artifacts=info.get("artifacts")))
        if kind in ("isq", "dicom", "bmp", "volume"):
            # Where a link would put the large files: beside the scan, on its own drive.
            root = storage_root(path)
            info["storage"] = {"path": str(root), "synced": synced_service(root)}
        self._json(info)

    @route("POST", r"/api/import/scan")
    def api_import_scan(self):
        body = self._body()
        paths = [p for p in body.get("paths") or [] if p]
        if not paths:
            raise HttpError(400, "Give at least one folder")
        names = set(self.st.ws.names())
        found = describe_all(scan_paths(paths), self.st.ws.known_values())
        imported = self.st.imported_paths()
        for cand in found:
            # Already imported under this name, or under another one it was given then.
            linked = [cand["source"].get("path") if cand["source"].get("kind") != "results" else None,
                      *cand["artifacts"].values()]
            keys = [os.path.normcase(os.path.realpath(p)) for p in linked if p]
            cand["existing"] = cand["name"] if cand["name"] in names else next(
                (imported[k] for k in keys if k in imported), None)
            cand["exists"] = cand["existing"] is not None
        self._json({"samples": found})

    @route("POST", r"/api/import/commit")
    def api_import_commit(self):
        self._json(self.st.import_many(self._body()), 201)

    # -- results ----------------------------------------------------------

    @route("GET", r"/api/results")
    def api_results(self):
        self._json(results.gather(self.st.ws))

    # A project scopes the Results page: {name, names, by, hidden}. Each change
    # reaches every browser with the workspace's settings.
    @route("POST", r"/api/results/projects")
    def api_project_add(self):
        try:
            project = self.st.ws.create_project(self._body())
        except ValueError as exc:
            raise HttpError(409 if isinstance(exc, ProjectExists) else 400, str(exc)) from exc
        self.st.bus.publish("settings", self.st.ws.settings())
        self._json(project, 201)

    @route("PATCH", r"/api/results/projects/([a-f0-9]+)")
    def api_project_patch(self, project_id):
        try:
            project = self.st.ws.update_project(project_id, self._body())
        except KeyError as exc:
            raise HttpError(404, "There is no such project") from exc
        except ValueError as exc:
            raise HttpError(409 if isinstance(exc, ProjectExists) else 400, str(exc)) from exc
        self.st.bus.publish("settings", self.st.ws.settings())
        self._json(project)

    @route("DELETE", r"/api/results/projects/([a-f0-9]+)")
    def api_project_remove(self, project_id):
        try:
            self.st.ws.remove_project(project_id)
        except KeyError as exc:
            raise HttpError(404, "There is no such project") from exc
        self.st.bus.publish("settings", self.st.ws.settings())
        self._json({"removed": project_id})

    @route("GET", r"/api/export/([a-z]+)")
    def api_export(self, kind):
        try:
            # ?s=a,b,c: those specimens alone, as the results page shows them.
            names = [n for n in (self.query.get("s") or "").split(",") if n] or None
            filename, text = results.export(self.st.ws, kind, names)
        except KeyError as exc:
            raise HttpError(404, f"No export {kind}") from exc
        ctype = "text/csv; charset=utf-8" if filename.endswith(".csv") else "text/plain; charset=utf-8"
        self._send(200, text.encode("utf-8"), ctype,
                   {"Content-Disposition": f'attachment; filename="{filename}"'})

    # -- desktop conveniences ---------------------------------------------

    @route("POST", r"/api/reveal")
    def api_reveal(self):
        # Absolute, so it starts with "/" (or a drive): never read as an option by open or xdg-open.
        path = Path(str(self._body().get("path") or "")).expanduser().resolve()
        if not path.exists():
            raise HttpError(404, f"{path} does not exist")
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        elif os.name == "nt":
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path if path.is_dir() else path.parent)])
        self._json({"ok": True})

    @route("POST", r"/api/napari")
    def api_napari(self):
        body = self._body()
        record = self.st._record(body.get("sample") or "")  # noqa: SLF001
        volume = self.st.ws.artifact(record, body.get("artifact") or "reoriented")
        if volume is None:
            raise HttpError(404, "That volume is not available")
        argv = [PYTHON, str(SCRIPTS["view_napari"]), str(volume)]
        labels = self.st.ws.artifact(record, body["labels"]) if body.get("labels") else None
        if labels is not None:
            argv += ["--labels", str(labels)]
        if body.get("downsample"):
            argv += ["--downsample", str(int(body["downsample"]))]
        kwargs = {"start_new_session": True} if os.name == "posix" else {}
        subprocess.Popen(argv, cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
        self._json({"ok": True})


# Requests that work with no workspace open: choosing one needs these.
WITHOUT_WORKSPACE = {"api_state", "api_events", "api_workspaces", "api_workspace_open", "api_workspace_remove",
                     "api_workspace_restore", "api_fs", "api_fs_inspect", "api_reveal"}


def serve(root: Path | None, host: str = "127.0.0.1", port: int = 8765, remember: bool = True,
          prefs: Path = PREFS, extra_hosts: tuple[str, ...] = ()) -> tuple[ThreadingHTTPServer, Studio]:
    """Start the server; ``extra_hosts`` are further names it answers to (a tailnet's MagicDNS names)."""
    studio = Studio(root, remember=remember, prefs=prefs)
    Handler.studio = studio
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    bound = server.server_address[1]
    names = {"127.0.0.1", "localhost", "[::1]"}
    Handler.remote = host not in ("127.0.0.1", "localhost", "::1")
    if Handler.remote and host:
        names.add(host)
    names.update(n.lower() for n in extra_hosts)
    Handler.allowed_hosts = {f"{n}:{bound}" for n in names}
    return server, studio
