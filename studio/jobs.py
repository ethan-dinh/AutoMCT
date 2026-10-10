"""
The job queue: every pipeline step runs here, as a subprocess.

Subprocesses rather than threads because the steps are the repository's own
scripts, each of which can hold several GB and none of which is safe to run
twice in one interpreter; a crash or an out-of-memory kill then takes down one
job, never the server.

Jobs run in lanes. ``heavy`` is for anything that loads whole scans
(conversion, segmentation, CMPR, quantification) and defaults to one at a
time, since two segmentations side by side can exhaust a laptop's memory.
``preview`` builds viewer previews next to it, and ``light`` is for small work.
A job can depend on others -- CMPR on segmentation, say -- and starts only once
they have succeeded; if one fails, everything downstream of it is cancelled
rather than run on missing inputs.

A job's command line is built when it starts, not when it is queued, so a step
queued behind another sees the files that step wrote.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .workspace import Workspace, now_iso, read_json, write_json

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
TQDM = re.compile(r"(?P<desc>[^|\r\n]*?)[: ]*(?P<pct>\d{1,3})%\|")
PROGRESS = re.compile(r"@@progress\s+(?P<frac>[0-9.]+)\s*(?P<label>.*)")
RESULT = re.compile(r"@@result\s+(?P<json>\{.*\})\s*$")

ACTIVE = ("queued", "waiting", "running")
FINAL = ("done", "failed", "cancelled")


@dataclass
class Launch:
    """What a step hands back when its job starts: the command to run."""

    argv: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    # (regex, fraction, label): log lines that mark the job's progress.
    stages: list[tuple[str, float, str]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    id: str
    step: str
    title: str
    sample: str | None = None
    params: dict = field(default_factory=dict)
    lane: str = "heavy"
    deps: list[str] = field(default_factory=list)
    status: str = "queued"
    created: str = field(default_factory=now_iso)
    started: str | None = None
    finished: str | None = None
    progress: float | None = None
    stage: str = ""
    error: str | None = None
    exit_code: int | None = None
    argv: list[str] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    batch: str | None = None

    def public(self) -> dict:
        data = asdict(self)
        data.pop("context", None)
        return data


class JobManager:
    """Queue, schedule, run and record jobs. One per server."""

    def __init__(self, ws: Workspace, *, prepare: Callable[[Job], Launch],
                 finish: Callable[[Job], None], publish: Callable[[str, dict], None],
                 failed: Callable[[Job], None] | None = None):
        self.ws = ws
        self._prepare = prepare
        self._finish = finish
        self._publish = publish
        self._failed = failed
        self.jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._cancelled: set[str] = set()
        self._cond = threading.Condition()
        self._last_publish: dict[str, float] = {}
        self._stop = False
        self._load()
        self._thread = threading.Thread(target=self._schedule_loop, name="scheduler", daemon=True)
        self._thread.start()

    # -- persistence ------------------------------------------------------

    def _path(self, job_id: str) -> Path:
        return self.ws.jobs_dir / f"{job_id}.json"

    def log_path(self, job_id: str) -> Path:
        return self.ws.jobs_dir / f"{job_id}.log"

    def _save(self, job: Job) -> None:
        write_json(self._path(job.id), asdict(job))

    def _load(self) -> None:
        records = sorted(self.ws.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)[-400:]
        for path in records:
            data = read_json(path)
            if not data:
                continue
            try:
                job = Job(**data)
            except TypeError:
                continue
            if job.status in ACTIVE:
                # The server that owned it is gone; its process went with it.
                job.status = "failed"
                job.error = "Interrupted: the studio was stopped while this job was queued or running"
                job.finished = job.finished or now_iso()
                self._save(job)
            self.jobs[job.id] = job

    # -- public API -------------------------------------------------------

    def submit(self, *, step: str, title: str, sample: str | None = None, params: dict | None = None,
               lane: str = "heavy", deps: list[str] | None = None, batch: str | None = None,
               context: dict | None = None) -> Job:
        job = Job(id=datetime.datetime.now().strftime("%y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                  step=step, title=title, sample=sample, params=params or {}, lane=lane,
                  deps=list(deps or []), batch=batch, context=context or {})
        job.status = "waiting" if job.deps else "queued"
        with self._cond:
            self.jobs[job.id] = job
            self._save(job)
            self._cond.notify_all()
        self._emit(job, force=True)
        return job

    def cancel(self, job_id: str) -> Job:
        with self._cond:
            job = self.jobs[job_id]
            if job.status in FINAL:
                return job
            self._cancelled.add(job_id)
            proc = self._procs.get(job_id)
            if proc is None:
                self._mark(job, "cancelled", error="Cancelled before it started")
        if proc is not None:
            self._terminate(proc)
        return job

    def retry(self, job_id: str) -> Job:
        old = self.jobs[job_id]
        return self.submit(step=old.step, title=old.title, sample=old.sample, params=old.params,
                           lane=old.lane, batch=old.batch, context=old.context)

    def clear_finished(self) -> int:
        with self._cond:
            gone = [j for j in self.jobs.values() if j.status in FINAL]
            for job in gone:
                self.jobs.pop(job.id, None)
                for path in (self._path(job.id), self.log_path(job.id)):
                    try:
                        path.unlink()
                    except OSError:
                        pass
        self._publish("jobs", {"cleared": len(gone)})
        return len(gone)

    def list(self, limit: int = 300) -> list[dict]:
        jobs = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)[:limit]
        return [j.public() for j in jobs]

    def active_for(self, sample: str) -> list[Job]:
        return [j for j in self.jobs.values() if j.sample == sample and j.status in ACTIVE]

    def read_log(self, job_id: str, offset: int = 0, limit: int = 256_000) -> tuple[str, int]:
        path = self.log_path(job_id)
        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                chunk = handle.read(limit)
        except OSError:
            return "", offset
        return chunk.decode("utf-8", errors="replace"), offset + len(chunk)

    def shutdown(self) -> None:
        self._stop = True
        with self._cond:
            self._cond.notify_all()
            procs = list(self._procs.values())
        for proc in procs:
            self._terminate(proc)

    # -- scheduling -------------------------------------------------------

    def _mark(self, job: Job, status: str, **fields) -> None:
        job.status = status
        for key, value in fields.items():
            setattr(job, key, value)
        if status in FINAL:
            job.finished = now_iso()
        self._save(job)
        if status == "failed" and self._failed is not None:
            try:
                self._failed(job)
            except Exception:  # noqa: BLE001 -- recording a failure must not raise
                pass
        self._emit(job, force=True)

    def _emit(self, job: Job, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_publish.get(job.id, 0.0) < 0.25:
            return
        self._last_publish[job.id] = now
        self._publish("job", job.public())

    def _schedule_loop(self) -> None:
        while not self._stop:
            with self._cond:
                self._promote_waiting()
                started = self._start_runnable()
                if not started:
                    self._cond.wait(timeout=1.0)

    def _promote_waiting(self) -> None:
        for job in list(self.jobs.values()):
            if job.status != "waiting":
                continue
            deps = [self.jobs.get(d) for d in job.deps]
            if any(d is None or d.status in ("failed", "cancelled") for d in deps):
                bad = next((d for d in deps if d is None or d.status in ("failed", "cancelled")), None)
                reason = f"{bad.title} {bad.status}" if bad else "a job it depended on is gone"
                self._mark(job, "cancelled", error=f"Not run: {reason}")
            elif all(d.status == "done" for d in deps):
                job.status = "queued"
                self._save(job)
                self._emit(job, force=True)

    def _start_runnable(self) -> bool:
        capacity = self.ws.settings().get("lanes", {})
        running: dict[str, int] = {}
        for job in self.jobs.values():
            if job.status == "running":
                running[job.lane] = running.get(job.lane, 0) + 1
        started = False
        for job in sorted(self.jobs.values(), key=lambda j: j.created):
            if job.status != "queued":
                continue
            limit = max(1, int(capacity.get(job.lane, 1)))
            if running.get(job.lane, 0) >= limit:
                continue
            running[job.lane] = running.get(job.lane, 0) + 1
            job.status = "running"
            job.started = now_iso()
            job.progress = 0.0
            self._save(job)
            self._emit(job, force=True)
            threading.Thread(target=self._run, args=(job,), name=f"job-{job.id}", daemon=True).start()
            started = True
        return started

    # -- running ----------------------------------------------------------

    def _run(self, job: Job) -> None:
        log = open(self.log_path(job.id), "ab")

        def write(line: str) -> None:
            log.write((line + "\n").encode("utf-8", errors="replace"))
            log.flush()

        try:
            launch = self._prepare(job)
        except Exception as exc:  # noqa: BLE001 -- any preparation error fails just this job
            write(f"[studio] Could not start: {exc}")
            log.close()
            self._mark(job, "failed", error=str(exc))
            self._wake()
            return

        job.argv = launch.argv
        job.context.update(launch.context)
        write(f"[studio] {now_iso()} {job.title}")
        write("[studio] $ " + " ".join(_quote(a) for a in launch.argv))
        # UTF-8 mode: on Windows a tool's own open() and print() would otherwise
        # use the ANSI code page and fail on the µ and — its logs are full of.
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb",
               "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **launch.env}
        # The repo root, so `-m studio.runners.*` resolves from any cwd.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parent.parent)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        stages = [(re.compile(p), f, label) for p, f, label in launch.stages]
        stage_idx = -1

        try:
            kwargs: dict[str, Any] = {}
            if os.name == "posix":
                kwargs["start_new_session"] = True
            else:
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            proc = subprocess.Popen(launch.argv, cwd=launch.cwd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs)
        except OSError as exc:
            write(f"[studio] Could not launch: {exc}")
            log.close()
            self._mark(job, "failed", error=f"Could not launch: {exc}")
            self._wake()
            return

        with self._cond:
            self._procs[job.id] = proc
            if job.id in self._cancelled:
                self._terminate(proc)

        tail: list[str] = []
        buffer = ""
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            pieces = re.split(r"(\r\n|\n|\r)", buffer)
            buffer = pieces.pop() if pieces else ""
            for text, sep in zip(pieces[0::2], pieces[1::2]):
                text = ANSI.sub("", text).rstrip()
                if not text:
                    continue
                if sep != "\r":
                    write(text)
                    tail.append(text)
                    del tail[:-40]
                stage_idx = self._track(job, text, stages, stage_idx)
        if buffer.strip():
            write(ANSI.sub("", buffer).rstrip())
        code = proc.wait()
        log.close()
        with self._cond:
            self._procs.pop(job.id, None)

        job.exit_code = code
        if job.id in self._cancelled:
            self._mark(job, "cancelled", error="Cancelled while running")
        elif code != 0:
            reason = next((t for t in reversed(tail) if re.search(r"(?i)error|exception|failed", t)),
                          tail[-1] if tail else "")
            self._mark(job, "failed", error=f"Exited with code {code}. {reason}".strip())
        else:
            try:
                self._finish(job)
            except Exception as exc:  # noqa: BLE001
                with open(self.log_path(job.id), "ab") as handle:
                    handle.write(f"[studio] Finished, but its outputs could not be recorded: {exc}\n".encode())
                self._mark(job, "failed", error=f"Outputs not recorded: {exc}")
            else:
                self._mark(job, "done", progress=1.0)
        self._wake()

    def _track(self, job: Job, text: str, stages, stage_idx: int) -> int:
        """Update progress from one output line; returns the current stage index."""
        m = PROGRESS.search(text)
        if m:
            job.progress = min(1.0, float(m.group("frac")))
            job.stage = m.group("label").strip() or job.stage
            self._emit(job)
            return stage_idx
        m = RESULT.search(text)
        if m:
            try:
                job.result.update(json.loads(m.group("json")))
            except ValueError:
                pass
            return stage_idx
        for i, (pattern, fraction, label) in enumerate(stages):
            if i > stage_idx and pattern.search(text):
                stage_idx = i
                job.progress = max(job.progress or 0.0, fraction)
                job.stage = label
                self._emit(job)
                return stage_idx
        m = TQDM.search(text)
        if m:
            pct = min(100, int(m.group("pct"))) / 100.0
            start = stages[stage_idx][1] if stage_idx >= 0 else 0.0
            end = stages[stage_idx + 1][1] if 0 <= stage_idx + 1 < len(stages) else 1.0
            if stage_idx < 0 and stages:
                end = stages[0][1] or 1.0
            job.progress = max(job.progress or 0.0, start + pct * (end - start))
            desc = m.group("desc").strip()
            if desc:
                job.stage = desc
            self._emit(job)
        return stage_idx

    def _terminate(self, proc: subprocess.Popen) -> None:
        def kill():
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    # The whole tree, as killpg does: a tool may have started its own workers.
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   capture_output=True, check=False)
                proc.wait(timeout=8)
            except (ProcessLookupError, PermissionError):
                return
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except (ProcessLookupError, PermissionError):
                    pass
        threading.Thread(target=kill, daemon=True).start()

    def _wake(self) -> None:
        with self._cond:
            self._cond.notify_all()


def _quote(arg: str) -> str:
    return f"'{arg}'" if (" " in arg or not arg) else arg


PYTHON = sys.executable
