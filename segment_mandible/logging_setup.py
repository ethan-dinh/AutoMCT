"""
Logging configuration for the segmentation pipeline.

Console output goes through ``rich`` when it is available: level-coloured,
column-aligned, with the emitting module and a wrapped message that stays
readable when a check explains itself in a sentence. The file handler stays
plain text so a log can be grepped and diffed.

Two things this setup gets right that a bare ``basicConfig`` does not:

- **Progress bars survive.** ``tqdm`` writes to stderr with carriage returns;
  a log record written underneath it leaves the bar smeared across the
  scrollback. Records are routed through ``tqdm.write`` so the bar is cleared
  first and redrawn after.
- **Re-running is safe.** Handlers are cleared before install, so calling this
  twice in one process (a notebook, the test TUI) does not double every line.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys

try:  # rich is declared in env/environment.yml, but degrade rather than crash
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme
except ImportError:  # pragma: no cover - exercised only on a partial install
    # Bound to None rather than left undefined, so referring to them below is
    # a clean None check instead of a NameError on a partial install.
    Console = None
    RichHandler = None
    Theme = None


# Derived from the names themselves so it cannot drift out of sync with them.
_HAVE_RICH = Console is not None and RichHandler is not None and Theme is not None

LOG_THEME = {
    "logging.level.debug": "dim cyan",
    "logging.level.info": "green",
    "logging.level.warning": "yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",
}


class _TqdmStream:
    """
    A stream that hands writes to ``tqdm.write``.

    tqdm keeps its bar on the current terminal line; anything else written
    there collides with it. ``tqdm.write`` clears the bar, writes the line, and
    redraws -- so log records and progress bars coexist instead of overwriting
    each other. Falls back to a direct write when tqdm is not importable.
    """

    def __init__(self, stream=None):
        self._stream = stream or sys.stderr

    def write(self, message: str) -> None:
        # Imported per-write rather than at module scope: this module is
        # imported by the pipeline's own modules, and a top-level tqdm import
        # here would make logging unusable wherever tqdm is missing -- the one
        # thing that should still work when a dependency is absent.
        try:
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel

            tqdm.write(message.rstrip("\n"), file=self._stream, end="\n")
        except ImportError:
            self._stream.write(message)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except ValueError:  # stream closed during interpreter shutdown
            pass

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


def progress_disabled() -> bool:
    """
    Whether tqdm bars should be suppressed.

    A bar is a live terminal affordance: it is noise in a redirected log and
    pointless under --quiet, where the whole point is to see only problems.
    """
    root_level = logging.getLogger().getEffectiveLevel()
    return not sys.stderr.isatty() or root_level > logging.INFO


class _PlainFormatter(logging.Formatter):
    """Fallback console format when rich is unavailable, with ANSI level colours."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31;1m",
        "CRITICAL": "\033[97;41m",
    }
    RESET = "\033[0m"

    def __init__(self, *args, use_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)
        # Colour a copy so the shared LogRecord reaches the file handler with a
        # clean levelname -- escape codes in a log file are unreadable.
        color = self.COLORS.get(record.levelname, "")
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


def _console_handler(verbose: bool, color: bool) -> logging.Handler:
    """Build the console handler, preferring rich and falling back to ANSI."""
    stream = _TqdmStream()

    if color and RichHandler is not None and Console is not None and Theme is not None:
        handler = RichHandler(
            # _TqdmStream is duck-typed, not an IO[str]: rich only calls
            # write/flush/isatty on it, which it implements.
            console=Console(
                file=stream,  # type: ignore[arg-type]
                theme=Theme(LOG_THEME),
                soft_wrap=False,
            ),
            rich_tracebacks=True,
            tracebacks_show_locals=verbose,
            show_time=True,
            show_path=verbose,
            omit_repeated_times=False,
            markup=False,          # messages carry file paths and brackets
            log_time_format="%H:%M:%S",
        )
        # RichHandler renders level, time and path itself; the formatter only
        # supplies the message body.
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler

    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        _PlainFormatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            use_color=color,
        )
    )
    return handler


def setup_logging(
    out_root: str | None = None,
    *,
    verbose: bool = False,
    quiet: bool = False,
    color: bool = True,
    log_file: str | None = None,
) -> str | None:
    """
    Install console (and optionally file) logging for the pipeline.

    Parameters:
        out_root: Directory to write the run log into. None writes no log file.
        verbose: Log at DEBUG, and include the emitting module and locals in
            tracebacks. DEBUG carries the per-slice tracking decisions and the
            individual passing checks.
        quiet: Log at WARNING -- only problems and check failures.
        color: Colour the console. Forced off when stderr is not a TTY or when
            NO_COLOR is set, so a redirected log stays clean.
        log_file: Explicit log file path, overriding the name derived from
            out_root.

    Returns:
        The path of the log file, or None if no file was written.
    """
    if verbose and quiet:
        raise ValueError("setup_logging: pass at most one of verbose/quiet")

    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO

    # Honour the NO_COLOR convention and drop colour for a redirected stream,
    # so escape codes never end up in a piped-to-file transcript.
    if os.environ.get("NO_COLOR") or not sys.stderr.isatty():
        color = False

    root = logging.getLogger()
    # Clear first: this is called again by the test TUI and by notebooks, and
    # re-adding handlers would duplicate every line.
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()

    handlers: list[logging.Handler] = [_console_handler(verbose, color)]

    resolved_log_file = log_file
    if resolved_log_file is None and out_root is not None:
        os.makedirs(out_root, exist_ok=True)
        resolved_log_file = os.path.join(
            out_root, f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
        )

    if resolved_log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(resolved_log_file)), exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
        # The file always gets DEBUG regardless of the console level: when a
        # run turns out to have gone wrong, the detail is already on disk and
        # the run does not have to be repeated with -vv to find out why.
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    root.setLevel(logging.DEBUG if resolved_log_file else level)
    for handler in handlers:
        if handler.level == logging.NOTSET:
            handler.setLevel(level)
        root.addHandler(handler)

    # These libraries log per-file chatter at INFO that says nothing about the
    # segmentation; leave them at WARNING so the pipeline's own lines stand out.
    for noisy in ("matplotlib", "PIL", "numba", "napari", "vispy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if resolved_log_file:
        logging.getLogger(__name__).info("Logging to %s", resolved_log_file)
    return resolved_log_file
