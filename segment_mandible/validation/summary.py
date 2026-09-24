"""
Run-level summary of the per-stage checks.

A batch run emits hundreds of log lines, and the two facts that matter at the
end -- which samples are trustworthy and which need a human -- are buried in
them. This collects each sample's reports and prints one table, so the answer
is at the bottom of the run rather than scattered through it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from .checks import CheckReport, Severity

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover
    # Bound to None rather than left undefined, so the call sites narrow on a
    # value check instead of risking a NameError on a partial install.
    Console = None
    Table = None


# Derived from the names themselves so it cannot drift out of sync with them.
_HAVE_RICH = Console is not None and Table is not None

_STATUS_STYLE = {
    "FAIL": ("FAIL", "bold red"),
    "WARN": ("WARN", "yellow"),
    "OK": ("OK", "green"),
}


@dataclass
class SampleValidation:
    """Every stage report for one sample."""

    sample: str
    reports: list[CheckReport] = field(default_factory=list)

    def add(self, report: CheckReport) -> CheckReport:
        self.reports.append(report)
        return report

    @property
    def status(self) -> str:
        if any(r.failed for r in self.reports):
            return "FAIL"
        if any(r.warned for r in self.reports):
            return "WARN"
        return "OK"

    @property
    def problems(self) -> list[tuple[str, str, str]]:
        """(stage, check name, message) for every non-passing result."""
        return [
            (report.stage, result.name, result.message)
            for report in self.reports
            for result in report.problems
        ]

    def to_dict(self) -> dict:
        return {
            "sample": self.sample,
            "status": self.status,
            "stages": [
                {
                    "stage": report.stage,
                    "results": [
                        {
                            "name": result.name,
                            "severity": result.severity.value,
                            "message": result.message,
                            "value": result.value,
                        }
                        for result in report.results
                    ],
                }
                for report in self.reports
            ],
        }


class ValidationSummary:
    """Collects per-sample validation across a run and reports it at the end."""

    def __init__(self) -> None:
        self.samples: list[SampleValidation] = []

    def sample(self, name: str) -> SampleValidation:
        """Start (or resume) collecting checks for one sample."""
        for existing in self.samples:
            if existing.sample == name:
                return existing
        entry = SampleValidation(name)
        self.samples.append(entry)
        return entry

    @property
    def failed(self) -> list[SampleValidation]:
        return [s for s in self.samples if s.status == "FAIL"]

    @property
    def warned(self) -> list[SampleValidation]:
        return [s for s in self.samples if s.status == "WARN"]

    def write_json(self, path: str) -> None:
        """Write the full report as JSON for downstream analysis scripts."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"samples": [s.to_dict() for s in self.samples]}, handle, indent=2
            )
        logger.info("Validation report written to %s", path)

    def log(self) -> None:
        """Print the end-of-run table, plus each problem in full underneath."""
        if not self.samples:
            return

        if Console is not None and Table is not None:
            self._log_rich()
        else:
            self._log_plain()

        # The table truncates; the detail lines below it do not, so a failure
        # can be acted on without re-running.
        for entry in self.samples:
            for stage, name, message in entry.problems:
                logger.warning("%s — [%s] %s: %s", entry.sample, stage, name, message)

        n_fail, n_warn = len(self.failed), len(self.warned)
        if n_fail:
            logger.error(
                "%d of %d sample(s) FAILED validation: %s",
                n_fail, len(self.samples),
                ", ".join(s.sample for s in self.failed),
            )
        elif n_warn:
            logger.warning(
                "%d of %d sample(s) passed with warnings: %s",
                n_warn, len(self.samples),
                ", ".join(s.sample for s in self.warned),
            )
        else:
            logger.info("All %d sample(s) passed validation", len(self.samples))

    def _log_rich(self) -> None:
        assert Console is not None and Table is not None  # guarded by log()
        table = Table(title="Segmentation validation", title_justify="left")
        table.add_column("Sample", overflow="fold")
        table.add_column("Status")
        table.add_column("Checks", justify="right")
        table.add_column("Problems", overflow="fold")

        for entry in self.samples:
            label, style = _STATUS_STYLE[entry.status]
            total = sum(len(r.results) for r in entry.reports)
            problems = entry.problems
            detail = (
                "; ".join(f"{stage}/{name}" for stage, name, _ in problems[:3])
                + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else "")
            ) or "—"
            table.add_row(entry.sample, f"[{style}]{label}[/{style}]", str(total), detail)

        Console(stderr=True).print(table)

    def _log_plain(self) -> None:
        logger.info("%-28s %-6s %-7s %s", "SAMPLE", "STATUS", "CHECKS", "PROBLEMS")
        for entry in self.samples:
            total = sum(len(r.results) for r in entry.reports)
            problems = entry.problems
            detail = "; ".join(f"{stage}/{name}" for stage, name, _ in problems[:3]) or "-"
            logger.info("%-28s %-6s %-7d %s", entry.sample, entry.status, total, detail)
