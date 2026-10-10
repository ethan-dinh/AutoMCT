"""Shared setup for the runners: logging to stdout, and the spec argument."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def progress(fraction: float, label: str) -> None:
    print(f"@@progress {fraction:.3f} {label}", flush=True)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Headless: nothing a job runs may try to open a window.
    os.environ.setdefault("MPLBACKEND", "Agg")


def read_spec(description: str) -> dict:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--spec", required=True, help="JSON job specification")
    args = parser.parse_args()
    return json.loads(args.spec)
