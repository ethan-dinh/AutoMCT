"""
Serving the studio to your own devices over Tailscale.

``python -m studio --tailscale`` listens on this computer's Tailscale address
only, so a phone or iPad signed in to the same tailnet can use the studio
while nothing else on the room's Wi-Fi can reach it. Tailscale does the
authentication and encryption; the server's Host check still applies, with
this machine's tailnet names added to the names it answers to.

Also keeps the computer from idle-sleeping while the studio runs (macOS
``caffeinate``), since a sleeping Mac drops off the tailnet.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The CLI is on PATH for the standalone and Homebrew installs; the Mac App
# Store app keeps it inside the bundle unless "Install CLI" was chosen.
CANDIDATES = (
    "/usr/local/bin/tailscale",
    "/opt/homebrew/bin/tailscale",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
)


class TailscaleError(RuntimeError):
    """Tailscale is missing, stopped or signed out: the message says which and what to do."""


@dataclass
class Tailnet:
    ip: str                                     # 100.x.y.z, what the server binds to
    names: list[str] = field(default_factory=list)  # MagicDNS names, short and full

    def urls(self, port: int) -> list[str]:
        """The address first (always works), then the names (need MagicDNS on)."""
        return [f"http://{self.ip}:{port}/"] + [f"http://{n}:{port}/" for n in self.names]


def find_cli() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    return next((c for c in CANDIDATES if Path(c).is_file()), None)


def parse_status(status: dict) -> Tailnet:
    """This machine's tailnet address and names, from ``tailscale status --json``."""
    state = status.get("BackendState")
    if state != "Running":
        hint = {"Stopped": "Open the Tailscale app and switch it on.",
                "NeedsLogin": "Open the Tailscale app and sign in.",
                "NoState": "Open the Tailscale app and sign in."}.get(state, "Open the Tailscale app.")
        raise TailscaleError(f"Tailscale is not connected (state: {state}). {hint}")
    me = status.get("Self") or {}
    ipv4 = [ip for ip in me.get("TailscaleIPs") or [] if "." in ip]
    if not ipv4:
        raise TailscaleError("Tailscale is connected but this computer has no IPv4 tailnet address.")
    names = []
    full = (me.get("DNSName") or "").rstrip(".").lower()
    if full:
        names += [full.split(".")[0], full]
    return Tailnet(ip=ipv4[0], names=list(dict.fromkeys(names)))


def tailnet() -> Tailnet:
    cli = find_cli()
    if cli is None:
        raise TailscaleError("The tailscale command was not found. In the Tailscale menu-bar app, "
                             "choose Settings → Install CLI, or install Tailscale from tailscale.com.")
    try:
        out = subprocess.run([cli, "status", "--json"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TailscaleError(f"Could not run {cli}: {exc}") from exc
    try:
        status = json.loads(out.stdout)
    except json.JSONDecodeError:
        raise TailscaleError((out.stderr or out.stdout).strip() or "tailscale status gave no answer") from None
    return parse_status(status)


def stay_awake() -> subprocess.Popen | None:
    """Hold off idle and system sleep until this process exits (macOS; elsewhere a no-op).

    Closing a MacBook's lid still sleeps it unless it is on power with an
    external display; remote.md says what to do about that.
    """
    if sys.platform != "darwin" or not shutil.which("caffeinate"):
        return None
    return subprocess.Popen(["caffeinate", "-i", "-s", "-m", "-w", str(os.getpid())])
