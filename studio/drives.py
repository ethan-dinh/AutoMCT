"""
Drives, their labels and their filesystems, on macOS, Windows and Linux.

A workspace kept on an external drive is reached at /Volumes/LabDrive on a
Mac and at E:\\ on a PC, and the letter can change from one plug-in to the
next. A path to anything on that drive is only meaningful relative to
something that travels with it: the workspace folder, or the drive's label.
This module finds the drive (and label) a path is on, finds where a labelled
drive is mounted on this computer, and names a drive's filesystem, so the
studio can say when a format is one only one of the two systems can write.

Everything here degrades to "don't know" (None) rather than raising: a drive
that cannot be queried is treated like the computer's own disk.
"""

from __future__ import annotations

import ntpath
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WINDOWS = os.name == "nt"
MAC = sys.platform == "darwin"

_TTL_S = 3.0
_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, fn):
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < _TTL_S:
        return hit[1]
    value = fn()
    _cache[key] = (now, value)
    return value


def forget() -> None:
    """Drop what is known about connected drives, after one is plugged in or out."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _win_volume_info(root: str) -> tuple[str, str] | None:
    """(label, filesystem) of a Windows drive root such as 'E:\\', or None."""
    try:
        import ctypes  # noqa: PLC0415
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetErrorMode(1)  # SEM_FAILCRITICALERRORS: no "insert a disk" dialog
        label = ctypes.create_unicode_buffer(261)
        fs = ctypes.create_unicode_buffer(261)
        ok = kernel32.GetVolumeInformationW(ctypes.c_wchar_p(root), label, 261, None, None, None, fs, 261)
    except (AttributeError, OSError, ValueError):
        return None
    return (label.value, fs.value) if ok else None


def _win_share(drive: str) -> str | None:
    """The share behind a mapped network drive ('Z:' -> 'Lab Data'), or None."""
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        rc = ctypes.windll.mpr.WNetGetConnectionW(drive, buf, ctypes.byref(size))  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return None
    parts = buf.value.strip("\\").split("\\") if rc == 0 and buf.value.startswith("\\\\") else []
    return parts[1] if len(parts) >= 2 else None


def _win_drives() -> list[str]:
    try:
        import ctypes  # noqa: PLC0415
        mask = ctypes.windll.kernel32.GetLogicalDrives()  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return []
    return [f"{chr(65 + i)}:" for i in range(26) if mask & (1 << i)]


def _system_drive() -> str:
    return os.environ.get("SystemDrive", "C:").upper()


# ---------------------------------------------------------------------------
# Where a path lives
# ---------------------------------------------------------------------------

def drive_root(path: str | Path) -> Path:
    """The root of the drive a path is on: E:\\, /Volumes/LabDrive, or / for the system disk."""
    if WINDOWS:
        # ntpath is os.path on Windows; naming it keeps this testable anywhere.
        text = ntpath.abspath(str(path))
        drive = ntpath.splitdrive(text)[0]
        return Path(drive + "\\") if drive else Path(text[:3])
    p = Path(os.path.abspath(path))
    if MAC and len(p.parts) >= 3 and p.parts[1] == "Volumes":
        return Path(*p.parts[:3])
    q = p
    while q != q.parent and not os.path.ismount(q):
        q = q.parent
    return q


def volume_of(path: str | Path) -> tuple[str, Path] | None:
    """(label, root) of the external or network drive a path is on.

    None for the computer's own disk, whose paths mean nothing on another
    computer, and for a drive without a label.
    """
    root = drive_root(path)
    if WINDOWS:
        text = str(root).rstrip("\\")
        if text.startswith("\\\\"):
            parts = text.strip("\\").split("\\")
            return (parts[1], root) if len(parts) >= 2 else None
        if text.upper() == _system_drive():
            return None
        label = _win_share(text) or (_win_volume_info(text + "\\") or ("", ""))[0]
        return (label, root) if label else None
    if MAC:
        return (root.name, root) if len(root.parts) == 3 and root.parts[1] == "Volumes" else None
    return None if str(root) == "/" else (root.name, root)


def mounted_volumes() -> dict[str, Path]:
    """Every external or network drive connected to this computer, label -> root."""
    return _cached("volumes", _scan_volumes)


def _scan_volumes() -> dict[str, Path]:
    found: dict[str, Path] = {}
    if WINDOWS:
        for drive in _win_drives():
            if drive.upper() == _system_drive():
                continue
            label = _win_share(drive) or (_win_volume_info(drive + "\\") or ("", ""))[0]
            if label:
                found.setdefault(label, Path(drive + "\\"))
        return found
    if MAC:
        bases = ["/Volumes"]
    else:
        user = os.environ.get("USER", "")
        bases = [f"/media/{user}", f"/run/media/{user}", "/media", "/mnt"]
    for base in bases:
        try:
            entries = list(os.scandir(base))
        except OSError:
            continue
        for entry in entries:
            # macOS mounts system volumes here too (Recovery, Time Machine snapshots).
            if entry.name.startswith((".", "com.apple.")) or (MAC and entry.name == "Recovery"):
                continue
            try:
                if entry.is_dir():
                    found.setdefault(entry.name, Path(entry.path))
            except OSError:
                continue
    return found


def browse_roots() -> list[tuple[str, Path]]:
    """Places to start browsing from: every drive, named the way this system names it."""
    if WINDOWS:
        out = []
        for drive in _win_drives():
            label = _win_share(drive) or (_win_volume_info(drive + "\\") or ("", ""))[0]
            out.append((f"{label} ({drive})" if label else drive, Path(drive + "\\")))
        return out
    roots = sorted(mounted_volumes().items(), key=lambda kv: kv[0].casefold())
    return roots if MAC else [("Computer", Path("/")), *roots]


def find_volume(label: str) -> Path | None:
    """Where the drive with this label is mounted on this computer, if it is connected."""
    table = mounted_volumes()
    if label in table:
        return table[label]
    for name, root in table.items():
        # Case can differ between systems, and macOS numbers a second drive
        # of the same name ("LabDrive 1").
        if name.casefold() == label.casefold() or re.fullmatch(re.escape(label) + r" \d+", name, re.IGNORECASE):
            return root
    return None


# ---------------------------------------------------------------------------
# Filesystems
# ---------------------------------------------------------------------------

def filesystem_of(path: str | Path) -> str | None:
    """The filesystem a path is on, lower case (exfat, apfs, ntfs, msdos, smbfs...), or None."""
    root = drive_root(path)
    if WINDOWS:
        text = str(root).rstrip("\\")
        if text.startswith("\\\\") or _win_share(text):
            return "network"
        info = _win_volume_info(text + "\\")
        return info[1].lower() if info and info[1] else None
    target = os.path.abspath(path)
    best: tuple[str, str] | None = None
    for point, fs in _cached("mounts", _unix_mounts):
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and (best is None or len(point) > len(best[0])):
            best = (point, fs)
    return best[1].lower() if best else None


def _unix_mounts() -> list[tuple[str, str]]:
    if not MAC:
        try:
            with open("/proc/mounts", encoding="utf-8") as handle:
                rows = [line.split() for line in handle]
        except OSError:
            return []
        return [(r[1].replace("\\040", " "), r[2]) for r in rows if len(r) >= 3]
    try:
        text = subprocess.run(["mount"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in text.splitlines():
        # /dev/disk5s1 on /Volumes/LabDrive (exfat, local, nodev, nosuid, noowners)
        m = re.match(r"^.+? on (.+?) \(([^,)]+)", line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


_NETWORK = {"network", "smbfs", "cifs", "nfs", "nfs4", "afpfs", "webdav", "smb2", "9p"}


def sharing_note(path: str | Path) -> dict:
    """Whether a folder can be opened from both a Mac and a PC, as {tone, fs, text}."""
    fs = filesystem_of(path)
    note = {"fs": fs, "tone": "info", "text": ""}
    if volume_of(path) is None and fs not in _NETWORK:
        note["text"] = ("On this computer's own disk, so it stays with this computer. To use it from "
                        "both a Mac and a PC, keep it on an external drive formatted as exFAT.")
        return note
    if fs == "exfat":
        note.update(tone="good", text="exFAT: both a Mac and a PC can read and write it.")
    elif fs in ("apfs", "hfs", "hfsplus"):
        note.update(tone="warning", text="This drive is formatted for Macs only (APFS or Mac OS Extended); "
                                         "a PC cannot open it. Use an exFAT drive to share it with a PC.")
    elif fs in ("ntfs", "ntfs3", "fuseblk"):
        note.update(tone="warning", text="NTFS: a Mac can read it but cannot write to it without extra "
                                         "software. exFAT works on both.")
    elif fs in ("msdos", "vfat", "fat32", "fat"):
        note.update(tone="warning", text="FAT32 cannot hold files over 4 GB, and converted scans are often "
                                         "bigger. exFAT works on both without that limit.")
    elif fs in _NETWORK:
        note["text"] = "A network folder: any computer that can reach it can open the workspace."
    elif fs in ("ext4", "ext3", "btrfs", "xfs"):
        note.update(tone="warning", text="A Linux filesystem: neither a Mac nor a PC can write to it.")
    return note
