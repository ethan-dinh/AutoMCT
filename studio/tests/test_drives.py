"""
Finding drives, their labels and their formats.

The Windows branches run here too, with Windows' answers about each drive
supplied by the test: drive letters, a mapped network share (known by its
share name, as a Mac mounts it), a UNC path to the same share, and the system
drive, whose paths belong to one computer only.

Run with:  python -m pytest studio/tests/test_drives.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio import drives  # noqa: E402


@pytest.fixture
def windows(monkeypatch):
    info = {"C:\\": ("Windows", "NTFS"), "E:\\": ("LabDrive", "exFAT"), "F:\\": ("Backup", "NTFS"),
            "Z:\\": ("Data", "NTFS")}
    shares = {"Z:": "Lab Data"}
    monkeypatch.setattr(drives, "WINDOWS", True)
    monkeypatch.setattr(drives, "MAC", False)
    monkeypatch.setattr(drives, "_win_volume_info", lambda root: info.get(root))
    monkeypatch.setattr(drives, "_win_share", lambda drive: shares.get(drive.rstrip("\\")))
    monkeypatch.setattr(drives, "_win_drives", lambda: ["C:", "E:", "F:", "Z:"])
    monkeypatch.setenv("SystemDrive", "C:")
    drives.forget()
    yield
    drives.forget()


def test_windows_drives_are_known_by_their_labels(windows):
    assert str(drives.drive_root("E:\\MicroCT-Studio\\samples")) == "E:\\"
    label, root = drives.volume_of("E:\\RAW\\C0004019.ISQ")
    assert label == "LabDrive" and str(root) == "E:\\"
    assert drives.volume_of("C:\\Users\\ethan\\scan.isq") is None
    assert drives.volume_of("Z:\\scans\\x.isq")[0] == "Lab Data"
    assert drives.volume_of("\\\\labserver\\Lab Data\\scans\\x.isq")[0] == "Lab Data"
    assert {k: str(v) for k, v in drives.mounted_volumes().items()} == {
        "LabDrive": "E:\\", "Backup": "F:\\", "Lab Data": "Z:\\"}
    assert str(drives.find_volume("labdrive")) == "E:\\"
    assert [label for label, _ in drives.browse_roots()] == [
        "Windows (C:)", "LabDrive (E:)", "Backup (F:)", "Lab Data (Z:)"]


def test_windows_formats_decide_what_both_systems_can_open(windows):
    assert drives.filesystem_of("E:\\x") == "exfat" and drives.sharing_note("E:\\x")["tone"] == "good"
    assert drives.sharing_note("F:\\x")["tone"] == "warning"
    assert drives.filesystem_of("Z:\\x") == "network"
    assert "own disk" in drives.sharing_note("C:\\Users\\x")["text"]


def test_mac_drives_and_formats(monkeypatch):
    monkeypatch.setattr(drives, "WINDOWS", False)
    monkeypatch.setattr(drives, "MAC", True)
    monkeypatch.setattr(drives, "_unix_mounts", lambda: [("/", "apfs"), ("/Volumes/LabDrive", "exfat"),
                                                          ("/Volumes/Old Drive", "apfs")])
    drives.forget()
    assert drives.volume_of("/Volumes/LabDrive/RAW/x.isq") == ("LabDrive", Path("/Volumes/LabDrive"))
    assert drives.volume_of("/Users/ethan/x.isq") is None
    assert drives.sharing_note("/Volumes/LabDrive/RAW")["tone"] == "good"
    note = drives.sharing_note("/Volumes/Old Drive/RAW")
    assert note["tone"] == "warning" and "PC cannot open it" in note["text"]
    drives.forget()
