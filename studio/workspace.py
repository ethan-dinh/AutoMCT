"""
The studio's library of specimens, and the folder layout that organizes them.

A workspace is a plain directory, so everything in it is readable without the
studio running:

    <workspace>/
        studio.json                 settings, the colour each group was given, the library's folders,
                                    the Results page's chart colours and titles
        samples/<name>/
            sample.json             metadata, where each artifact lives, step history
            raw/                    uploaded or copied raw data
            converted/              <name>.nrrd from the converter
            segmentation/<name>/    the segmentation pipeline's own layout
            cmpr/                   <name>_resliced_volume.nrrd + transform
            quant/incisor/          enamel density profile
            quant/molar/            molar enamel quantification
            landmarks/              picked marks and their mapped positions
            preview/                downsampled copies the viewer streams
        jobs/                       one JSON record and one log per job
        uploads/                    files mid-upload, before they join a sample
        trash/                      archived samples

Artifacts are recorded by path rather than copied, so a sample can point at
data that stays where it is -- the Z drive's SEG DATA folder, say -- and still
be processed and viewed here. Paths inside the sample folder are stored
relative to it, so a workspace can be moved as a whole.

A scan linked in place keeps its two large files, its conversion and its
reoriented mandible, on its own drive rather than in converted/ and
segmentation/ here. They are written beside it, and the record keeps that
folder as "storage":

    <scan's folder>/MicroCT-Studio/<name>/
        <name>.nrrd                     the conversion (and its _meta.json)
        <name>_reoriented_volume.nrrd   the reoriented mandible
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from . import drives

# ---------------------------------------------------------------------------
# JSON on disk
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def write_json(path: Path, payload: Any) -> None:
    """Write atomically: a reader never sees half a file, even mid-crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Names and groups
# ---------------------------------------------------------------------------

# The metadata a specimen carries, in form order. `age`, `line`, `genotype`,
# `sex` and `replicate` compose the name, in the lab's convention
# {Age}_{BaseMouse}_{Gene}_{Sex}_{N} -- e.g. P52_Odam-Cre_WT_F_1. The sex is
# written F or M, and left out while it is not recorded (P52_Odam-Cre_WT_1).
META_FIELDS = (
    "age", "line", "genotype", "sex", "replicate", "side",
    "study", "scan_date", "scanner", "voxel_um", "notes",
)

_AGE = re.compile(r"^(P|E)\d+(\.\d+)?$|^\d+(\.\d+)?(d|wk|w|mo|m|y)$", re.IGNORECASE)
# How a name writes the sex, from every spelling one might use for it.
_SEX_TOKENS = {"F": "F", "M": "M", "FEMALE": "F", "MALE": "M"}


def sex_token(value: Any) -> str:
    """F or M for a sex written any of the usual ways, else ""."""
    return _SEX_TOKENS.get(str(value or "").strip().upper(), "")


def name_token(value: Any) -> str:
    """One underscore-free token of a sample name. Underscores separate tokens."""
    text = re.sub(r"\s+", "-", str(value).strip())
    return re.sub(r"[^A-Za-z0-9.+-]", "", text).strip("-.")


def compose_name(meta: dict) -> str:
    parts = [meta.get("age"), meta.get("line"), meta.get("genotype"), sex_token(meta.get("sex")), meta.get("replicate")]
    return "_".join(t for t in (name_token(p) for p in parts if p not in (None, "")) if t)


def parse_name(name: str) -> dict:
    """Recover metadata from a name that follows {Age}_{BaseMouse}_{Gene}_{Sex}_{N}.

    Only names that open with an age are parsed; anything else (Odam-CRE-A,
    Frag_4) returns nothing rather than a guess that would file the specimen
    into the wrong group. The sex may be missing (the names from before it
    was part of them), spelled out (Female), or written after the number.
    """
    tokens = [t for t in name.split("_") if t]
    if not tokens or not _AGE.match(tokens[0]):
        return {}
    meta: dict[str, Any] = {"age": tokens.pop(0)}

    def take_sex() -> None:
        if tokens and "sex" not in meta and sex_token(tokens[-1]):
            meta["sex"] = sex_token(tokens.pop())

    take_sex()                     # P52_Odam-Cre_WT_1_F
    if tokens and tokens[-1].isdigit():
        meta["replicate"] = int(tokens.pop())
    take_sex()                     # P52_Odam-Cre_WT_F_1, the convention
    if len(tokens) >= 2:
        meta["line"] = tokens[0]
        meta["genotype"] = " ".join(tokens[1:]).replace("-", " ")
    elif tokens:
        meta["genotype"] = tokens[0].replace("-", " ")
    return meta


def group_of(meta: dict) -> str:
    """The analysis unit: genotype crossed with age, as the R report defines it."""
    parts = [str(meta.get(k) or "").strip() for k in ("genotype", "age")]
    label = " ".join(p for p in parts if p)
    return label or "Ungrouped"


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
# Names Windows will not give a file or folder, whatever the extension.
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def valid_name(name: str) -> bool:
    """A name that makes a folder and files on macOS and Windows alike."""
    return (bool(_SAFE_NAME.match(name)) and len(name) <= 120 and not name.endswith(".")
            and name.split(".")[0].upper() not in _RESERVED)


def safe_name(text: str) -> str:
    """A usable sample name made from any text, or "" if nothing usable is left.

    Runs of characters a name cannot hold become one hyphen, so a folder called
    "Odam CRE (A)" is filed as Odam-CRE-A.
    """
    name = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(text or "").strip()).strip("-._+")[:120].rstrip("-._+")
    if name and name.split(".")[0].upper() in _RESERVED:
        name = f"sample-{name}"
    return name if name and valid_name(name) else ""


# What follows a group's stem in one of its names: the number, perhaps after the sex.
_REPLICATE_TAIL = re.compile(r"^(?:(?:F|M|FEMALE|MALE)_)?(\d+)$", re.IGNORECASE)


def composes_name(meta: dict) -> bool:
    """Whether the details are enough for a name in the lab's convention: an age and a genotype.

    The line may be left out (P82_WT_1), but not the genotype: P52_Odam-Cre_1
    would read back with the line taken for the genotype.
    """
    return bool(name_token(meta.get("age") or "")) and bool(name_token(meta.get("genotype") or ""))


# ---------------------------------------------------------------------------
# Library folders
# ---------------------------------------------------------------------------

# Folders sort the library's specimens -- by study, by batch, by who scanned
# them -- and nest, written with a slash: "Odam study/Batch 2". They are the
# library's and not the disk's: a specimen's own folder stays samples/<name>,
# so filing it elsewhere renames nothing and every recorded path still leads
# where it did. A specimen's record keeps the folder it is in ("" or no key:
# the top of the library), and studio.json keeps the folders themselves, so
# one stays after it is emptied until it is removed.

FOLDER_PART_MAX = 80
FOLDER_DEPTH_MAX = 8
_FOLDER_BAD = re.compile(r"[\x00-\x1f\x7f\\]")


class FolderExists(ValueError):
    """A folder is being made or renamed where one of that name already is."""


class ProjectExists(ValueError):
    """A Results page project is being made or renamed where one of that name already is."""


# What a Results page project keeps besides its name: the specimens it covers,
# the details it compares them by (the main one first), and its filters: the
# values its chips hide ("sex\u0001M") and the specimens unticked for now.
PROJECT_LISTS = ("names", "by", "hidden", "left")
PROJECT_NAME_MAX = 80


def folder_path(text: Any) -> str:
    """A folder as it is stored: each part trimmed, spaces collapsed, joined by "/".

    "" is the top of the library. Raises ValueError for a part no folder can
    have: "." or "..", a backslash or a control character, or too long a name.
    """
    parts = [" ".join(p.split()) for p in str(text or "").split("/")]
    parts = [p for p in parts if p]
    for part in parts:
        if part in (".", "..") or _FOLDER_BAD.search(part):
            raise ValueError(f"{part!r} is not a usable folder name")
        if len(part) > FOLDER_PART_MAX:
            raise ValueError(f"A folder name can be at most {FOLDER_PART_MAX} characters")
    if len(parts) > FOLDER_DEPTH_MAX:
        raise ValueError(f"Folders nest at most {FOLDER_DEPTH_MAX} deep")
    return "/".join(parts)


def folder_ancestors(folder: str) -> list[str]:
    """A folder and every folder it is in, outermost first: a/b/c -> a, a/b, a/b/c."""
    parts = folder.split("/") if folder else []
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def in_folder(folder: str, parent: str) -> bool:
    """Whether a folder is ``parent`` or anywhere inside it."""
    return folder == parent or folder.startswith(parent + "/")


# ---------------------------------------------------------------------------
# Recognizing files
# ---------------------------------------------------------------------------

VOLUME_EXTS = (".nrrd", ".tif", ".tiff")

# <sample><suffix><ext> -> artifact key. Longest suffixes first, since
# "_molar_enamel" is a prefix of "_molar_enamel_profile".
_SUFFIXES: tuple[tuple[str, dict[str, str]], ...] = (
    ("_enamel_density_profile", {".csv": "incisor_profile", ".png": "incisor_plot_density"}),
    ("_enamel_density_ratio", {".png": "incisor_plot_ratio"}),
    ("_enamel_density_run", {".json": "incisor_run"}),
    ("_molar_enamel_profile", {".csv": "molar_profile"}),
    ("_molar_enamel_teeth", {".csv": "molar_teeth"}),
    ("_molar_enamel", {".json": "molar_quant", ".png": "molar_plot"}),
    ("_thickness_map", {".html": "thickness_map", ".png": "thickness_plot"}),
    ("_enamel_mask", {e: "enamel_mask" for e in VOLUME_EXTS}),
    ("_tooth_labels", {e: "tooth_labels" for e in VOLUME_EXTS}),
    ("_reoriented_volume", {e: "reoriented" for e in VOLUME_EXTS}),
    ("_incisor_volume", {e: "incisor_volume" for e in VOLUME_EXTS}),
    ("_bone_volume", {e: "bone_volume" for e in VOLUME_EXTS}),
    ("_molar_volume", {e: "molar_volume" for e in VOLUME_EXTS}),
    ("_incisor_mask", {e: "incisor_mask" for e in VOLUME_EXTS}),
    ("_bone_mask", {e: "bone_mask" for e in VOLUME_EXTS}),
    ("_molar_mask", {e: "molar_mask" for e in VOLUME_EXTS}),
    ("_resliced_volume", {e: "cmpr_volume" for e in VOLUME_EXTS}),
    ("_cmpr_transform", {".npz": "cmpr_transform"}),
    ("_landmarks", {".csv": "landmarks", ".json": "landmark_report"}),
    ("_marks", {".csv": "landmark_marks"}),
    ("_voxel_size", {".json": "voxel_size"}),
    ("_prefiltered", {".nrrd": "prefiltered"}),
)

ARTIFACT_LABELS = {
    "raw": "Raw scan",
    "converted": "Converted volume",
    "prefiltered": "Unfiltered converted volume",
    "convert_meta": "Conversion metadata",
    "reoriented": "Reoriented scan",
    "incisor_volume": "Incisor volume",
    "bone_volume": "Bone volume",
    "molar_volume": "Molar volume",
    "incisor_mask": "Incisor mask",
    "bone_mask": "Bone mask",
    "molar_mask": "Molar mask",
    "voxel_size": "Voxel size record",
    "validation": "Validation report",
    "cmpr_volume": "Straightened incisor (CMPR)",
    "cmpr_transform": "CMPR transform",
    "incisor_profile": "Enamel density profile",
    "incisor_run": "Enamel density run record",
    "incisor_plot_density": "Enamel density figure",
    "incisor_plot_ratio": "Enamel/dentin ratio figure",
    "landmark_marks": "Picked landmarks",
    "landmarks": "Mapped landmarks",
    "landmark_report": "Landmark detection",
    "molar_quant": "Molar enamel summary",
    "molar_teeth": "Per-tooth table",
    "molar_profile": "Along-jaw enamel profile",
    "enamel_mask": "Molar enamel mask",
    "tooth_labels": "Molar teeth (1 = M1, 2 = M2, 3 = M3)",
    "molar_plot": "Molar enamel figure",
    "thickness_map": "Enamel thickness map (3-D)",
    "thickness_plot": "Enamel thickness map figure",
    "ml_incisor_mask": "ML incisor mask",
    "ml_bone_mask": "ML bone mask",
    "ml_molar_mask": "ML molar mask",
}


def classify_file(path: Path) -> tuple[str | None, str | None]:
    """(sample name, artifact key) for a file the pipeline wrote, else (None, None).

    A name with no sample prefix -- the legacy ``volumes/incisor_volume.tif``
    layout -- returns ("", key): the caller supplies the name from the folder.
    """
    name = path.name
    if name.startswith("._"):
        return None, None
    ext = path.suffix.lower()
    stem = name[: -len(path.suffix)] if path.suffix else name
    for suffix, by_ext in _SUFFIXES:
        key = by_ext.get(ext)
        if key is None:
            continue
        if stem.endswith(suffix):
            return stem[: -len(suffix)], key
        if stem == suffix.lstrip("_"):
            return "", key
    return None, None


def _has_dicom_magic(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def _first_files(directory: Path, limit: int = 400) -> list[Path]:
    out = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_file():
                    out.append(Path(entry.path))
                    if len(out) >= limit:
                        break
    except OSError:
        pass
    return out


def detect_kind(path: Path) -> str:
    """What a path holds, as far as importing is concerned.

    isq / rsq / dicom / bmp      raw scanner output
    volume                       an already-converted .nrrd/.tif scan
    segmentation                 a sample folder the segmentation pipeline wrote
    collection                   a folder of several samples
    unknown
    """
    if path.is_file():
        ext = path.suffix.lower()
        if ext == ".isq":
            return "isq"
        if ext == ".rsq":
            return "rsq"
        if ext in VOLUME_EXTS:
            return "volume"
        if _has_dicom_magic(path):
            return "dicom"
        return "unknown"
    if not path.is_dir():
        return "unknown"
    if any(path.glob("volumes/*incisor_volume.*")) or any(path.glob("*_reoriented_volume.*")):
        return "segmentation"
    files = _first_files(path)
    bmps = [f for f in files if f.suffix.lower() == ".bmp" and not f.name.lower().endswith("spr.bmp")]
    if len(bmps) >= 10:
        return "bmp"
    probe = [f for f in files if f.suffix.lower() in ("", ".dcm", ".ima")][:5]
    if probe and any(_has_dicom_magic(f) for f in probe):
        return "dicom"
    return "collection"


def source_format(kind: str | None, path: str | Path | None) -> str | None:
    """A raw source's file format: isq, rsq, dicom, bmp, nrrd or tiff.

    A "volume" is one .nrrd or .tif file, and which of the two it is decides
    whether it is converted before segmentation, so the kind alone is not
    enough. Other kinds are their own format.
    """
    if kind != "volume":
        return kind
    suffix = Path(str(path)).suffix.lower() if path else ""
    return {".nrrd": "nrrd", ".tif": "tiff", ".tiff": "tiff"}.get(suffix)


# Large files of a scan linked in place (its conversion, its reoriented
# mandible) are written beside the scan, in this folder, one subfolder per
# specimen, so they stay on the scan's own drive rather than filling the
# workspace's.
STORAGE_FOLDER = "MicroCT-Studio"
_STORED_KINDS = ("isq", "dicom", "bmp", "volume")


def storage_root(scan: str | Path) -> Path:
    """The MicroCT-Studio folder beside a scan, on the scan's own drive."""
    scan = Path(scan)
    base = scan.parent
    # A scan folder at the root of a drive has /Volumes, on the internal disk,
    # as its parent; the studio's folder then goes inside the drive instead.
    try:
        if scan.is_dir() and os.stat(base).st_dev != os.stat(scan).st_dev:
            base = scan
    except OSError:
        pass
    return base / STORAGE_FOLDER


def synced_service(path: str | Path) -> str | None:
    """The cloud service a folder syncs to (Box, Dropbox, iCloud Drive...), if any."""
    text = str(Path(path).expanduser())
    cloud = str(Path.home() / "Library" / "CloudStorage") + os.sep
    if text.startswith(cloud):
        provider = text[len(cloud):].split(os.sep, 1)[0]
        return provider.split("-", 1)[0] or "a cloud service"
    if text.startswith(str(Path.home() / "Library" / "Mobile Documents")):
        return "iCloud Drive"
    if f"{os.sep}Dropbox{os.sep}" in text + os.sep:
        return "Dropbox"
    return None


# ---------------------------------------------------------------------------
# Paths that travel with the drive
# ---------------------------------------------------------------------------

# A path outside a sample's own folder is stored so the record still leads to
# the file when the workspace's drive is plugged into another computer, where
# it is mounted somewhere else (/Volumes/LabDrive on a Mac, E:\ or F:\ on a PC):
#
#     @ws/<path>             inside the workspace folder
#     @vol/<label>/<path>    on an external or network drive, found by its label
#     /an/absolute/path      on this computer's own disk, which no other
#                            computer has anyway
#
# Paths inside a sample's folder are stored relative to it, with forward
# slashes, which every system reads.

_FOREIGN_ABS = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def portable_path(path: str | Path, ws_root: Path) -> str:
    real = Path(os.path.realpath(path))
    root = Path(ws_root)
    if real == root or root in real.parents:
        rel = real.relative_to(root).as_posix()
        return "@ws/" + ("" if rel == "." else rel)
    vol = drives.volume_of(real)
    if vol is not None:
        label, vroot = vol
        rel = real.relative_to(vroot).as_posix()
        return f"@vol/{label}/" + ("" if rel == "." else rel)
    return str(real)


def resolve_portable(stored: str, ws_root: Path) -> Path | None:
    """Where a stored path is on this computer; None when its drive is not connected."""
    if stored.startswith("@ws/"):
        return Path(ws_root, *[p for p in stored[4:].split("/") if p not in ("", ".")])
    if stored.startswith("@vol/"):
        label, _, rel = stored[5:].partition("/")
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        root = drives.find_volume(label)
        if root is not None:
            return Path(root, *parts)
        # A renamed drive, or the workspace's own drive under another name:
        # the file is at the same place on the drive that holds the workspace.
        candidate = Path(drives.drive_root(ws_root), *parts)
        return candidate if parts and candidate.exists() else None
    return Path(stored)


APP_MARK = "MicroCT Studio"
_SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information", "Library", "Applications", "System",
              "Windows", "Program Files", "Program Files (x86)", "ProgramData", "node_modules"}


def is_workspace(path: str | Path) -> bool:
    data = read_json(Path(path) / "studio.json")
    return isinstance(data, dict) and (data.get("app") == APP_MARK or "lanes" in data)


def workspace_summary(path: str | Path) -> dict:
    """What the workspace chooser shows about a workspace folder."""
    path = Path(path)
    data = read_json(path / "studio.json") or {}
    try:
        count = sum(1 for p in (path / "samples").iterdir() if (p / "sample.json").exists())
    except OSError:
        count = 0
    vol = drives.volume_of(path)
    return {"path": str(path), "name": path.name, "id": data.get("id"), "samples": count,
            "drive": vol[0] if vol else None, "sharing": drives.sharing_note(path)}


def find_workspaces(max_depth: int = 2, budget_s: float = 1.5) -> list[Path]:
    """Workspaces near the top of every connected drive and of the home folder.

    Each place gets a time budget, so a slow network drive cannot hold up the
    chooser; a workspace deeper down is opened with "Open another folder".
    """
    tops = [r for r in drives.mounted_volumes().values() if os.path.realpath(r) != os.sep]
    found: dict[str, Path] = {}
    for top in [*tops, Path.home()]:
        deadline = time.monotonic() + budget_s
        level = [top]
        for depth in range(max_depth + 1):
            deeper = []
            for folder in level:
                if time.monotonic() > deadline:
                    break
                if is_workspace(folder):
                    found.setdefault(os.path.realpath(folder), folder)
                    continue
                if depth == max_depth:
                    continue
                try:
                    with os.scandir(folder) as entries:
                        for entry in entries:
                            if entry.name.startswith((".", "$")) or entry.name in _SKIP_DIRS:
                                continue
                            try:
                                if entry.is_dir(follow_symlinks=False):
                                    deeper.append(Path(entry.path))
                            except OSError:
                                continue
                except OSError:
                    continue
            level = deeper
    return list(found.values())


def directory_size(path: Path, limit_files: int = 200_000) -> tuple[int, int]:
    """(bytes, files) under a path, stopping early on enormous trees."""
    if path.is_file():
        return path.stat().st_size, 1
    total = count = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
            count += 1
            if count >= limit_files:
                return total, count
    return total, count


def artifacts_in(folder: Path, sample_name: str, max_depth: int = 3) -> dict[str, str]:
    """Every pipeline output under a sample folder, keyed by artifact."""
    found: dict[str, str] = {}
    base_depth = len(folder.parts)
    for root, dirs, files in os.walk(folder):
        depth = len(Path(root).parts) - base_depth
        dirs[:] = [d for d in dirs if not d.startswith(".") and depth < max_depth]
        for name in files:
            path = Path(root) / name
            owner, key = classify_file(path)
            if key is None or (owner and owner != sample_name):
                continue
            # A prefixed file beats an unprefixed legacy one; NRRD beats TIFF.
            current = found.get(key)
            if current is None or (owner and path.suffix == ".nrrd"):
                found[key] = str(path)
        if "validation_report.json" in files and "validation" not in found:
            found["validation"] = str(Path(root) / "validation_report.json")
    return found


def _isq_files(folder: Path) -> list[Path]:
    """The .isq scans directly in a folder, whatever the case of their extension.

    Matched by suffix rather than globbed twice, which on a case-insensitive
    filesystem lists every scan once per spelling.
    """
    try:
        return sorted(p for p in folder.iterdir()
                      if p.suffix.lower() == ".isq" and not p.name.startswith(".") and p.is_file())
    except OSError:
        return []


def scan_paths(paths: Iterable[str]) -> list[dict]:
    """Find every specimen under the given folders, for a bulk import.

    Folders of sample folders (SEG DATA), folders of loose per-sample outputs
    (CMPR, RESULTS), raw scans (.isq files, DICOM/BMP folders) -- all are
    matched up by sample name, so pointing at the segmentation root and the
    CMPR and results folders in one go yields one candidate per specimen with
    everything already attached.

    Each candidate lists the folders it was found under, nearest first, since
    a folder called P52 or WT says something about every scan inside it.
    """
    candidates: dict[str, dict] = {}

    def entry(name: str, folders: list[str]) -> dict:
        """The candidate for a name, noting the folders it was found under, nearest first."""
        if name not in candidates:
            candidates[name] = {
                "name": name, "source": None, "artifacts": {},
                "meta": parse_name(name), "folders": [],
            }
        cand = candidates[name]
        cand["folders"] += [f for f in folders if f and f not in cand["folders"]]
        return cand

    for raw in paths:
        root = Path(raw).expanduser()
        kind = detect_kind(root)
        # A path that is itself one specimen was found in its parent folder.
        here = [root.parent.name]
        if kind in ("isq", "rsq", "dicom", "bmp", "volume"):
            name = root.stem if root.is_file() else root.name
            owner, key = classify_file(root) if root.is_file() else (None, None)
            if key:
                entry(owner or root.parent.name, here)["artifacts"][key] = str(root)
            else:
                entry(name, here)["source"] = {"kind": kind, "path": str(root)}
            continue
        if kind == "segmentation":
            entry(root.name, here)["artifacts"].update(artifacts_in(root, root.name))
            continue
        if kind != "collection":
            continue
        try:
            children = sorted(p for p in root.iterdir() if not p.name.startswith((".", "_")))
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                # The studio's own large files beside the scans are outputs of
                # specimens already in a workspace, not specimens to import.
                if child.name in ("Compressed", "masks", "volumes", STORAGE_FOLDER):
                    continue
                sub = detect_kind(child)
                if sub == "segmentation":
                    entry(child.name, [root.name])["artifacts"].update(artifacts_in(child, child.name))
                elif sub in ("dicom", "bmp"):
                    entry(child.name, [root.name])["source"] = {"kind": sub, "path": str(child)}
                elif sub == "collection":
                    arts = artifacts_in(child, child.name, max_depth=2)
                    raw_scans = _isq_files(child)
                    if arts or len(raw_scans) == 1:
                        # One specimen's folder: its outputs and its scan.
                        cand = entry(child.name, [root.name])
                        cand["artifacts"].update(arts)
                        if raw_scans:
                            cand["source"] = {"kind": "isq", "path": str(raw_scans[0])}
                    else:
                        # A folder of several specimens' scans, one age or
                        # genotype to a folder, say: each scan is a specimen.
                        for scan in raw_scans:
                            entry(scan.stem, [child.name, root.name])["source"] = {"kind": "isq", "path": str(scan)}
                continue
            owner, key = classify_file(child)
            if key and owner:
                entry(owner, [root.name])["artifacts"].setdefault(key, str(child))
                continue
            ext = child.suffix.lower()
            if ext in (".isq", ".rsq"):
                entry(child.stem, [root.name])["source"] = {"kind": ext[1:], "path": str(child)}
            elif ext in VOLUME_EXTS:
                entry(child.stem, [root.name])["source"] = {"kind": "volume", "path": str(child)}

    for cand in candidates.values():
        if cand["source"] is None:
            arts = cand["artifacts"]
            origin = arts.get("reoriented") or arts.get("incisor_volume") or arts.get("cmpr_volume")
            cand["source"] = {"kind": "results", "path": str(Path(origin).parent) if origin else ""}
    return sorted(candidates.values(), key=lambda c: c["name"])


# ---------------------------------------------------------------------------
# The workspace
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS = {
    "lanes": {"heavy": 1, "preview": 1, "light": 2},
    "preview_budget_voxels": 24_000_000,
    "auto_preview": True,
    "group_colors": {},
    "step_defaults": {},
    "folders": [],
    "results_style": {},
    "results_projects": [],
}

# Order of the categorical palette slots groups are assigned to. A group keeps
# its slot for the life of the workspace, so its colour never changes when a
# filter hides others.
PALETTE_SLOTS = 8


class Workspace:
    """All reads and writes of sample records go through here, under one lock."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.samples_dir = self.root / "samples"
        self.jobs_dir = self.root / "jobs"
        self.uploads_dir = self.root / "uploads"
        self.trash_dir = self.root / "trash"
        self.cache_dir = self.root / ".cache"
        for d in (self.samples_dir, self.jobs_dir, self.uploads_dir, self.trash_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._cache: dict[str, tuple[float, dict]] = {}
        settings_path = self.root / "studio.json"
        if not settings_path.exists():
            write_json(settings_path, {"app": APP_MARK, "id": uuid.uuid4().hex, "created": now_iso(),
                                       **DEFAULT_SETTINGS})
        else:
            stored = read_json(settings_path)
            if isinstance(stored, dict) and not (stored.get("app") and stored.get("id")):
                stored.setdefault("app", APP_MARK)
                stored.setdefault("id", uuid.uuid4().hex)
                write_json(settings_path, stored)
        self._drop_stale_uploads()
        self.normalize_paths()

    @property
    def id(self) -> str | None:
        return (read_json(self.settings_path) or {}).get("id")

    def _drop_stale_uploads(self, max_age_s: float = 86_400) -> None:
        """Uploads abandoned mid-way (a closed tab, a cancelled drop) never join a sample."""
        cutoff = datetime.datetime.now().timestamp() - max_age_s
        for folder in self.uploads_dir.iterdir():
            try:
                if folder.is_dir() and folder.stat().st_mtime < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
            except OSError:
                continue

    # -- settings ---------------------------------------------------------

    @property
    def settings_path(self) -> Path:
        return self.root / "studio.json"

    def settings(self) -> dict:
        stored = read_json(self.settings_path, {}) or {}
        merged = {**DEFAULT_SETTINGS, **stored}
        merged["lanes"] = {**DEFAULT_SETTINGS["lanes"], **(stored.get("lanes") or {})}
        return merged

    def update_settings(self, patch: dict) -> dict:
        with self.lock:
            current = self.settings()
            for key, value in patch.items():
                if key in ("lanes", "step_defaults", "group_colors") and isinstance(value, dict):
                    current[key] = {**current.get(key, {}), **value}
                else:
                    current[key] = value
            write_json(self.settings_path, current)
            return current

    # -- Results page projects ---------------------------------------------
    # A project is a named scope of the Results page, kept in studio.json
    # (results_projects) so every browser that opens the workspace has it.

    def projects(self) -> list[dict]:
        return [dict(p) for p in self.settings().get("results_projects") or [] if isinstance(p, dict) and p.get("id")]

    @staticmethod
    def _project_fields(fields: dict, existing: list[dict], own: str | None = None) -> dict:
        out = {}
        if "name" in fields or own is None:
            name = " ".join(str(fields.get("name") or "").split())
            if not name:
                raise ValueError("Give the project a name")
            if len(name) > PROJECT_NAME_MAX:
                raise ValueError(f"A project name can be at most {PROJECT_NAME_MAX} characters")
            taken = next((p for p in existing if p.get("name", "").casefold() == name.casefold() and p["id"] != own), None)
            if taken:
                raise ProjectExists(f"There is already a project called {taken['name']}")
            out["name"] = name
        for key in PROJECT_LISTS:
            if key in fields:
                value = fields[key]
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise ValueError(f"A project's {key} must be a list of text")
                out[key] = list(dict.fromkeys(value))
        return out

    def create_project(self, fields: dict) -> dict:
        with self.lock:
            existing = self.projects()
            stamp = now_iso()
            project = {"id": uuid.uuid4().hex[:12], **{k: [] for k in PROJECT_LISTS},
                       **self._project_fields(fields, existing), "created": stamp, "updated": stamp}
            self.update_settings({"results_projects": [*existing, project]})
            return project

    def update_project(self, project_id: str, fields: dict) -> dict:
        with self.lock:
            existing = self.projects()
            i = next((i for i, p in enumerate(existing) if p["id"] == project_id), None)
            if i is None:
                raise KeyError(project_id)
            existing[i] = {**existing[i], **self._project_fields(fields, existing, project_id), "updated": now_iso()}
            self.update_settings({"results_projects": existing})
            return existing[i]

    def remove_project(self, project_id: str) -> None:
        with self.lock:
            existing = self.projects()
            kept = [p for p in existing if p["id"] != project_id]
            if len(kept) == len(existing):
                raise KeyError(project_id)
            self.update_settings({"results_projects": kept})

    def _rename_in_projects(self, old: str, new: str) -> bool:
        """A renamed specimen stays in its projects. Whether any held it."""
        projects = self.projects()
        if not any(old in p.get("names", []) or old in p.get("left", []) for p in projects):
            return False
        for p in projects:
            for key in ("names", "left"):
                p[key] = [new if n == old else n for n in p.get(key, [])]
        self.update_settings({"results_projects": projects})
        return True

    def group_slot(self, group: str) -> int:
        """The palette slot a group owns, assigning the next free one on first sight."""
        with self.lock:
            settings = self.settings()
            colors = settings.get("group_colors") or {}
            if group not in colors:
                used = set(colors.values())
                free = [i for i in range(PALETTE_SLOTS) if i not in used]
                colors[group] = free[0] if free else len(colors) % PALETTE_SLOTS
                self.update_settings({"group_colors": {group: colors[group]}})
            return int(colors[group])

    # -- sample records ---------------------------------------------------

    def sample_dir(self, name: str) -> Path:
        return self.samples_dir / name

    def _record_path(self, name: str) -> Path:
        return self.sample_dir(name) / "sample.json"

    def names(self) -> list[str]:
        return sorted(p.name for p in self.samples_dir.iterdir()
                      if (p / "sample.json").exists())

    def get(self, name: str) -> dict | None:
        path = self._record_path(name)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        cached = self._cache.get(name)
        if cached and cached[0] == mtime:
            return json.loads(json.dumps(cached[1]))
        record = read_json(path)
        if record is None:
            return None
        self._cache[name] = (mtime, record)
        return json.loads(json.dumps(record))

    def all(self) -> list[dict]:
        return [r for r in (self.get(n) for n in self.names()) if r]

    def save(self, record: dict, touch: bool = True) -> dict:
        if touch:
            record["updated"] = now_iso()
        record["group"] = group_of(record.get("meta") or {})
        with self.lock:
            write_json(self._record_path(record["name"]), record)
            self._cache.pop(record["name"], None)
        return record

    def update(self, name: str, fn, touch: bool = True) -> dict:
        """Read-modify-write a record under the lock. ``fn`` edits it in place.

        With ``touch`` False the record's "updated" time is left alone, for a
        change that is not to the specimen itself, such as the folder it is in.
        """
        with self.lock:
            record = self.get(name)
            if record is None:
                raise KeyError(name)
            fn(record)
            return self.save(record, touch)

    # -- folders ----------------------------------------------------------

    def folders(self) -> list[str]:
        """Every folder in the library, parents before what they hold.

        The folders studio.json keeps, and any a record names that it does not
        (a record edited by hand, say), with every folder they are inside.
        """
        found = set()
        for folder in list(self.settings().get("folders") or []) + [r.get("folder") or "" for r in self.all()]:
            try:
                found.update(folder_ancestors(folder_path(folder)))
            except ValueError:
                continue
        return sorted(found, key=lambda f: [p.casefold() for p in f.split("/")])

    def _keep_folders(self, folders: Iterable[str]) -> None:
        """Replace the folders studio.json keeps, each written once."""
        self.update_settings({"folders": sorted(set(folders), key=lambda f: [p.casefold() for p in f.split("/")])})

    def add_folder(self, folder: str) -> str:
        """Make a folder (and the folders it is in). Raises ValueError if it is already there."""
        with self.lock:
            folder = folder_path(folder)
            if not folder:
                raise ValueError("Give the folder a name")
            existing = self.folders()
            if folder.casefold() in {f.casefold() for f in existing}:
                raise FolderExists(f"There is already a folder called {folder}")
            folder = _as_written(folder, existing)
            self._keep_folders([*existing, *folder_ancestors(folder)])
            return folder

    def rename_folder(self, old: str, new: str) -> tuple[str, list[str]]:
        """Rename or move a folder, with everything in it: (its new path, the specimens that moved).

        Moving a folder into another is renaming it: "Batch 2" to "Odam study/Batch 2".
        """
        with self.lock:
            existing = self.folders()
            old, new = _as_written(folder_path(old), existing), folder_path(new)
            if not old or old not in existing:
                raise KeyError(old)
            if not new:
                raise ValueError("Give the folder a name")
            # Moved into a folder typed with another case, it goes into the one already there.
            parent, _, leaf = new.rpartition("/")
            new = "/".join(p for p in (_as_written(parent, existing), leaf) if p)
            if new == old:
                return new, []
            if in_folder(new, old):
                raise ValueError("A folder cannot go inside itself")
            # Only a change of case may land on a folder that is already there: it is the same one.
            if any(f.casefold() == new.casefold() and not in_folder(f, old) for f in existing):
                raise FolderExists(f"There is already a folder called {new}")

            def moved(folder: str) -> str:
                return new + folder[len(old):] if in_folder(folder, old) else folder

            self._keep_folders([*(moved(f) for f in existing), *folder_ancestors(new)])
            return new, self._refile(lambda folder: moved(folder) if in_folder(folder, old) else None)

    def remove_folder(self, folder: str) -> list[str]:
        """Remove a folder. What it held moves up into the folder it was in; no specimen is removed.

        Its folders keep their own contents: removing "Odam study" from
        "Odam study/Batch 2" leaves "Batch 2" at the top, as it was.
        Returns the specimens that moved.
        """
        with self.lock:
            existing = self.folders()
            folder = _as_written(folder_path(folder), existing)
            if not folder or folder not in existing:
                raise KeyError(folder)
            parent = folder.rpartition("/")[0]
            outside = [f for f in existing if not in_folder(f, folder)]

            def lifted(f: str) -> str:
                # A folder moving up beside one of the same name, but for case, joins it.
                return _as_written("/".join(p for p in (parent, f[len(folder) + 1:]) if p), outside)

            self._keep_folders([lifted(f) for f in existing if f != folder and in_folder(f, folder)] + outside)
            return self._refile(lambda f: lifted(f) if in_folder(f, folder) else None)

    def move_to_folder(self, names: Iterable[str], folder: str) -> str:
        """File specimens in a folder ("" for the top of the library), making it if it is new."""
        with self.lock:
            folder = folder_path(folder)
            names = list(names)
            for name in names:
                if self.get(name) is None:
                    raise KeyError(name)
            if folder:
                folder = self._claim_folder(folder)
            for name in names:
                self.update(name, lambda r: _file_in(r, folder), touch=False)
            return folder

    def _claim_folder(self, folder: str) -> str:
        """A folder specimens are being filed in, made if it is new.

        Typed with another case, a folder already there is the one meant, so
        "batch 2" files into "Batch 2" rather than beside it.
        """
        existing = self.folders()
        folder = _as_written(folder, existing)
        self._keep_folders([*existing, *folder_ancestors(folder)])
        return folder

    def _refile(self, fn) -> list[str]:
        """Give each specimen whose folder ``fn`` maps to a new one that folder."""
        changed = []
        for record in self.all():
            try:
                new = fn(folder_path(record.get("folder") or ""))
            except ValueError:
                continue
            if new is not None and new != (record.get("folder") or ""):
                self.update(record["name"], lambda r, f=new: _file_in(r, f), touch=False)
                changed.append(record["name"])
        return changed

    def unique_name(self, base: str, taken: set[str] | None = None, ignore: Iterable[str] = ()) -> str:
        """``base`` if free, otherwise the next replicate number, then a suffix.

        ``taken`` holds names already given out in a batch still being planned;
        ``ignore`` names specimens to count as free, such as one being renamed.
        """
        base = base or "sample"
        ignore = set(ignore)

        def used(name: str) -> bool:
            return (taken is not None and name in taken) or (name not in ignore and self.sample_dir(name).exists())

        if not used(base):
            return base
        match = re.match(r"^(.*_)(\d+)$", base)
        if match:
            prefix, n = match.group(1), int(match.group(2))
            while used(f"{prefix}{n}"):
                n += 1
            return f"{prefix}{n}"
        n = 2
        while used(f"{base}-{n}"):
            n += 1
        return f"{base}-{n}"

    def next_replicate(self, meta: dict, taken: set[str] | None = None, ignore: Iterable[str] = ()) -> int:
        """The next free replicate number among samples sharing age, line and genotype.

        Numbered across the sexes -- P52_Odam-Cre_WT_F_1, then _M_2 -- so the
        number still tells a group's animals apart on its own, as it did in
        the names from before they carried the sex, and goes on from them.
        """
        stem = compose_name({**meta, "sex": None, "replicate": None})
        numbers = set()
        for name in (set(self.names()) - set(ignore)) | (taken or set()):
            if name.startswith(stem + "_"):
                tail = _REPLICATE_TAIL.match(name[len(stem) + 1:])
                if tail:
                    numbers.add(int(tail.group(1)))
        n = 1
        while n in numbers:
            n += 1
        return n

    def plan_names(self, items: list[dict], ignore: Iterable[str] = ()) -> list[dict]:
        """The name each of a batch of new specimens will be given, in order.

        Each item holds the specimen's ``meta``, the name it was ``found`` under
        and any ``name`` typed in for it. A typed name wins. A found name that
        already says what the details say is kept as it is, so it goes on
        matching the files it names. Otherwise details with an age and a
        genotype compose the lab's name, taking the next free replicate number
        when none is given, and anything less keeps the found name, made safe
        for a folder. Names are unique across the workspace and the batch,
        so a preview shows exactly what an import makes. Specimens named in
        ``ignore`` do not count as taken: a specimen renamed to follow its
        details may keep its own number.
        """
        taken: set[str] = set()
        ignore = set(ignore)
        out = []
        for item in items:
            meta = {k: v for k, v in (item.get("meta") or {}).items() if v not in (None, "")}
            typed = str(item.get("name") or "").strip()
            found = str(item.get("found") or "").strip()
            said = parse_name(found) if found else {}
            replicate = meta.get("replicate")
            error = None
            if typed:
                source, base = "typed", typed
                if not valid_name(typed):
                    error = f"{typed!r} is not a usable name: letters, digits, . _ + - only"
            elif said and valid_name(found) and all(
                    str(said.get(k) or "") == str(meta.get(k) or "") for k in ("age", "line", "genotype", "sex", "replicate")):
                source, base = "found", found
            elif composes_name(meta):
                source = "details"
                if replicate in (None, ""):
                    replicate = self.next_replicate(meta, taken, ignore)
                base = compose_name({**meta, "replicate": replicate})
            else:
                source, base = "found", safe_name(item.get("found") or "") or "sample"
            name = base if error else self.unique_name(base, taken, ignore)
            if source == "details":
                # A replicate already taken moves on to the next free number,
                # and the record's number follows its name.
                tail = name.rsplit("_", 1)[-1]
                replicate = int(tail) if tail.isdigit() else replicate
            taken.add(name)
            out.append({"key": item.get("key"), "name": name, "from": source, "error": error,
                        "replicate": replicate})
        return out

    def known_values(self) -> dict[str, list[str]]:
        """Every value each detail has taken in this workspace, for suggestions and matching."""
        values: dict[str, set] = {k: set() for k in META_FIELDS if k not in ("notes", "replicate")}
        for record in self.all():
            for k, v in (record.get("meta") or {}).items():
                if k in values and v not in (None, ""):
                    values[k].add(str(v))
        return {k: sorted(v) for k, v in values.items() if v}

    def create(self, meta: dict, *, name: str | None = None,
               source: dict | None = None, artifacts: dict | None = None, folder: str = "") -> dict:
        with self.lock:
            shelf = folder_path(folder)  # the library folder; `folder` below is the one on disk
            meta = {k: meta.get(k) for k in META_FIELDS if meta.get(k) not in (None, "")}
            name = name or compose_name(meta) or "sample"
            if not valid_name(name):
                raise ValueError(f"{name!r} is not a usable sample name: letters, digits, . _ + - only")
            name = self.unique_name(name)
            folder = self.sample_dir(name)
            folder.mkdir(parents=True)
            source = dict(source or {})
            if source.get("kind") and not source.get("format"):
                source["format"] = source_format(source["kind"], source.get("path"))
            record = {
                "name": name,
                "created": now_iso(),
                "meta": meta,
                "source": source,
                "artifacts": {},
                "steps": {},
                "crop_box": None,
            }
            if shelf:
                record["folder"] = self._claim_folder(shelf)
            scan = source.get("path")
            if scan:
                source["path"] = self.store_path(record, scan)
            for key, path in (artifacts or {}).items():
                self._set_artifact(record, key, path)
            if scan and source.get("kind") in ("isq", "rsq", "dicom", "bmp", "volume"):
                self._set_artifact(record, "raw", scan)
            storage = self._default_storage(record)
            if storage is not None:
                record["storage"] = self.store_path(record, storage)
            self.group_slot(group_of(meta))
            return self.save(record)

    # -- large files ------------------------------------------------------

    def storage_dir(self, record: dict) -> Path | None:
        """The folder beside the scan that takes this sample's large files, or None.

        A scan linked in place keeps its conversion and its reoriented mandible
        on its own drive, in a MicroCT-Studio folder beside it, and the record
        keeps that folder's path. A scan that was uploaded or copied already
        lives in the workspace, and so do its large files (None).
        """
        if record.get("storage"):
            return self.resolve_path(record, record["storage"])
        return self._default_storage(record)

    def _default_storage(self, record: dict) -> Path | None:
        src = record.get("source") or {}
        if src.get("mode") != "link" or src.get("kind") not in _STORED_KINDS:
            return None
        scan = self.source_path(record)
        if scan is None:
            return None
        real = Path(os.path.realpath(scan))
        if real == self.root or self.root in real.parents:
            return None
        return storage_root(scan) / record["name"]

    def source_path(self, record: dict) -> Path | None:
        """The raw scan's location on this computer (None when its drive is not connected)."""
        return self.resolve_path(record, (record.get("source") or {}).get("path"))

    # -- recorded paths ---------------------------------------------------

    def store_path(self, record: dict, path: str | Path) -> str:
        """How a record keeps a path: relative inside the sample's folder, portable outside it."""
        folder = self.sample_dir(record["name"]).resolve()
        real = Path(os.path.realpath(path))
        try:
            return real.relative_to(folder).as_posix()
        except ValueError:
            return portable_path(real, self.root)

    def resolve_path(self, record: dict, value: str | None) -> Path | None:
        """A recorded path on this computer, which may not exist; None when its drive is not connected."""
        if not value:
            return None
        if value.startswith("@"):
            return resolve_portable(value, self.root)
        if os.path.isabs(value) or _FOREIGN_ABS.match(value):
            return Path(value)
        return self.sample_dir(record["name"]) / value.replace("\\", "/")

    @staticmethod
    def is_local(value: str) -> bool:
        """Whether a recorded path is inside the workspace (the sample's folder or @ws/)."""
        return value.startswith("@ws/") or not (value.startswith("@") or os.path.isabs(value)
                                                or _FOREIGN_ABS.match(value))

    def normalize_paths(self) -> int:
        """Rewrite absolute paths in records into the portable form, where it can be worked out.

        Records written before paths were portable hold absolute ones, which
        stop leading anywhere when the workspace's drive is opened on another
        computer. The records' "updated" times are left alone.
        """
        changed = 0
        with self.lock:
            for name in self.names():
                record = self.get(name)
                if record is None:
                    continue

                def fix(value):
                    if not isinstance(value, str) or not value or value.startswith("@") or _FOREIGN_ABS.match(value):
                        return value
                    if os.path.isabs(value):
                        return self.store_path(record, value)
                    return value.replace("\\", "/")

                before = json.dumps(record, sort_keys=True)
                arts = record.get("artifacts") or {}
                for key in list(arts):
                    arts[key] = fix(arts[key])
                src = record.get("source") or {}
                if src.get("path"):
                    src["path"] = fix(src["path"])
                if record.get("storage"):
                    record["storage"] = fix(record["storage"])
                if json.dumps(record, sort_keys=True) != before:
                    write_json(self._record_path(name), record)
                    self._cache.pop(name, None)
                    changed += 1
        return changed

    def rename(self, old: str, new: str) -> dict:
        """Rename a sample: its folder, every file carrying its name, and its record."""
        with self.lock:
            if old == new:
                return self.get(old)
            if not valid_name(new):
                raise ValueError(f"{new!r} is not a usable sample name")
            if self.sample_dir(new).exists():
                raise ValueError(f"A sample named {new} already exists")
            record = self.get(old)
            if record is None:
                raise KeyError(old)
            src, dst = self.sample_dir(old), self.sample_dir(new)
            os.replace(src, dst)
            _rename_named(dst, old, new)
            record["name"] = new
            fixed = {}
            for key, path in record.get("artifacts", {}).items():
                if not (path.startswith("@") or os.path.isabs(path) or _FOREIGN_ABS.match(path)):
                    path = "/".join(_renamed(p, old, new) for p in path.replace("\\", "/").split("/"))
                fixed[key] = path
            record["artifacts"] = fixed
            self._rename_storage(record, old, new)
            self._cache.pop(old, None)
            saved = self.save(record)
            self._rename_in_projects(old, new)
            return saved

    def _rename_storage(self, record: dict, old: str, new: str) -> None:
        """Carry a rename over to the large files kept beside the scan.

        Their folder and the files named after the sample are renamed, and the
        record follows. With the scan's drive disconnected nothing can be
        renamed, so the record keeps the old folder, whose files still work.
        """
        old_dir = self.resolve_path(record, record.get("storage"))
        if old_dir is None or old_dir.name != old:
            return
        new_dir = old_dir.with_name(new)
        if not old_dir.parent.parent.is_dir() or new_dir.exists():
            return
        moved = {}
        for key, value in record.get("artifacts", {}).items():
            path = self.resolve_path(record, value)
            if path is not None and old_dir in path.parents:
                moved[key] = new_dir.joinpath(*(_renamed(part, old, new) for part in path.relative_to(old_dir).parts))
        if old_dir.is_dir():
            try:
                os.replace(old_dir, new_dir)
            except OSError:
                return
            _rename_named(new_dir, old, new)
        record["storage"] = self.store_path(record, new_dir)
        for key, path in moved.items():
            record["artifacts"][key] = self.store_path(record, path)

    def archive(self, name: str) -> Path:
        """Move a sample to trash/.

        Linked data outside the workspace is never touched, and neither are the
        large files kept beside a scan: moving them into the workspace's trash
        would fill the drive they were put elsewhere to spare.
        """
        with self.lock:
            src = self.sample_dir(name)
            if not src.exists():
                raise KeyError(name)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            dst = self.trash_dir / f"{name}__{stamp}"
            shutil.move(str(src), str(dst))
            self._cache.pop(name, None)
            return dst

    # -- artifacts --------------------------------------------------------

    def _set_artifact(self, record: dict, key: str, path: str | Path | None) -> None:
        arts = record.setdefault("artifacts", {})
        if path is None:
            arts.pop(key, None)
            return
        arts[key] = self.store_path(record, path)

    def set_artifacts(self, name: str, updates: dict[str, str | Path | None]) -> dict:
        def apply(record):
            for key, path in updates.items():
                self._set_artifact(record, key, path)
        return self.update(name, apply)

    def artifact(self, record: dict | str, key: str) -> Path | None:
        """Absolute path of an artifact, or None if unrecorded or gone from disk."""
        if isinstance(record, str):
            record = self.get(record) or {}
        path = self.resolve_path(record, (record.get("artifacts") or {}).get(key))
        return path if path is not None and path.exists() else None

    def artifact_paths(self, record: dict) -> dict[str, str]:
        out = {}
        for key in (record.get("artifacts") or {}):
            path = self.artifact(record, key)
            if path is not None:
                out[key] = str(path)
        return out

    # -- steps ------------------------------------------------------------

    def record_step(self, name: str, step: str, **fields) -> dict:
        def apply(record):
            entry = record.setdefault("steps", {}).setdefault(step, {})
            entry.update(fields)
        return self.update(name, apply)


def _as_written(folder: str, existing: Iterable[str]) -> str:
    """A folder with each level spelled as the library already spells it, whatever the case typed."""
    known = {f.casefold(): f for f in existing}
    parts: list[str] = []
    for part in folder.split("/") if folder else []:
        path = "/".join([*parts, part])
        parts = known.get(path.casefold(), path).split("/")
    return "/".join(parts)


def _file_in(record: dict, folder: str) -> None:
    """Put a record in a library folder; the top of the library is no folder at all."""
    if folder:
        record["folder"] = folder
    else:
        record.pop("folder", None)


def _renamed(entry: str, old: str, new: str) -> str:
    """A file or folder name with a sample's old name swapped for its new one."""
    if entry == old or entry.startswith(old + "_") or entry.startswith(old + "."):
        return new + entry[len(old):]
    return entry


def _rename_named(folder: Path, old: str, new: str) -> None:
    """Rename every file and folder under ``folder`` that the pipeline named after the sample.

    Deepest first, so a renamed folder never invalidates a path still to be visited.
    """
    for root, dirs, files in os.walk(folder, topdown=False):
        for entry in files + dirs:
            renamed = _renamed(entry, old, new)
            if renamed != entry:
                os.replace(os.path.join(root, entry), os.path.join(root, renamed))


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
