"""
What a scan's own files and folders say about it, to fill in its details.

Scanners record more than voxels. A Scanco ISQ header holds the date the scan
was measured, its voxel size and the name typed at the scanner; a DICOM series
holds its dates, the scanner's make and model, and often the animal's sex and
age; a Bruker SkyScan reconstruction keeps a .log beside its BMP slices with the
pixel size, the scanner and the study date. Folders say something too: a scan
filed in RAW/P52/WT is most likely a P52 wild type.

Every value comes back with where it was found and whether it was read or
guessed. A reading is what a header states, or what a name in the lab's
{Age}_{BaseMouse}_{Gene}_{Sex}_{N} convention spells out. A guess is inferred -- from
a folder's name, from the free text typed at the scanner, from what the rest of
the workspace holds -- and the import form marks it, so it is checked before
it goes into a specimen's name.

Only headers are read, never voxel data, and a file that cannot be parsed gives
fewer details rather than an error: an import must not fail over a detail it
could have asked for.

The voxel size is the one detail a header never fills in. The studio's voxel
size overrides the file's own (as one isotropic value), so it is set only when
the file records none -- a BMP stack, whose SkyScan log does. Otherwise the
file's own value is reported, for display, as ``recorded["voxel_um"]``.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .workspace import _AGE, _first_files, _has_dicom_magic, parse_name

log = logging.getLogger("studio.scaninfo")


class Facts:
    """Details gathered from several places; the first place to give a detail keeps it."""

    def __init__(self) -> None:
        self.meta: dict = {}
        self.sources: dict[str, dict] = {}
        self.recorded: dict = {}

    def add(self, values: dict, source: str, *, guess: bool = False) -> None:
        for key, value in values.items():
            if value in (None, "") or self.meta.get(key) not in (None, ""):
                continue
            self.meta[key] = value
            self.sources[key] = {"from": source, "guess": guess}


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def _day(year, month, day) -> str | None:
    try:
        year = int(year)
        year += 2000 if year < 100 else 0
        value = dt.date(year, int(month), int(day))
    except (TypeError, ValueError):
        return None
    # Micro-CT scans postdate 1990; anything else is a misread field.
    return value.isoformat() if 1990 <= value.year <= dt.date.today().year + 1 else None


def _month(word: str) -> int | None:
    return _MONTHS.get(word[:3].lower()) if len(word) >= 3 else None


_DATE_FORMS = (
    # 2025-03-14, 2025/3/14, 2025.03.14
    (re.compile(r"(?<!\d)(\d{4})([-/._])(\d{1,2})\2(\d{1,2})(?!\d)"), lambda m: (m[1], m[3], m[4])),
    # 20250314, as DICOM writes a date
    (re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)"), lambda m: (m[1], m[2], m[3])),
    # 25 Jan 2019, 25-Jan-2019, as SkyScan logs and some labs write one
    (re.compile(r"(?<![\w.])(\d{1,2})[\s-]*([A-Za-z]{3,9})\.?[\s,-]*(\d{4})(?!\d)"),
     lambda m: (m[3], _month(m[2]), m[1])),
    # Jan 25, 2019
    (re.compile(r"(?<![A-Za-z])([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})(?!\d)"),
     lambda m: (m[3], _month(m[1]), m[2])),
    # 3/14/2025 and 3/14/25, the US order
    (re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])"), lambda m: (m[3], m[1], m[2])),
    # 14.03.2025, the European order
    (re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?![\d.])"), lambda m: (m[3], m[2], m[1])),
)


def parse_date(text) -> str | None:
    """An ISO date from the ways scanners and people write one, or None."""
    text = str(text or "").strip()
    for pattern, parts in _DATE_FORMS:
        for match in pattern.finditer(text):
            year, month, day = parts(match)
            if month and (value := _day(year, month, day)):
                return value
    return None


_VMS_EPOCH = dt.datetime(1858, 11, 17)


def _vms_date(ticks: int) -> str | None:
    """A VMS timestamp -- 100 ns ticks since 17 November 1858, as Scanco stores one -- as an ISO date."""
    if ticks <= 0:
        return None
    try:
        when = _VMS_EPOCH + dt.timedelta(microseconds=ticks // 10)
    except OverflowError:
        return None
    return _day(when.year, when.month, when.day)


# ---------------------------------------------------------------------------
# Voxel sizes
# ---------------------------------------------------------------------------

def voxel_um(zyx_um) -> float | list[float] | None:
    """A voxel size in µm: one number when the three axes agree, else all three (Z, Y, X)."""
    try:
        values = [float(v) for v in zyx_um]
    except (TypeError, ValueError):
        return None
    if len(values) != 3 or min(values) <= 0 or max(values) > 10_000:
        return None
    if max(values) - min(values) <= 0.005 * max(values):
        return round(sum(values) / 3, 3)
    return [round(v, 3) for v in values]


def _number(text) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(text or ""))
    return float(match.group()) if match else None


# ---------------------------------------------------------------------------
# Each format's header
# ---------------------------------------------------------------------------

_ISQ_MAGIC = b"CTDATA-HEADER_V1"


def read_isq(path: Path, facts: Facts) -> None:
    """The Scanco pre-header, at the offsets tools/convert_ISQ_NRRD.py's _FIELDS gives.

    Parsed here rather than through the converter, whose import pulls in the
    whole volume pipeline. The creation date is a VMS quadword, which the
    converter's table reads as text.
    """
    with open(path, "rb") as handle:
        raw = handle.read(512)
    if len(raw) < 512 or not raw.startswith(_ISQ_MAGIC):
        return
    (ticks,) = struct.unpack_from("<q", raw, 36)
    nx, ny, nz = struct.unpack_from("<3i", raw, 44)
    ux, uy, uz = struct.unpack_from("<3i", raw, 56)
    (increment,) = struct.unpack_from("<i", raw, 72)
    name = raw[128:168].split(b"\0", 1)[0].decode("latin-1", "replace").strip()
    facts.add({"scan_date": _vms_date(ticks)}, "From the ISQ header")
    if nx > 0 and ny > 0 and nz > 0:
        facts.recorded["voxel_um"] = voxel_um([increment or uz / nz, uy / ny, ux / nx])
    if name:
        facts.recorded["scanner_name"] = name


def _dicom_age(text: str) -> str | None:
    """DICOM's age string (052D, 008W, 003M, 001Y) as the lab writes an age."""
    match = re.fullmatch(r"0*(\d+)([DWMY])", text.strip().upper())
    if not match:
        return None
    n, unit = int(match.group(1)), match.group(2)
    age = {"D": f"P{n}", "W": f"{n}wk", "M": f"{n}mo", "Y": f"{n}y"}[unit]
    return age if _AGE.match(age) else None


def read_dicom(path: Path, facts: Facts) -> None:
    try:
        import pydicom  # noqa: PLC0415
    except ImportError:
        return
    first = path if path.is_file() else next((f for f in _first_files(path, 60) if _has_dicom_magic(f)), None)
    if first is None:
        return
    ds = pydicom.dcmread(str(first), stop_before_pixels=True)

    def text(keyword: str) -> str:
        return str(ds.get(keyword, "") or "").replace("^", " ").strip()

    source = "From the DICOM header"
    for keyword in ("AcquisitionDate", "StudyDate", "SeriesDate", "ContentDate"):
        if value := parse_date(text(keyword)):
            facts.add({"scan_date": value}, source)
            break
    make, model = text("Manufacturer"), text("ManufacturerModelName")
    # "SCANCO Medical AG" and "SCANCO uCT 50" name one scanner, not two.
    same = make and model and make.split()[0].lower() in model.lower()
    facts.add({"scanner": model if same else " ".join(p for p in (make, model) if p)}, source)
    sex = text("PatientSex").upper()[:1]
    facts.add({"sex": sex if sex in ("F", "M") else None, "age": _dicom_age(text("PatientAge"))}, source)
    typed = text("PatientName") or text("PatientID")
    if typed:
        facts.recorded["scanner_name"] = typed
    try:
        dy, dx = (float(v) for v in ds.get("PixelSpacing", None) or ())
        dz = float(ds.get("SpacingBetweenSlices", None) or ds.get("SliceThickness", None) or 0)
        facts.recorded["voxel_um"] = voxel_um([dz * 1000, dy * 1000, dx * 1000]) if dz else voxel_um(
            [dy * 1000, dy * 1000, dx * 1000])
    except (TypeError, ValueError):
        pass


def _tidy_scanner(text: str | None) -> str | None:
    text = (text or "").strip()
    return re.sub(r"(?i)^sky\s*scan\s*(\d+)", r"SkyScan \1", text) or None


def read_skyscan_log(folder: Path, facts: Facts) -> None:
    """The .log a Bruker SkyScan reconstruction writes beside its BMP slices."""
    try:
        logs = [p for p in folder.iterdir()
                if p.suffix.lower() == ".log" and not p.name.startswith(".") and p.is_file()]
    except OSError:
        logs = []
    # The reconstruction's log (<prefix>_rec.log) gives the size of the voxels in the slices.
    logs.sort(key=lambda p: (not p.stem.lower().endswith("_rec"), p.name.lower()))
    fields: dict[str, str] = {}
    if logs:
        with open(logs[0], encoding="latin-1", errors="replace") as handle:
            for line in handle.read(400_000).splitlines():
                key, sep, value = line.partition("=")
                if sep:
                    fields.setdefault(key.strip().lower(), value.strip())
    size = _number(fields.get("pixel size (um)") or fields.get("image pixel size (um)"))
    # A BMP slice records no voxel size of its own, so without a log there is none.
    facts.recorded["voxel_um"] = round(size, 3) if size else None
    if not logs:
        return
    source = f"From the SkyScan log, {logs[0].name}"
    facts.add({"voxel_um": round(size, 3) if size else None,
               "scanner": _tidy_scanner(fields.get("scanner") or fields.get("dataset origin")),
               "scan_date": parse_date(fields.get("study date and time"))}, source)


def read_volume(path: Path, facts: Facts) -> None:
    """The voxel size an .nrrd or .tif records, or None when it records none.

    Exactly 1 mm on every axis counts as none: it is what a TIFF writer puts
    in when given no resolution, and what the converters write for a scan
    whose size they were not told. No micro-CT scan has voxels that big.
    """
    from .volumes import _segment_io  # noqa: PLC0415

    data_io = _segment_io()
    if path.suffix.lower() == ".nrrd":
        import nrrd  # noqa: PLC0415
        spacing = data_io.spacing_from_nrrd_header(nrrd.read_header(str(path)))
    else:
        spacing = data_io.spacing_from_tiff(str(path))
    if spacing and all(abs(s - 1.0) < 1e-9 for s in spacing):
        spacing = None
    facts.recorded["voxel_um"] = voxel_um([s * 1000 for s in spacing]) if spacing else None


READERS = {"isq": read_isq, "dicom": read_dicom, "bmp": read_skyscan_log, "volume": read_volume}


# ---------------------------------------------------------------------------
# Words in folder names and labels
# ---------------------------------------------------------------------------

_GENOTYPE_WORDS = {"wt": "WT", "ko": "KO", "cko": "cKO", "het": "Het", "hom": "Hom"}
_SEX_WORDS = {"female": "F", "females": "F", "male": "M", "males": "M"}
_SIDE_WORDS = {"left": "L", "right": "R"}


def _plain(text: str) -> str:
    return re.sub(r"[\s_]+", " ", str(text or "")).strip().lower()


def read_words(text: str, known: dict[str, list[str]] | None = None) -> dict:
    """The details a folder name or a free-text label states in so many words.

    An age (P52, E18.5, 8wk), a date, a sex or side spelled out, and any line,
    genotype or study this workspace already uses, matched as whole words. The
    common genotype words (WT, KO, cKO, Het, Hom) count when the workspace
    gives nothing longer.
    """
    known = known or {}
    plain = _plain(text)
    words = plain.split()
    out: dict = {}
    for word in words + [piece for w in words for piece in w.split("-")]:
        if _AGE.match(word):
            out["age"] = word[0].upper() + word[1:] if word[0] in "pe" else word
            break
    if value := parse_date(text):
        out["scan_date"] = value
    for key in ("study", "line", "genotype"):
        hits = [v for v in known.get(key, []) if _plain(v) and re.search(
            rf"(?<![a-z0-9]){re.escape(_plain(v))}(?![a-z0-9])", plain)]
        if hits:
            out[key] = max(hits, key=len)
    if "genotype" not in out:
        out["genotype"] = next((_GENOTYPE_WORDS[w] for w in words if w in _GENOTYPE_WORDS), None)
    for key, table in (("sex", _SEX_WORDS), ("side", _SIDE_WORDS)):
        said = {table[w] for w in words if w in table}
        if len(said) == 1:
            out[key] = said.pop()
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# A whole candidate
# ---------------------------------------------------------------------------

def describe(cand: dict, known: dict[str, list[str]] | None = None) -> dict:
    """A found specimen with the details its name, its files and its folders give.

    Returns the candidate with ``meta`` filled in, ``sources`` saying where each
    detail came from and whether it is a guess, and ``recorded`` holding what
    the file states that is not a detail to fill: its own voxel size, the name
    typed at the scanner, and ``needs_voxel`` when it records no voxel size.
    """
    known = known or {}
    facts = Facts()
    name = cand.get("name") or ""
    facts.add(cand.get("meta") or {}, f"From the name “{name}”")
    source = cand.get("source") or {}
    kind, path = source.get("kind"), source.get("path")
    if kind == "results":
        kind, path = "volume", (cand.get("artifacts") or {}).get("reoriented")
    reader = READERS.get(kind)
    if reader and path:
        try:
            reader(Path(path), facts)
        except Exception as exc:  # noqa: BLE001 - a header that will not parse is a detail to ask for
            log.info("Could not read the header of %s: %s", path, exc)

    typed = facts.recorded.get("scanner_name")
    if typed:
        label = f"From the name typed at the scanner, “{typed}”"
        facts.add(parse_name(re.sub(r"\s+", "_", typed)), label, guess=True)
        facts.add(read_words(typed, known), label, guess=True)
    for folder in cand.get("folders") or []:
        facts.add(read_words(folder, known), f"From the folder “{folder}”", guess=True)

    scanners = known.get("scanner", [])
    if kind in ("isq", "rsq"):
        scanco = [s for s in scanners if "scanco" in s.lower()]
        facts.add({"scanner": scanco[0] if len(scanco) == 1 else "Scanco"},
                  "ISQ files come from a Scanco scanner" + (", and this workspace has one" if len(scanco) == 1 else ""),
                  guess=True)
    elif kind == "volume" and len(scanners) == 1:
        # A volume was converted from some scanner's output, most likely the
        # lab's own. A BMP stack or DICOM series says which scanner it is from.
        facts.add({"scanner": scanners[0]}, "Every specimen in this workspace was scanned on it", guess=True)

    if kind in ("bmp", "volume") and "voxel_um" in facts.recorded and facts.recorded["voxel_um"] is None \
            and facts.meta.get("voxel_um") in (None, ""):
        facts.recorded["needs_voxel"] = True
    return {**cand, "meta": facts.meta, "sources": facts.sources, "recorded": facts.recorded}


def describe_all(cands: list[dict], known: dict[str, list[str]] | None = None) -> list[dict]:
    """``describe`` for a batch, reading headers in parallel: on a network drive each read waits."""
    if len(cands) <= 4:
        return [describe(c, known) for c in cands]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda c: describe(c, known), cands))
