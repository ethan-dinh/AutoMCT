"""
Scanco's hydroxyapatite density calibration, read from a scan's own files.

A Scanco scanner converts attenuation to mineral density with a calibration
measured on its HA phantom, and writes that calibration into every ISQ it
reconstructs. Stored voxel values are linear attenuation times the header's
``mu_scaling``, so

    density [mg HA/cm³] = value / mu_scaling * slope + intercept

with ``mu_scaling`` taken as 1 for a volume the converter already rescaled to
1/cm (``--apply-scaling``). On f0004028 (70 kVp, 0.5 mm Al) the calibration is
slope 375.507 and intercept -197.288 mg HA/cm³ per 1/cm at mu_scaling 4096; the
water attenuation it records, 0.4741 1/cm, maps to -19 mg HA/cm³, and the
enamel and dentin peaks of the same specimen to about 1930 and 1180.

Where the calibration sits in an ISQ
------------------------------------
The extended header follows the 512-byte pre-header. At byte 512 a
"MultiHeader" record points to an index of 128-byte entries, each a 16-byte
name at +8, its first block (512 bytes each, counted from byte 512) at +24 and
its size in bytes at +40. The "Calibration" entry's block holds:

    +28   64 chars   description  ("70 kVp, 0.5mm Al, BH: 1200 mg HA/ccm, Scaling 4096")
    +112  256 chars  calibration file
    +632  int32      rescale type
    +640  int32      mu scaling
    +648  16 chars   units        ("mg HA/ccm")
    +664  double     slope
    +672  double     intercept
    +688  double     attenuation of water, 1/cm

The doubles are VAX G_floating, from the scanners' OpenVMS lineage, not IEEE:
read as IEEE they come out as denormals near 1e-310.

Where the calibration goes next
-------------------------------
``convert_ISQ_NRRD.py`` copies it into ``<stem>_meta.json`` as
``"calibration"``, beside the record of whether the volume was rescaled to
1/cm and whether the noise floor shifted it. ``find_calibration`` reads any of
the three places a calibration can be found -- that metadata file, the ISQ
itself, or a hand-written JSON with ``slope`` and ``intercept`` -- and returns
it with the divisor and offset that apply to the converted volume's values.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

BLOCK = 512
_MULTI = b"MultiHeader     "
_CALIBRATION = b"Calibration     "
_ENTRY = 128

# The pre-header's own scaling, which is what the stored values carry.
_PRE_MU_SCALING = 88
_PRE_EXTENDED_BLOCKS = 508

# Longest extended header read looking for the calibration: real ones are 6
# blocks, and the stated count can undercount by a block or two.
_MAX_BLOCKS = 64


def vax_g_to_float(raw: bytes) -> float:
    """A VAX G_floating double as a Python float.

    G_floating has IEEE's 1/11/52 bit layout, but stored as four little-endian
    16-bit words, most significant first, with an exponent bias of 1024 and a
    0.1f mantissa: reversing the words gives an IEEE double four times too big.
    """
    words = struct.unpack("<4H", raw[:8])
    return struct.unpack("<d", struct.pack("<4H", *words[::-1]))[0] / 4.0


def float_to_vax_g(value: float) -> bytes:
    """The inverse of ``vax_g_to_float``, for writing test headers."""
    words = struct.unpack("<4H", struct.pack("<d", value * 4.0))
    return struct.pack("<4H", *words[::-1])


def extended_header(slope: float, intercept: float, *, mu_scaling: int = 4096, mu_water: float = 0.0,
                    units: str = "mg HA/ccm", description: str = "") -> bytes:
    """An extended header holding one calibration, laid out as the scanner lays it out.

    Five blocks -- the MultiHeader record, its index, and the calibration's
    three -- so the pre-header in front of it states 5 in
    header_data_offset_blocks. For writing test scans.
    """
    out = bytearray(5 * BLOCK)

    def entry(at: int, name: bytes, block: int, size: int) -> None:
        out[at + 8:at + 24] = name
        struct.pack_into("<i", out, at + 24, block)
        struct.pack_into("<i", out, at + 40, size)

    entry(0, _MULTI, 1, 256)                # the index is block 1, at byte 1024
    entry(BLOCK, _CALIBRATION, 2, 3 * BLOCK)  # the calibration is block 2, at byte 1536
    cal = 2 * BLOCK
    out[cal + 28:cal + 28 + 64] = description.encode("ascii")[:64].ljust(64)
    struct.pack_into("<i", out, cal + 632, 2)
    struct.pack_into("<i", out, cal + 640, mu_scaling)
    out[cal + 648:cal + 664] = units.encode("ascii")[:16].ljust(16)
    out[cal + 664:cal + 672] = float_to_vax_g(slope)
    out[cal + 672:cal + 680] = float_to_vax_g(intercept)
    out[cal + 688:cal + 696] = float_to_vax_g(mu_water)
    return bytes(out)


def _int(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _text(data: bytes, offset: int, length: int) -> str:
    return data[offset:offset + length].split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


def parse_calibration(header: bytes) -> dict | None:
    """The calibration block of an ISQ header (pre-header plus extended header).

    Returns None when the header has no extended header, no calibration entry,
    or a slope that is not a usable positive number.
    """
    if len(header) < 2 * BLOCK or header[BLOCK + 8:BLOCK + 24] != _MULTI:
        return None
    index = BLOCK + _int(header, BLOCK + 24) * BLOCK
    index_bytes = _int(header, BLOCK + 40)
    if not 0 < index_bytes <= BLOCK:
        index_bytes = BLOCK
    payload = None
    for entry in range(index, min(index + index_bytes, len(header)) - _ENTRY + 1, _ENTRY):
        if header[entry + 8:entry + 24] == _CALIBRATION:
            payload = BLOCK + _int(header, entry + 24) * BLOCK
            break
    if payload is None or payload + 696 > len(header):
        return None

    slope = vax_g_to_float(header[payload + 664:payload + 672])
    intercept = vax_g_to_float(header[payload + 672:payload + 680])
    if not (math.isfinite(slope) and math.isfinite(intercept)) or slope <= 0:
        return None
    mu_water = vax_g_to_float(header[payload + 688:payload + 696])
    return {
        "slope": slope,
        "intercept": intercept,
        "units": _text(header, payload + 648, 16) or "mg HA/ccm",
        "mu_scaling": _int(header, payload + 640),
        "mu_water": mu_water if math.isfinite(mu_water) else None,
        "rescale_type": _int(header, payload + 632),
        "description": _text(header, payload + 28, 64),
        "file": _text(header, payload + 112, 256),
    }


def read_isq_calibration(path: str | Path) -> dict | None:
    """The calibration stored in a Scanco .isq file, or None if it holds none."""
    path = Path(path)
    with open(path, "rb") as handle:
        pre = handle.read(BLOCK)
        if len(pre) < BLOCK:
            return None
        blocks = _int(pre, _PRE_EXTENDED_BLOCKS)
        if blocks <= 0:
            return None
        header = pre + handle.read(min(blocks + 2, _MAX_BLOCKS) * BLOCK)
    calibration = parse_calibration(header)
    if calibration is None:
        return None
    # Voxel values are scaled by the pre-header's mu_scaling; the calibration
    # block repeats it, but the pre-header is what the pixel data answers to.
    calibration["mu_scaling"] = _int(pre, _PRE_MU_SCALING) or calibration["mu_scaling"]
    calibration["source"] = str(path)
    return calibration


# ---------------------------------------------------------------------------
# Finding the calibration for a converted volume
# ---------------------------------------------------------------------------

def _from_json(path: Path) -> tuple[dict | None, dict]:
    """(calibration, conversion facts) from a converter meta file or a calibration JSON."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return None, {}
    facts: dict = {}
    params = data.get("_filter_params")
    if isinstance(params, dict) and "apply_rescale" in params:
        facts["rescaled"] = bool(params["apply_rescale"])
    floor = data.get("_noise_floor")
    if isinstance(floor, dict) and floor.get("mode") == "shift":
        # Shift mode subtracted the level from every surviving voxel.
        facts["offset"] = float(floor.get("level") or 0.0)

    block = data.get("calibration") if isinstance(data.get("calibration"), dict) else data
    if "slope" not in block or "intercept" not in block:
        return None, facts
    calibration = dict(block)
    calibration["slope"] = float(calibration["slope"])
    calibration["intercept"] = float(calibration["intercept"])
    if not calibration.get("mu_scaling"):
        calibration["mu_scaling"] = data.get("mu_scaling") or 1
    calibration.setdefault("units", "mg HA/ccm")
    calibration["source"] = str(path)
    return calibration, facts


def find_calibration(paths: list[str | Path]) -> tuple[dict | None, list[str]]:
    """The HA calibration for a volume, from the first of ``paths`` that holds one.

    Each path is a converter ``<stem>_meta.json``, a Scanco ``.isq``, or a JSON
    with ``slope`` and ``intercept`` (and optionally ``mu_scaling`` and
    ``units``). What a converter meta file says about the conversion applies
    whichever path the numbers came from: a volume rescaled to 1/cm is divided
    by 1, not by mu_scaling, and a noise floor in shift mode is added back.

    Returns (calibration, notes). The calibration carries ``divisor`` and
    ``offset``, so ``density = (value + offset) / divisor * slope + intercept``;
    the notes say which paths were read and why any were passed over.
    """
    calibration = None
    facts: dict = {}
    notes: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            notes.append(f"{path} does not exist")
            continue
        try:
            if path.suffix.lower() == ".isq":
                found, more = read_isq_calibration(path), {}
            else:
                found, more = _from_json(path)
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            notes.append(f"Could not read {path.name}: {exc}")
            continue
        for key, value in more.items():
            facts.setdefault(key, value)
        if found is None:
            notes.append(f"No HA calibration in {path.name}")
        elif calibration is None:
            calibration = found
    if calibration is None:
        return None, notes
    divisor = 1.0 if facts.get("rescaled") else float(calibration.get("mu_scaling") or 1)
    calibration.update(divisor=divisor, offset=float(facts.get("offset", 0.0)),
                       rescaled=bool(facts.get("rescaled", False)))
    return calibration, notes


def to_density(value, calibration: dict):
    """Stored voxel value(s) -> density in the calibration's units (mg HA/cm³)."""
    return ((value + calibration.get("offset", 0.0)) / calibration["divisor"]
            * calibration["slope"] + calibration["intercept"])
