"""
Tests for reading Scanco's HA density calibration out of an ISQ.

The reference is the real header of f0004028 (70 kVp, 0.5 mm Al): its
calibration block decodes to slope 375.507, intercept -197.288 and water at
0.4741 1/cm. Those values are stored as VAX G_floating doubles, pinned here
byte for byte, so a decoder that reads them as IEEE -- and gets numbers near
1e-310 -- fails loudly rather than converting every density to zero.

The rest of the suite builds scans with ``extended_header`` and checks the
claims a density depends on: the calibration is found through the header's own
index, the stored values answer to the pre-header's mu_scaling, a volume the
converter rescaled to 1/cm is not divided again, and a shifted noise floor is
added back.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scanco_calibration as sc  # noqa: E402

SLOPE, INTERCEPT, WATER = 375.507, -197.288, 0.4741

# Bytes 2200-2231 of f0004028.isq, as the scanner wrote them.
REAL_DOUBLES = {
    "slope": (bytes.fromhex("97401c7800a00000"), SLOPE),
    "intercept": (bytes.fromhex("88c037a900400000"), INTERCEPT),
    "mu_water": (bytes.fromhex("fe3fa75700800000"), WATER),
}


def write_isq(path: Path, extended: bytes | None, mu_scaling: int = 4096) -> Path:
    """A pre-header, an optional extended header, and a little int16 data."""
    pre = bytearray(sc.BLOCK)
    pre[:16] = b"CTDATA-HEADER_V1"
    struct.pack_into("<i", pre, 88, mu_scaling)
    struct.pack_into("<i", pre, 508, len(extended) // sc.BLOCK if extended else 0)
    with open(path, "wb") as handle:
        handle.write(bytes(pre))
        if extended:
            handle.write(extended)
        handle.write(np.arange(64, dtype="<i2").tobytes())
    return path


@pytest.mark.parametrize("name", sorted(REAL_DOUBLES))
def test_the_real_headers_vax_doubles_decode(name):
    raw, expected = REAL_DOUBLES[name]
    assert sc.vax_g_to_float(raw) == pytest.approx(expected, rel=1e-6)
    assert abs(struct.unpack("<d", raw)[0]) < 1e-300, "read as IEEE these are denormals"


@pytest.mark.parametrize("value", [SLOPE, INTERCEPT, WATER, 1.0, -0.5, 12345.678, 0.0])
def test_vax_doubles_round_trip(value):
    assert sc.vax_g_to_float(sc.float_to_vax_g(value)) == value


def test_the_calibration_is_found_through_the_headers_index(tmp_path):
    extended = sc.extended_header(SLOPE, INTERCEPT, mu_water=WATER,
                                  description="70 kVp, 0.5mm Al, BH: 1200 mg HA/ccm, Scaling 4096")
    cal = sc.read_isq_calibration(write_isq(tmp_path / "scan.isq", extended))
    assert cal["slope"] == pytest.approx(SLOPE) and cal["intercept"] == pytest.approx(INTERCEPT)
    assert cal["mu_water"] == pytest.approx(WATER) and cal["mu_scaling"] == 4096
    assert cal["units"] == "mg HA/ccm" and cal["description"].startswith("70 kVp")
    # Water sits near zero mg HA/cm³ on a sound calibration.
    assert abs(WATER * cal["slope"] + cal["intercept"]) < 25


def test_a_scan_without_an_extended_header_has_no_calibration(tmp_path):
    assert sc.read_isq_calibration(write_isq(tmp_path / "bare.isq", None)) is None


def test_the_values_answer_to_the_pre_headers_scaling(tmp_path):
    extended = sc.extended_header(SLOPE, INTERCEPT, mu_scaling=4096)
    cal = sc.read_isq_calibration(write_isq(tmp_path / "scan.isq", extended, mu_scaling=8192))
    assert cal["mu_scaling"] == 8192


def test_a_raw_scan_is_divided_by_its_scaling(tmp_path):
    isq = write_isq(tmp_path / "scan.isq", sc.extended_header(SLOPE, INTERCEPT))
    cal, notes = sc.find_calibration([isq])
    assert cal["divisor"] == 4096 and cal["offset"] == 0 and not notes
    assert sc.to_density(4096.0 * 5, cal) == pytest.approx(5 * SLOPE + INTERCEPT)


def test_a_volume_rescaled_to_attenuation_is_not_divided_again(tmp_path):
    """Converted before the converter copied calibrations: the meta says how, the ISQ says what."""
    meta = tmp_path / "scan_meta.json"
    meta.write_text(json.dumps({"mu_scaling": 4096, "_filter_params": {"apply_rescale": True}}))
    isq = write_isq(tmp_path / "scan.isq", sc.extended_header(SLOPE, INTERCEPT))
    cal, notes = sc.find_calibration([meta, isq])
    assert cal["divisor"] == 1 and cal["rescaled"] and cal["source"] == str(isq)
    assert notes == ["No HA calibration in scan_meta.json"]
    assert sc.to_density(5.0, cal) == pytest.approx(5 * SLOPE + INTERCEPT)


def test_the_converters_copy_of_the_calibration_is_used_first(tmp_path):
    meta = tmp_path / "scan_meta.json"
    meta.write_text(json.dumps({"mu_scaling": 4096, "_filter_params": {"apply_rescale": False},
                                "calibration": {"slope": 400.0, "intercept": -200.0, "mu_scaling": 4096}}))
    isq = write_isq(tmp_path / "scan.isq", sc.extended_header(SLOPE, INTERCEPT))
    cal, _ = sc.find_calibration([meta, isq])
    assert cal["slope"] == 400.0 and cal["source"] == str(meta) and cal["divisor"] == 4096


def test_a_shifted_noise_floor_is_added_back(tmp_path):
    meta = tmp_path / "scan_meta.json"
    meta.write_text(json.dumps({"_noise_floor": {"level": 1000.0, "mode": "shift", "fill": 0.0},
                                "calibration": {"slope": SLOPE, "intercept": INTERCEPT, "mu_scaling": 4096}}))
    cal, _ = sc.find_calibration([meta])
    assert cal["offset"] == 1000.0
    assert sc.to_density(20000.0 - 1000.0, cal) == pytest.approx(20000.0 / 4096 * SLOPE + INTERCEPT)


def test_a_hand_written_calibration(tmp_path):
    path = tmp_path / "phantom.json"
    path.write_text(json.dumps({"slope": 1000.0, "intercept": -100.0}))
    cal, _ = sc.find_calibration([path])
    assert cal["divisor"] == 1 and sc.to_density(2.0, cal) == pytest.approx(1900.0)


def test_paths_without_a_calibration_are_noted(tmp_path):
    bare = write_isq(tmp_path / "bare.isq", None)
    cal, notes = sc.find_calibration([tmp_path / "missing.json", bare])
    assert cal is None
    assert notes[0].endswith("does not exist") and notes[1] == "No HA calibration in bare.isq"


def test_the_converter_keeps_the_calibration_with_the_header(tmp_path):
    from convert_ISQ_NRRD import _FIELDS, _HEADER_SIZE, _MAGIC, read_isq  # noqa: PLC0415

    extended = sc.extended_header(SLOPE, INTERCEPT, mu_water=WATER)
    header = bytearray(_HEADER_SIZE)
    header[:len(_MAGIC)] = _MAGIC
    values = {"dim_x_pixels": 4, "dim_y_pixels": 4, "dim_z_pixels": 2, "dim_x_um": 40, "dim_y_um": 40,
              "dim_z_um": 20, "slice_increment_um": 10, "mu_scaling": 4096,
              "header_data_offset_blocks": len(extended) // _HEADER_SIZE}
    for name, offset, fmt in _FIELDS:
        if name in values:
            struct.pack_into("<" + fmt, header, offset, values[name])
    path = tmp_path / "scan.isq"
    path.write_bytes(bytes(header) + extended + np.full(32, 20000, dtype="<i2").tobytes())

    volume, parsed = read_isq(path)
    assert volume.shape == (2, 4, 4) and int(volume[0, 0, 0]) == 20000
    assert parsed["calibration"]["slope"] == pytest.approx(SLOPE)
    assert parsed["calibration"]["mu_scaling"] == 4096
