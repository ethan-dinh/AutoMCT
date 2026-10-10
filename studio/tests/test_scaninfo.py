"""
Reading a scan's details from its own files and folders.

Each scanner format is written here the way the scanner writes it -- a Scanco
ISQ pre-header with a VMS creation date and the name typed at the scanner, a
DICOM file with its dates and the animal's sex and age, a SkyScan BMP stack
with the reconstruction's .log beside it -- and the tests pin what the import
form is told: the value, where it came from, and whether it is a reading or a
guess. The voxel size gets its own tests, since the studio's value overrides
the file's: it is filled in only for a stack that records none of its own.

Run with:  python -m pytest studio/tests/test_scaninfo.py
"""

import datetime as dt
import struct
import sys
from pathlib import Path

import nrrd
import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio import scaninfo  # noqa: E402
from studio.scaninfo import describe, parse_date, read_words  # noqa: E402
from studio.tests.phantom import write_isq  # noqa: E402

KNOWN = {"line": ["Odam-Cre"], "genotype": ["WT", "Hif1a cKO"], "study": ["Odam aging"], "scanner": ["Scanco µCT 50"]}


def isq_with(path: Path, when: dt.datetime | None, name: str = "") -> Path:
    """A phantom ISQ with a real creation date and scanner-typed name in its header."""
    write_isq(path, shape=(20, 16, 12))
    header = bytearray(path.read_bytes())
    if when is not None:
        struct.pack_into("<q", header, 36, int((when - dt.datetime(1858, 11, 17)).total_seconds() * 10_000_000))
    header[128:168] = name.encode("latin-1").ljust(40, b"\0")
    path.write_bytes(bytes(header))
    return path


def candidate(kind: str, path: Path, name: str | None = None, folders=()) -> dict:
    from studio.workspace import parse_name
    name = name or (path.stem if path.is_file() else path.name)
    return {"name": name, "meta": parse_name(name), "source": {"kind": kind, "path": str(path)},
            "artifacts": {}, "folders": list(folders)}


def test_an_isq_header_gives_its_date_its_voxel_size_and_the_name_typed_at_the_scanner(tmp_path):
    isq = isq_with(tmp_path / "C0004019.ISQ", dt.datetime(2025, 3, 14, 10, 30), "P52 Odam-Cre WT 3")
    c = describe(candidate("isq", isq), KNOWN)
    assert c["meta"]["scan_date"] == "2025-03-14"
    assert c["sources"]["scan_date"] == {"from": "From the ISQ header", "guess": False}
    # The name typed at the scanner is free text: what it says is a guess.
    assert (c["meta"]["age"], c["meta"]["line"], c["meta"]["genotype"], c["meta"]["replicate"]) == ("P52", "Odam-Cre", "WT", 3)
    assert c["sources"]["age"]["guess"] and "P52 Odam-Cre WT 3" in c["sources"]["age"]["from"]
    assert c["recorded"]["scanner_name"] == "P52 Odam-Cre WT 3"
    # The file's own voxel size is reported, never copied into the details.
    assert c["recorded"]["voxel_um"] == 12.0 and "voxel_um" not in c["meta"]
    # ISQ files come from a Scanco scanner; this workspace has one by name.
    assert c["meta"]["scanner"] == "Scanco µCT 50" and c["sources"]["scanner"]["guess"]


def test_an_isq_with_no_date_or_name_gives_only_what_it_holds(tmp_path):
    isq = isq_with(tmp_path / "C0004020.ISQ", None)
    c = describe(candidate("isq", isq), {})
    assert "scan_date" not in c["meta"] and "scanner_name" not in c["recorded"]
    assert c["meta"]["scanner"] == "Scanco"


def test_the_isq_offsets_are_the_converters(tmp_path):
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    sys.path.insert(0, tools)
    from convert_ISQ_NRRD import _FIELDS  # noqa: PLC0415
    offsets = {name: offset for name, offset, _ in _FIELDS}
    assert (offsets["creation_date"], offsets["dim_x_pixels"], offsets["dim_x_um"],
            offsets["slice_increment_um"], offsets["name"]) == (36, 44, 56, 72, 128)


def test_a_skyscan_log_gives_a_bmp_stack_its_voxel_size(tmp_path):
    from PIL import Image
    stack = tmp_path / "Mouse7_Rec"
    stack.mkdir()
    for i in range(12):
        Image.new("L", (4, 4)).save(stack / f"Mouse7__rec{i:04d}.bmp")
    missing = describe(candidate("bmp", stack), {})
    assert missing["recorded"]["needs_voxel"] and "voxel_um" not in missing["meta"]

    (stack / "Mouse7__rec.log").write_text(
        "[System]\nScanner=Skyscan1272\n[Acquisition]\nImage Pixel Size (um)=    9.99937\n"
        "Study Date and Time=25 Jan 2019  11h:33m:24s\n[Reconstruction]\nPixel Size (um)=9.00001\n"
        "Time and Date=Jan 28, 2019  13:17:00\n", encoding="latin-1")
    c = describe(candidate("bmp", stack), {})
    # The reconstruction's pixel size, since that is the size of the slices' voxels.
    assert c["meta"]["voxel_um"] == 9.0 and not c["sources"]["voxel_um"]["guess"]
    assert c["meta"]["scanner"] == "SkyScan 1272"
    assert c["meta"]["scan_date"] == "2019-01-25"
    assert "needs_voxel" not in c["recorded"]


def write_dicom(folder: Path, **tags) -> Path:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    folder.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
    ds.Rows = ds.Columns = 4
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelData = bytes(32)
    for key, value in tags.items():
        setattr(ds, key, value)
    path = folder / "IM0001.dcm"
    ds.save_as(path, enforce_file_format=True)
    return path


def test_a_dicom_header_gives_dates_the_scanner_and_the_animal(tmp_path):
    folder = tmp_path / "S0001"
    write_dicom(folder, PatientName="P52^WT", PatientSex="F", PatientAge="052D", StudyDate="20240611",
                Manufacturer="SCANCO Medical AG", ManufacturerModelName="SCANCO uCT 50",
                PixelSpacing=[0.009, 0.009], SliceThickness=0.009)
    c = describe(candidate("dicom", folder), {})
    assert c["meta"]["scan_date"] == "2024-06-11"
    assert (c["meta"]["sex"], c["meta"]["age"]) == ("F", "P52")
    assert not c["sources"]["age"]["guess"], "the header's age is a reading"
    # "SCANCO Medical AG" and "SCANCO uCT 50" are one scanner, named once.
    assert c["meta"]["scanner"] == "SCANCO uCT 50"
    assert c["meta"]["genotype"] == "WT" and c["sources"]["genotype"]["guess"]
    assert c["recorded"]["voxel_um"] == 9.0


def test_a_volume_reports_the_voxel_size_it_records_or_that_it_has_none(tmp_path):
    vol = np.zeros((6, 5, 4), np.int16)
    nrrd.write(str(tmp_path / "f0004019.nrrd"), vol, {"space directions": np.diag([0.008, 0.008, 0.016]),
                                                     "space": "left-posterior-superior"})
    nrrd.write(str(tmp_path / "f0004021.nrrd"), vol, {"space directions": np.eye(3), "space": "left-posterior-superior"})
    # A TIFF written with no resolution gets the writer's 1 pixel per unit.
    tifffile.imwrite(tmp_path / "f0004020.tif", vol, photometric="minisblack")
    with_size = describe(candidate("volume", tmp_path / "f0004019.nrrd"), {})
    # NRRD rows are X, Y, Z on disk; the pipeline's order is Z, Y, X.
    assert with_size["recorded"]["voxel_um"] == [16.0, 8.0, 8.0]
    assert "needs_voxel" not in with_size["recorded"]
    # 1 mm on every axis is a placeholder, not a micro-CT voxel.
    assert describe(candidate("volume", tmp_path / "f0004020.tif"), {})["recorded"]["needs_voxel"]
    assert describe(candidate("volume", tmp_path / "f0004021.nrrd"), {})["recorded"]["needs_voxel"]


def test_folders_say_what_their_names_spell_out():
    assert read_words("P52 Hif1a cKO females", KNOWN) == {"age": "P52", "genotype": "Hif1a cKO", "sex": "F"}
    assert read_words("2025-03-14 Odam-Cre", KNOWN) == {"scan_date": "2025-03-14", "line": "Odam-Cre"}
    assert read_words("Odam aging", KNOWN) == {"study": "Odam aging"}
    assert read_words("e18.5_KO_left", {}) == {"age": "E18.5", "genotype": "KO", "side": "L"}
    # Nothing is read into a name that does not say it outright.
    assert read_words("RAW", KNOWN) == {} and read_words("Mouse 3", KNOWN) == {}
    assert read_words("males and females", {}) == {}


def test_a_folder_is_a_guess_and_never_overrides_the_file(tmp_path):
    isq = isq_with(tmp_path / "C0004021.ISQ", dt.datetime(2024, 1, 2))
    c = describe(candidate("isq", isq, folders=["2023-05-06 P60 WT", "RAW"]), KNOWN)
    assert c["meta"]["scan_date"] == "2024-01-02" and not c["sources"]["scan_date"]["guess"]
    assert c["meta"]["age"] == "P60" and c["sources"]["age"] == {"from": "From the folder “2023-05-06 P60 WT”", "guess": True}
    assert c["meta"]["genotype"] == "WT"


def test_a_name_in_the_lab_convention_is_read_not_guessed(tmp_path):
    isq = isq_with(tmp_path / "P82_Odam-Cre_WT_2.isq", None)
    c = describe(candidate("isq", isq, folders=["P60"]), KNOWN)
    assert c["meta"]["age"] == "P82", "the specimen's own name beats its folder"
    assert c["sources"]["age"] == {"from": "From the name “P82_Odam-Cre_WT_2”", "guess": False}


@pytest.mark.parametrize("text, iso", [
    ("2025-03-14", "2025-03-14"), ("20250314", "2025-03-14"), ("3/14/2025", "2025-03-14"), ("3/14/25", "2025-03-14"),
    ("14.03.2025", "2025-03-14"), ("25 Jan 2019  11h:33m:24s", "2019-01-25"), ("Jan 25, 2019  13:17:00", "2019-01-25"),
    ("14-Mar-2025", "2025-03-14"), ("C0004019", None), ("P52_Odam-Cre_WT_1", None), ("2025-13-40", None),
    ("1858-11-17", None), ("", None),
])
def test_dates_in_the_forms_scanners_and_people_write_them(text, iso):
    assert parse_date(text) == iso


def test_a_header_that_cannot_be_read_leaves_the_details_to_ask_for(tmp_path):
    bad = tmp_path / "broken.isq"
    bad.write_bytes(b"not an isq at all")
    c = describe(candidate("isq", bad), {})
    assert c["recorded"] == {} and set(c["meta"]) == {"scanner"}
    assert scaninfo.describe_all([candidate("isq", bad)] * 6, {})[5]["meta"]["scanner"] == "Scanco"
