"""
Tests for how the studio names, files and finds specimens.

The naming rule is the lab's own -- {Age}_{BaseMouse}_{Gene}_{Sex}_{N} -- and the
tests pin the two directions the studio uses it in: composing a name from what
the user typed, and reading metadata back out of an existing folder name when
bulk-importing. The second must refuse to guess: a name that does not open
with an age (Odam-CRE-A, Frag_4) yields nothing, so a legacy specimen cannot be
filed into a group on a misreading.

Run with:  python -m pytest studio/tests/test_workspace.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio import drives  # noqa: E402
from studio.workspace import (  # noqa: E402
    STORAGE_FOLDER,
    FolderExists,
    ProjectExists,
    Workspace,
    find_workspaces,
    is_workspace,
    valid_name,
    workspace_summary,
    artifacts_in,
    classify_file,
    compose_name,
    detect_kind,
    folder_path,
    group_of,
    parse_name,
    safe_name,
    scan_paths,
    synced_service,
)
from studio.tests.phantom import write_isq, write_segmentation  # noqa: E402


def test_compose_follows_the_lab_convention():
    meta = {"age": "P52", "line": "Odam-Cre", "genotype": "Hif1a cKO", "replicate": 2}
    assert compose_name(meta) == "P52_Odam-Cre_Hif1a-cKO_2"
    assert compose_name({"age": "P82", "genotype": "WT"}) == "P82_WT"
    # The sex goes between the genotype and the number, as F or M.
    assert compose_name({**meta, "sex": "F"}) == "P52_Odam-Cre_Hif1a-cKO_F_2"
    assert compose_name({**meta, "sex": "male"}) == "P52_Odam-Cre_Hif1a-cKO_M_2"
    assert compose_name({**meta, "sex": ""}) == "P52_Odam-Cre_Hif1a-cKO_2"
    # Underscores separate tokens, so they cannot appear inside one.
    assert compose_name({"line": "odam_cre", "genotype": "W/T"}) == "odamcre_WT"


def test_parse_reads_names_that_open_with_an_age():
    assert parse_name("P52_Odam-Cre_WT_1") == {"age": "P52", "line": "Odam-Cre", "genotype": "WT", "replicate": 1}
    assert parse_name("P82_Odam-Cre_CFTR-cKO_3")["genotype"] == "CFTR cKO"
    both = {"age": "P52", "line": "Odam-Cre", "genotype": "WT", "sex": "F", "replicate": 1}
    assert parse_name("P52_Odam-Cre_WT_F_1") == both
    # Spelled out, or written after the number, it is still the sex.
    assert parse_name("P52_Odam-Cre_WT_Female_1") == both and parse_name("P52_Odam-Cre_WT_1_F") == both
    assert parse_name("P82_Odam-Cre_Hif1a-cKO_M_2") == {"age": "P82", "line": "Odam-Cre", "genotype": "Hif1a cKO",
                                                     "sex": "M", "replicate": 2}
    assert parse_name("Odam-CRE-A") == {}
    assert parse_name("Frag_4") == {}


def test_group_is_genotype_by_age():
    assert group_of({"genotype": "WT", "age": "P52"}) == "WT P52"
    assert group_of({}) == "Ungrouped"


def test_classify_prefixed_and_legacy_outputs():
    assert classify_file(Path("S1_incisor_volume.nrrd")) == ("S1", "incisor_volume")
    assert classify_file(Path("S1_enamel_density_profile.csv")) == ("S1", "incisor_profile")
    assert classify_file(Path("S1_enamel_density_profile.png")) == ("S1", "incisor_plot_density")
    assert classify_file(Path("S1_molar_enamel_profile.csv")) == ("S1", "molar_profile")
    assert classify_file(Path("S1_molar_enamel.json")) == ("S1", "molar_quant")
    assert classify_file(Path("S1_tooth_labels.nrrd")) == ("S1", "tooth_labels")
    assert classify_file(Path("incisor_volume.tif")) == ("", "incisor_volume")
    assert classify_file(Path("._S1_incisor_volume.nrrd")) == (None, None)
    assert classify_file(Path("notes.txt")) == (None, None)


def test_workspace_create_rename_archive(tmp_path):
    ws = Workspace(tmp_path / "ws")
    meta = {"age": "P52", "line": "Odam-Cre", "genotype": "WT"}
    meta["replicate"] = ws.next_replicate(meta)
    first = ws.create(meta)
    assert first["name"] == "P52_Odam-Cre_WT_1"
    assert ws.next_replicate({k: v for k, v in meta.items() if k != "replicate"}) == 2
    # The same name again becomes the next replicate rather than an error.
    assert ws.create(meta)["name"] == "P52_Odam-Cre_WT_2"

    # Files the pipeline named after the sample follow a rename.
    folder = ws.sample_dir("P52_Odam-Cre_WT_1") / "quant" / "molar"
    folder.mkdir(parents=True)
    (folder / "P52_Odam-Cre_WT_1_molar_enamel.json").write_text("{}")
    ws.set_artifacts("P52_Odam-Cre_WT_1", {"molar_quant": folder / "P52_Odam-Cre_WT_1_molar_enamel.json"})
    renamed = ws.rename("P52_Odam-Cre_WT_1", "P52_Odam-Cre_WT_9")
    path = ws.artifact(renamed, "molar_quant")
    assert path is not None and path.name == "P52_Odam-Cre_WT_9_molar_enamel.json"
    assert not ws.sample_dir("P52_Odam-Cre_WT_1").exists()

    with pytest.raises(ValueError):
        ws.rename("P52_Odam-Cre_WT_9", "P52_Odam-Cre_WT_2")      # taken
    with pytest.raises(ValueError):
        ws.create({"genotype": "WT"}, name="bad name with spaces")

    dest = ws.archive("P52_Odam-Cre_WT_9")
    assert dest.exists() and "P52_Odam-Cre_WT_9" not in ws.names()


def test_linked_artifacts_stay_absolute_and_local_ones_relative(tmp_path):
    ws = Workspace(tmp_path / "ws")
    outside = tmp_path / "elsewhere.nrrd"
    outside.write_bytes(b"x")
    record = ws.create({"genotype": "WT"}, artifacts={"reoriented": str(outside)})
    stored = json.loads((ws.sample_dir(record["name"]) / "sample.json").read_text())
    assert os.path.isabs(stored["artifacts"]["reoriented"])
    inside = ws.sample_dir(record["name"]) / "cmpr" / "x_resliced_volume.nrrd"
    inside.parent.mkdir()
    inside.write_bytes(b"x")
    ws.set_artifacts(record["name"], {"cmpr_volume": inside})
    stored = json.loads((ws.sample_dir(record["name"]) / "sample.json").read_text())
    assert stored["artifacts"]["cmpr_volume"] == os.path.join("cmpr", "x_resliced_volume.nrrd")
    assert ws.artifact(record["name"], "cmpr_volume") == inside


def test_scan_matches_folders_and_loose_outputs_by_name(tmp_path):
    seg_root = tmp_path / "SEG DATA"
    write_segmentation(seg_root, "P52_Odam-Cre_WT_1", seed=1)
    cmpr_root = tmp_path / "CMPR"
    cmpr_root.mkdir()
    (cmpr_root / "P52_Odam-Cre_WT_1_resliced_volume.nrrd").write_bytes(b"x")
    (cmpr_root / "P52_Odam-Cre_WT_1_cmpr_transform.npz").write_bytes(b"x")
    raw_root = tmp_path / "RAW"
    raw_root.mkdir()
    write_isq(raw_root / "P82_Odam-Cre_WT_2.isq", shape=(40, 30, 20))

    assert detect_kind(seg_root / "P52_Odam-Cre_WT_1") == "segmentation"
    assert detect_kind(raw_root / "P82_Odam-Cre_WT_2.isq") == "isq"
    found = {c["name"]: c for c in scan_paths([str(seg_root), str(cmpr_root), str(raw_root)])}
    wt1 = found["P52_Odam-Cre_WT_1"]
    assert {"reoriented", "incisor_volume", "molar_volume", "cmpr_volume", "cmpr_transform"} <= set(wt1["artifacts"])
    assert wt1["meta"]["genotype"] == "WT"
    assert found["P82_Odam-Cre_WT_2"]["source"]["kind"] == "isq"
    assert artifacts_in(seg_root / "P52_Odam-Cre_WT_1", "P52_Odam-Cre_WT_1")["voxel_size"].endswith(".json")


def test_a_folder_of_several_scans_is_one_specimen_per_scan(tmp_path):
    raw = tmp_path / "RAW"
    (raw / "P52").mkdir(parents=True)
    write_isq(raw / "P52" / "C0004019.ISQ", shape=(20, 16, 12))
    write_isq(raw / "P52" / "C0004020.isq", shape=(20, 16, 12))
    (raw / "P52" / "._C0004019.ISQ").write_bytes(b"\0" * 64)   # macOS litter on an exFAT drive
    (raw / "P82_Odam-Cre_WT_1").mkdir()
    write_isq(raw / "P82_Odam-Cre_WT_1" / "C0004021.ISQ", shape=(20, 16, 12))
    found = {c["name"]: c for c in scan_paths([str(raw)])}
    assert set(found) == {"C0004019", "C0004020", "P82_Odam-Cre_WT_1"}
    # Each scan remembers the folders it sat in, for what their names say.
    assert found["C0004019"]["folders"] == ["P52", "RAW"]
    assert found["P82_Odam-Cre_WT_1"]["source"]["path"].endswith("C0004021.ISQ")
    assert found["P82_Odam-Cre_WT_1"]["folders"] == ["RAW"]


def test_any_text_becomes_a_usable_name():
    assert safe_name("Odam CRE (A)") == "Odam-CRE-A"
    assert safe_name("  __P52 WT__ ") == "P52-WT"
    assert safe_name("CON") == "sample-CON" and safe_name("µ…") == ""


def test_a_batch_is_named_as_it_will_be_imported(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.create({"age": "P52", "line": "Odam-Cre", "genotype": "WT", "replicate": 1})
    details = {"age": "P52", "line": "Odam-Cre", "genotype": "WT"}
    plan = ws.plan_names([
        # Two raw scans with the same details take the next free numbers in turn.
        {"key": "a", "found": "C0004019", "meta": details},
        {"key": "b", "found": "C0004020", "meta": {**details, "replicate": ""}},
        # Too little to compose a name: the found one, made safe.
        {"key": "c", "found": "Odam CRE (A)", "meta": {"genotype": "WT"}},
        # A name in the convention that says what its details say stays as found.
        {"key": "d", "found": "P52_Odam-Cre_Hif1a_cKO_2",
         "meta": {"age": "P52", "line": "Odam-Cre", "genotype": "Hif1a cKO", "replicate": 2}},
        # Details changed from what the found name says: the name follows them.
        {"key": "e", "found": "P60_Odam-Cre_WT_4", "meta": {"age": "P60", "line": "Odam-Cre", "genotype": "KO", "replicate": 4}},
        # A taken replicate moves to the next free one, and the number follows.
        {"key": "f", "found": "x", "meta": {**details, "replicate": 1}},
        # A typed name wins, and a bad one is reported rather than changed.
        {"key": "g", "found": "C0004030", "meta": details, "name": "Pilot-1"},
        {"key": "h", "found": "C0004031", "meta": details, "name": "no spaces allowed"},
    ])
    by = {p["key"]: p for p in plan}
    assert (by["a"]["name"], by["a"]["from"], by["a"]["replicate"]) == ("P52_Odam-Cre_WT_2", "details", 2)
    assert (by["b"]["name"], by["b"]["replicate"]) == ("P52_Odam-Cre_WT_3", 3)
    assert (by["c"]["name"], by["c"]["from"]) == ("Odam-CRE-A", "found")
    assert (by["d"]["name"], by["d"]["from"]) == ("P52_Odam-Cre_Hif1a_cKO_2", "found")
    assert by["e"]["name"] == "P60_Odam-Cre_KO_4"
    assert (by["f"]["name"], by["f"]["replicate"]) == ("P52_Odam-Cre_WT_4", 4)
    assert (by["g"]["name"], by["g"]["from"]) == ("Pilot-1", "typed")
    assert by["h"]["error"] and by["h"]["name"] == "no spaces allowed"


def test_replicates_are_numbered_across_the_sexes(tmp_path):
    ws = Workspace(tmp_path / "ws")
    group = {"age": "P52", "line": "Odam-Cre", "genotype": "WT"}
    ws.create({**group, "replicate": 1})                     # named before names carried the sex
    ws.create({**group, "sex": "F", "replicate": 2})
    assert ws.names() == ["P52_Odam-Cre_WT_1", "P52_Odam-Cre_WT_F_2"]
    # A male of the same group takes the group's next number, not a 1 of his own.
    assert ws.next_replicate({**group, "sex": "M"}) == 3
    plan = ws.plan_names([{"found": "C0004019", "meta": {**group, "sex": "M"}},
                          {"found": "C0004020", "meta": {**group, "sex": "F"}}])
    assert [p["name"] for p in plan] == ["P52_Odam-Cre_WT_M_3", "P52_Odam-Cre_WT_F_4"]
    # A found name in the old convention takes the sex once it is known.
    kept = ws.plan_names([{"found": "P60_Odam-Cre_KO_7", "meta": {"age": "P60", "line": "Odam-Cre", "genotype": "KO",
                                                                   "replicate": 7}}])[0]
    assert (kept["name"], kept["from"]) == ("P60_Odam-Cre_KO_7", "found")
    moved = ws.plan_names([{"found": "P60_Odam-Cre_KO_7", "meta": {"age": "P60", "line": "Odam-Cre", "genotype": "KO",
                                                                    "sex": "F", "replicate": 7}}])[0]
    assert moved["name"] == "P60_Odam-Cre_KO_F_7"
    # An existing specimen renamed to follow its details keeps its own number.
    again = ws.plan_names([{"found": "P52_Odam-Cre_WT_1", "meta": {**group, "sex": "M"}}], ignore=["P52_Odam-Cre_WT_1"])[0]
    assert again["name"] == "P52_Odam-Cre_WT_M_1"


def test_a_linked_scan_keeps_its_large_files_on_its_own_drive(tmp_path):
    ws = Workspace(tmp_path / "ws")
    raw = tmp_path / "drive" / "RAW"
    raw.mkdir(parents=True)
    isq = write_isq(raw / "C0004019.ISQ", shape=(20, 12, 10))
    meta = {"age": "P52", "line": "Odam-Cre", "genotype": "WT"}
    linked = ws.create(meta, source={"kind": "isq", "mode": "link", "path": str(isq)})
    assert linked["storage"] == str(raw / STORAGE_FOLDER / linked["name"])
    assert ws.storage_dir(linked) == raw / STORAGE_FOLDER / linked["name"]
    # A folder of slices gets its studio folder beside it, not inside the series.
    dicom = tmp_path / "drive" / "DICOM" / "P52_Odam-Cre_KO_1"
    dicom.mkdir(parents=True)
    series = ws.create(meta, source={"kind": "dicom", "mode": "link", "path": str(dicom)})
    assert ws.storage_dir(series) == dicom.parent / STORAGE_FOLDER / series["name"]
    # Uploaded or copied scans live in the workspace, and so do their large files.
    for mode in ("upload", "copy"):
        rec = ws.create(meta, source={"kind": "isq", "mode": mode, "path": ""})
        assert "storage" not in rec and ws.storage_dir(rec) is None
    inside = ws.root / "linked.isq"
    inside.write_bytes(b"")
    rec = ws.create(meta, source={"kind": "isq", "mode": "link", "path": str(inside)})
    assert ws.storage_dir(rec) is None


def test_rename_carries_the_large_files_along(tmp_path):
    ws = Workspace(tmp_path / "ws")
    raw = tmp_path / "drive" / "RAW"
    raw.mkdir(parents=True)
    isq = write_isq(raw / "C0004019.ISQ", shape=(20, 12, 10))
    rec = ws.create({"age": "P52", "genotype": "WT"}, source={"kind": "isq", "mode": "link", "path": str(isq)})
    old = rec["name"]
    storage = Path(rec["storage"])
    storage.mkdir(parents=True)
    (storage / f"{old}.nrrd").write_bytes(b"converted")
    (storage / f"{old}_reoriented_volume.nrrd").write_bytes(b"reoriented")
    local = ws.sample_dir(old) / "quant" / "molar" / f"{old}_molar_enamel.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}")
    ws.set_artifacts(old, {"converted": storage / f"{old}.nrrd",
                           "reoriented": storage / f"{old}_reoriented_volume.nrrd", "molar_quant": local})

    renamed = ws.rename(old, "P52_WT_7")
    moved = raw / STORAGE_FOLDER / "P52_WT_7"
    assert renamed["storage"] == str(moved) and not storage.exists()
    assert ws.artifact(renamed, "converted") == moved / "P52_WT_7.nrrd"
    assert ws.artifact(renamed, "reoriented") == moved / "P52_WT_7_reoriented_volume.nrrd"
    assert ws.artifact(renamed, "molar_quant").name == "P52_WT_7_molar_enamel.json"
    assert ws.artifact(renamed, "raw") == isq  # the scan itself is never renamed


def test_rename_updates_a_file_named_exactly_after_the_sample(tmp_path):
    ws = Workspace(tmp_path / "ws")
    rec = ws.create({"age": "P52", "genotype": "WT"})
    converted = ws.sample_dir(rec["name"]) / "converted" / f"{rec['name']}.nrrd"
    converted.parent.mkdir(parents=True)
    converted.write_bytes(b"x")
    ws.set_artifacts(rec["name"], {"converted": converted})
    renamed = ws.rename(rec["name"], "P52_WT_9")
    assert ws.artifact(renamed, "converted") == ws.sample_dir("P52_WT_9") / "converted" / "P52_WT_9.nrrd"


def test_folder_paths_are_written_one_way():
    assert folder_path("  Odam  study / Batch 2/ ") == "Odam study/Batch 2"
    assert folder_path("") == folder_path(None) == folder_path("/") == ""
    assert folder_path("tab\there") == "tab here"
    for bad in ("..", "a/./b", "back\\slash", "bell\x07", "x" * 81, "/".join("abcdefghi")):
        with pytest.raises(ValueError):
            folder_path(bad)


def test_library_folders_sort_specimens_without_moving_them(tmp_path):
    ws = Workspace(tmp_path / "ws")
    for n in ("P52_WT_1", "P52_WT_2", "P52_KO_1", "P82_WT_1"):
        ws.create({}, name=n)
    ws.create({}, name="P52_KO_2", folder=" Odam study / Batch 1 ")
    assert ws.folders() == ["Odam study", "Odam study/Batch 1"]
    assert ws.get("P52_KO_2")["folder"] == "Odam study/Batch 1"

    # An empty folder stays until it is removed; a folder typed in another case is the one already there.
    assert ws.add_folder("odam study/Batch 2") == "Odam study/Batch 2"
    with pytest.raises(FolderExists):
        ws.add_folder("ODAM STUDY")
    updated = ws.get("P52_WT_1")["updated"]
    assert ws.move_to_folder(["P52_WT_1", "P52_WT_2"], "odam study/batch 2") == "Odam study/Batch 2"
    assert ws.move_to_folder(["P52_KO_1"], "Odam study") == "Odam study"
    # Filing a specimen is not a change to it, and its own folder on disk stays where it was.
    assert ws.get("P52_WT_1")["updated"] == updated
    assert (ws.samples_dir / "P52_WT_1" / "sample.json").exists()
    with pytest.raises(KeyError):
        ws.move_to_folder(["P52_WT_1", "Nobody_1"], "Elsewhere")
    assert "Elsewhere" not in ws.folders()

    # Renaming a folder carries what it holds; moving one into another is renaming it.
    assert ws.rename_folder("Odam study", "UCSF") == ("UCSF", ["P52_KO_1", "P52_KO_2", "P52_WT_1", "P52_WT_2"])
    assert ws.folders() == ["UCSF", "UCSF/Batch 1", "UCSF/Batch 2"]
    new, moved = ws.rename_folder("UCSF/Batch 2", "Archive/Batch 2")
    assert (new, sorted(moved)) == ("Archive/Batch 2", ["P52_WT_1", "P52_WT_2"])
    assert ws.get("P52_WT_2")["folder"] == "Archive/Batch 2"
    with pytest.raises(ValueError, match="inside itself"):
        ws.rename_folder("Archive", "archive/Batch 2/Deeper")
    with pytest.raises(FolderExists):
        ws.rename_folder("Archive", "ucsf")
    assert ws.rename_folder("archive", "ARCHIVE")[0] == "ARCHIVE"  # a change of case alone

    # Removing a folder removes no specimen: what it held moves up a level, its folders keeping theirs.
    assert sorted(ws.remove_folder("UCSF")) == ["P52_KO_1", "P52_KO_2"]
    assert ws.folders() == ["ARCHIVE", "ARCHIVE/Batch 2", "Batch 1"]
    assert "folder" not in ws.get("P52_KO_1") and ws.get("P52_KO_2")["folder"] == "Batch 1"
    assert sorted(ws.names()) == ["P52_KO_1", "P52_KO_2", "P52_WT_1", "P52_WT_2", "P82_WT_1"]

    # Back at the top of the library, a specimen has no folder at all, and a rename keeps its folder.
    ws.move_to_folder(["P52_WT_1"], "")
    assert "folder" not in ws.get("P52_WT_1")
    assert ws.rename("P52_WT_2", "P52_WT_9")["folder"] == "ARCHIVE/Batch 2"
    stored = json.loads((ws.root / "studio.json").read_text())["folders"]
    assert stored == ["ARCHIVE", "ARCHIVE/Batch 2", "Batch 1"]


def test_folders_that_sync_to_the_cloud_are_recognised():
    home = Path.home()
    assert synced_service(home / "Library" / "CloudStorage" / "Box-Box" / "Lab" / "RAW") == "Box"
    assert synced_service(home / "Library" / "Mobile Documents" / "com~apple~CloudDocs") == "iCloud Drive"
    assert synced_service("/Users/someone/Dropbox/scans") == "Dropbox"
    assert synced_service("/Volumes/Z Drive/RAW") is None


@pytest.fixture
def fake_drives(monkeypatch):
    """Drives by label, mounted wherever a test says: a Mac's /Volumes/X or a PC's E:\\."""
    mounts: dict[str, Path] = {}

    def root_of(path):
        real = Path(os.path.realpath(path))
        for label, root in mounts.items():
            r = Path(os.path.realpath(root))
            if real == r or r in real.parents:
                return label, r
        return None

    monkeypatch.setattr(drives, "volume_of", root_of)
    monkeypatch.setattr(drives, "find_volume", lambda label: mounts.get(label))
    monkeypatch.setattr(drives, "drive_root", lambda path: (root_of(path) or (None, Path(os.sep)))[1])
    monkeypatch.setattr(drives, "mounted_volumes", lambda: dict(mounts))
    return mounts


def test_a_workspace_on_a_drive_opens_wherever_the_drive_is_mounted(tmp_path, fake_drives):
    drive = tmp_path / "mac" / "LabDrive"
    fake_drives["LabDrive"] = drive
    raw = drive / "RAW"
    raw.mkdir(parents=True)
    isq = write_isq(raw / "C0004019.ISQ", shape=(20, 12, 10))
    ws = Workspace(drive / "MicroCT-Studio")
    rec = ws.create({"age": "P52", "genotype": "WT"}, source={"kind": "isq", "mode": "link", "path": str(isq)})
    name = rec["name"]
    stored = json.loads((ws.sample_dir(name) / "sample.json").read_text())
    assert stored["source"]["path"] == stored["artifacts"]["raw"] == "@vol/LabDrive/RAW/C0004019.ISQ"
    assert stored["storage"] == f"@vol/LabDrive/RAW/{STORAGE_FOLDER}/{name}"

    # The same drive plugged into another computer, which mounts it elsewhere.
    elsewhere = tmp_path / "pc" / "E"
    elsewhere.parent.mkdir()
    shutil.move(str(drive), str(elsewhere))
    fake_drives["LabDrive"] = elsewhere
    ws = Workspace(elsewhere / "MicroCT-Studio")
    rec = ws.get(name)
    assert ws.artifact(rec, "raw") == ws.source_path(rec) == elsewhere / "RAW" / "C0004019.ISQ"
    assert ws.storage_dir(rec) == elsewhere / "RAW" / STORAGE_FOLDER / name
    # Unplugged: nothing resolves, rather than something wrong.
    fake_drives.clear()
    assert ws.artifact(rec, "raw") is None and ws.storage_dir(rec) is None


def test_absolute_paths_in_old_records_become_portable_when_opened(tmp_path, fake_drives):
    drive = tmp_path / "LabDrive"
    fake_drives["LabDrive"] = drive
    (drive / "RAW").mkdir(parents=True)
    isq = write_isq(drive / "RAW" / "C0004019.ISQ", shape=(20, 12, 10))
    ws = Workspace(tmp_path / "home" / "MicroCT-Studio")
    rec = ws.create({"age": "P52", "genotype": "WT"})
    local = ws.sample_dir(rec["name"]) / "converted" / "x.nrrd"
    local.parent.mkdir()
    local.write_bytes(b"x")
    # As an older studio wrote it: absolute paths, and Windows separators.
    legacy = {**rec, "source": {"kind": "isq", "mode": "link", "path": str(isq)},
              "artifacts": {"raw": str(isq), "converted": "converted\\x.nrrd"}}
    (ws.sample_dir(rec["name"]) / "sample.json").write_text(json.dumps(legacy))
    reopened = Workspace(ws.root)
    stored = json.loads((ws.sample_dir(rec["name"]) / "sample.json").read_text())
    assert stored["source"]["path"] == stored["artifacts"]["raw"] == "@vol/LabDrive/RAW/C0004019.ISQ"
    assert stored["artifacts"]["converted"] == "converted/x.nrrd"
    assert stored["updated"] == legacy["updated"]
    assert reopened.artifact(reopened.get(rec["name"]), "converted") == local


def test_names_windows_cannot_use_are_refused():
    assert valid_name("P52_Odam-Cre_WT_1")
    for bad in ("CON", "nul", "com1.nrrd", "P52_WT_1.", "a/b", "P52 WT"):
        assert not valid_name(bad), bad


def test_workspaces_are_found_on_connected_drives(tmp_path, fake_drives, monkeypatch):
    fake_drives["LabDrive"] = tmp_path / "LabDrive"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    (tmp_path / "home").mkdir()
    ws = Workspace(tmp_path / "LabDrive" / "Lab" / "MicroCT-Studio")
    ws.create({"age": "P52", "genotype": "WT"})
    (tmp_path / "LabDrive" / "Other").mkdir()
    assert is_workspace(ws.root) and not is_workspace(tmp_path / "LabDrive" / "Other")
    found = [os.path.realpath(p) for p in find_workspaces()]
    assert found == [os.path.realpath(ws.root)]
    summary = workspace_summary(ws.root)
    assert summary["samples"] == 1 and summary["drive"] == "LabDrive" and summary["id"] == ws.id


def test_results_projects_keep_their_specimens_through_a_rename(tmp_path):
    ws = Workspace(tmp_path / "ws")
    a = ws.create({"age": "P52", "genotype": "WT", "replicate": 1})["name"]
    b = ws.create({"age": "P52", "genotype": "KO", "replicate": 1})["name"]
    odam = ws.create_project({"name": "  Odam   study ", "names": [a, b, a], "by": ["genotype", "age"]})
    assert odam["name"] == "Odam study" and odam["names"] == [a, b], "names trimmed, specimens once each"
    assert odam["hidden"] == [] and odam["left"] == [] and odam["created"] == odam["updated"]
    other = ws.create_project({"name": "Sexes", "names": [b]})
    with pytest.raises(ProjectExists):
        ws.create_project({"name": "odam STUDY"})
    with pytest.raises(ProjectExists):
        ws.update_project(other["id"], {"name": "Odam study"})
    with pytest.raises(ValueError):
        ws.create_project({"name": " "})
    with pytest.raises(ValueError):
        ws.update_project(odam["id"], {"names": "P52_WT_1"})
    with pytest.raises(KeyError):
        ws.update_project("0" * 12, {"name": "Gone"})

    # A rename of the project keeps its specimens; a rename of a specimen keeps its projects.
    assert ws.update_project(odam["id"], {"name": "Odam", "hidden": ["sex\u0001M"], "left": [a]})["names"] == [a, b]
    ws.rename(a, "P52_WT_9")
    by_id = {p["id"]: p for p in ws.projects()}
    assert by_id[odam["id"]]["names"] == ["P52_WT_9", b] and by_id[odam["id"]]["hidden"] == ["sex\u0001M"]
    assert by_id[odam["id"]]["left"] == ["P52_WT_9"], "an unticked specimen stays unticked under its new name"
    assert by_id[other["id"]]["left"] == []
    assert by_id[other["id"]]["names"] == [b]
    ws.remove_project(other["id"])
    assert [p["name"] for p in Workspace(tmp_path / "ws").projects()] == ["Odam"], "kept in studio.json"
