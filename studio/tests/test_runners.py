"""
The runners' own logic, apart from the tools they call.

Run with:  python -m pytest studio/tests/test_runners.py
"""

import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio.runners.segment import LOADERS, crop_loaders, redirect_reoriented  # noqa: E402


def test_an_nrrd_is_cropped_as_it_loads():
    volume = np.arange(10 * 12 * 14, dtype=np.int16).reshape(10, 12, 14)
    module = types.SimpleNamespace(
        load_nrrd=lambda path, return_spacing=False: (volume, (0.01, 0.01, 0.01)),
        load_tiff=lambda path, return_spacing=False: (None, None),
        load_dicom_series=lambda path, **kwargs: (volume, (0.02, 0.02, 0.02)),
        load_bmp_stack=lambda path, **kwargs: volume,
    )
    # The far x edge lies past the volume, as a box drawn on a strided preview can.
    crop_loaders(module, [2, 8, 0, 12, 3, 20])
    data, spacing = module.load_nrrd("scan.nrrd", return_spacing=True)
    assert data.shape == (6, 12, 11) and spacing == (0.01, 0.01, 0.01)
    assert np.array_equal(data, volume[2:8, :, 3:])
    assert data.flags["C_CONTIGUOUS"]
    assert module.load_bmp_stack("slices").shape == (6, 12, 11)
    assert module.load_tiff("scan.tif") == (None, None)  # a failed load stays a failed load


def test_the_pipeline_still_loads_through_the_wrapped_names():
    import pipeline  # segment_mandible/pipeline.py; the runner puts it on the path

    assert all(callable(getattr(pipeline, name, None)) for name in LOADERS)


def test_only_the_reoriented_volume_goes_beside_the_scan(tmp_path):
    written = []

    def save_volume(volume, output_dir, name, base_dir=None, output_format="nrrd", spacing=None):
        written.append((base_dir, output_dir, name))
        return f"{base_dir}/{output_dir}/{name}.{output_format}"

    main = types.SimpleNamespace(save_volume=save_volume)
    beside = tmp_path / "RAW" / "MicroCT-Studio" / "S1"
    redirect_reoriented(main, str(beside))
    main.save_volume(None, "S1", "S1_reoriented_volume", base_dir="/ws/seg", output_format="nrrd")
    main.save_volume(None, "S1/volumes", "S1_incisor_volume", base_dir="/ws/seg")
    main.save_volume(None, "S1", "S1_reoriented_volume", "/ws/seg")  # base_dir given positionally
    assert written == [
        (str(beside.parent), "S1", "S1_reoriented_volume"),
        ("/ws/seg", "S1/volumes", "S1_incisor_volume"),
        (str(beside.parent), "S1", "S1_reoriented_volume"),
    ]


def test_main_still_saves_through_the_redirected_name():
    import main  # segment_mandible/main.py

    assert callable(main.save_volume) and "save_volume(bmp_data" in Path(main.__file__).read_text()


# The lab's conversion, as it is typed at the command line.
LAB_RECIPE = ["scan.isq", "--autocrop", "--target-voxel-um", "6", "--gauss-sigma", "0.15", "--nlm-h", "0.3",
              "--median-radius", "0", "--aniso-iterations", "10", "--aniso-kappa", "600", "--tv-weight", "0.15",
              "--unsharp-amount", "0.04", "--reorient", "--reorient-tight", "--reorient-margin-mm", "0.5",
              "--save-raw", "--save-dtype", "int16", "--autocrop-margin-mm", "0.1", "--clamp-negative"]


def test_the_convert_step_defaults_to_the_labs_recipe():
    import dataclasses

    from studio import steps
    from studio.paths import TOOLS
    sys.path.insert(0, str(TOOLS))
    from convert_ISQ_NRRD import _build_parser  # noqa: PLC0415
    from volume_pipeline import options_from_args  # noqa: PLC0415

    # Read by the converter's own command line, so a flag means what it means there.
    args = _build_parser().parse_args(LAB_RECIPE)
    recipe = {**dataclasses.asdict(options_from_args(args)),
              "trim_saturated": args.trim_saturated, "clamp_negative": args.clamp_negative}
    defaults = steps.resolve_params(steps.CONVERT_PARAMS, {})
    assert {k: v for k, v in defaults.items() if recipe.get(k) != v} == {}
    assert set(defaults) <= set(recipe)


def test_an_unfiltered_copy_is_reused_only_for_the_geometry_it_was_made_with(tmp_path):
    from studio.runners.convert import set_aside_stale_copy

    geometry = {"autocrop": True, "autocrop_margin_mm": 0.1, "crop_box": None, "source": "f0004030.isq"}
    copy, note = tmp_path / "S_prefiltered.nrrd", tmp_path / "S_prefiltered.studio.json"
    assert not set_aside_stale_copy(tmp_path, "S", geometry)          # nothing there yet

    copy.write_bytes(b"x")
    note.write_text(json.dumps(geometry))
    assert not set_aside_stale_copy(tmp_path, "S", dict(geometry)) and copy.exists()

    # Another crop margin: the copy keeps the old crop, so it goes.
    assert set_aside_stale_copy(tmp_path, "S", {**geometry, "autocrop_margin_mm": 1.0})
    assert not copy.exists() and not note.exists()

    # A copy with no note -- from a run that failed after writing it -- is not trusted.
    copy.write_bytes(b"x")
    assert set_aside_stale_copy(tmp_path, "S", geometry) and not copy.exists()


def test_the_marks_keep_who_placed_them(tmp_path):
    """The marks file says which landmarks were found and which set by hand; the report says where each was found."""
    from studio import steps

    marks = [{"landmark": "eruption", "z_start": 4.0, "z_end": None, "source": "manual"},
             {"landmark": "M1", "z_start": 10.0, "z_end": 15.0, "source": "auto"}]
    path = tmp_path / "S1_marks.csv"
    steps.write_marks(path, marks)
    assert steps.read_marks(path) == marks
    path.write_text("landmark,kind,z_start,z_end\nM1,span,10.0,15.0\n")
    assert steps.read_marks(path)[0]["source"] == "manual", "marks from before sources were picked by hand"

    report = {"detected": [{"landmark": "eruption", "z_start": 1621.0, "z_end": None, "source": "auto"},
                           {"landmark": "M1", "z_start": 1216.0, "z_end": 975.0, "source": "auto"},
                           {"landmark": "broken"}]}
    assert steps.detected_marks(report) == [
        {"landmark": "eruption", "z_start": 1621.0, "z_end": None, "source": "auto"},
        {"landmark": "M1", "z_start": 1216.0, "z_end": 975.0, "source": "auto"}]
    assert steps.detected_marks(None) == []


def test_the_landmarks_need_only_the_segmentation_and_the_straightening(tmp_path):
    """Not the molar or incisor steps: the landmarks are found from the segmentation's own volumes."""
    from studio import steps

    class Workspace:
        def __init__(self, keys):
            self.files = {k: tmp_path / k for k in keys}

        def artifact(self, _record, key):
            return self.files.get(key)

    record = {"name": "S1"}
    assert steps.STEPS["landmarks"].upstream == ("segment", "cmpr")
    assert steps.blocked_reason(Workspace(()), record, "landmarks") == "Needs the segmentation"
    ws = Workspace(("incisor_volume", "molar_volume"))
    assert steps.blocked_reason(ws, record, "landmarks") == "Needs the straightened incisor's transform"
    assert steps.can_run_after(ws, record, "landmarks", {"cmpr"})
    assert not steps.can_run_after(ws, record, "landmarks", {"molar_quant", "incisor_quant"})
    assert steps.can_run_after(Workspace(()), record, "landmarks", {"segment", "cmpr"})

    ready = Workspace(("incisor_volume", "molar_volume", "cmpr_transform"))
    assert steps.blocked_reason(ready, record, "landmarks") is None
    # Run with the molar step, the landmarks do not wait for it.
    assert steps.plan(ready, record, ["landmarks", "molar_quant"]) == [("molar_quant", []), ("landmarks", [])]
