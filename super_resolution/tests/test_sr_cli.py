"""
Tests for the command line: argument parsing, and each subcommand run once on
a tiny phantom (tiny network, CPU) so a typo in the wiring fails here rather
than an hour into a run.

Run with:  python -m pytest super_resolution/tests
"""

# pylint: disable=wrong-import-position

import argparse
import os
import subprocess
import sys

import sr_phantoms as P

import numpy as np
import pytest

from data_io import load_nrrd  # segment_mandible

from super_resolution import cli
from super_resolution import sr_segment as S
from super_resolution.volume_io import Geometry, write_nrrd

TINY = [
    "--device", "cpu", "--features", "4", "--growth", "4", "--blocks", "1",
    "--lr-patch", "4", "--batch-size", "2", "--pretrain-iters", "2", "--gan-iters", "1",
    "--val-every", "10", "--checkpoint-every", "10", "--quiet",
]


def test_parse_scale_and_voxel_size():
    assert cli.parse_scale("2") == (2, 2, 2)
    assert cli.parse_scale("1,2,2") == (1, 2, 2)
    assert cli.parse_voxel_size("0.008") == (0.008, 0.008, 0.008)
    for bad in ("0", "2,2", "a"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_scale(bad)


def test_check_passes_on_the_cpu(capsys):
    assert cli.main(["check", "--device", "cpu", "--quiet"]) == 0
    assert "All checks passed." in capsys.readouterr().out


def test_train_upscale_and_segment_from_the_command_line(tmp_path, monkeypatch):
    volume, _ = P.jaw_phantom(body_value=1000, tip_value=3000, air_value=100, dtype=np.uint16)
    scan = str(tmp_path / "jaw.nrrd")
    write_nrrd(scan, volume, Geometry.from_spacing((0.016, 0.016, 0.016)))
    model_dir = str(tmp_path / "model")

    assert cli.main(["train", "-i", scan, "-o", model_dir, *TINY]) == 0
    model = os.path.join(model_dir, "model.pt")
    assert os.path.exists(model)

    fine = str(tmp_path / "jaw_x2.nrrd")
    assert cli.main(["upscale", "-i", scan, "-m", model, "-o", fine, "--tile", "8", "--device", "cpu", "--quiet"]) == 0
    assert load_nrrd(fine).shape == tuple(2 * n for n in volume.shape)

    calls = []

    def fake_pipeline(path, *, spacing_out, **_kwargs):
        calls.append(path)
        loaded = load_nrrd(path)
        spacing_out["record"] = {}
        return loaded, loaded, np.where(loaded > 2000, loaded, 0), np.where(
            (loaded > 500) & (loaded <= 2000), loaded, 0), np.zeros_like(loaded)

    monkeypatch.setattr(S, "load_classical_pipeline", lambda: fake_pipeline)
    out = str(tmp_path / "results")
    code = cli.main(["segment", "-i", scan, "-m", model, "-o", out, "--tile", "8",
                     "--device", "cpu", "--format", "nrrd", "--quiet"])
    assert code == 0 and len(calls) == 1
    assert load_nrrd(os.path.join(out, "jaw", "masks", "jaw_incisor_mask.nrrd")).shape == volume.shape


def test_pipeline_imports_without_napari():
    """The Docker image has no napari; segment_mandible must still import there."""
    code = "\n".join([
        "import sys",
        "sys.modules['napari'] = None  # makes `import napari` fail, as in the image",
        "from super_resolution.classical import load_classical_pipeline",
        "load_classical_pipeline()",
        "import visualization",
        "assert getattr(visualization, '__file__', None) is None, 'expected the stand-in'",
        "try:",
        "    visualization.create_3d_visualization(None)",
        "except RuntimeError as exc:",
        "    assert 'napari' in str(exc)",
        "else:",
        "    raise AssertionError('a viewer call should fail with a clear message')",
        "print('headless ok')",
    ])
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=P.REPO, capture_output=True, text=True, timeout=600, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "headless ok" in result.stdout
