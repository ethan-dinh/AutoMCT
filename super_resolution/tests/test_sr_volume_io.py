"""
Tests for reading scans and writing the upscaled volume.

The upscaled NRRD is only useful if segment_mandible reads it back as the
same array in the same orientation, and if it lands exactly on top of the
original in physical space. Both are checked through segment_mandible's own
loader, not this package's.

Run with:  python -m pytest super_resolution/tests
"""

# pylint: disable=wrong-import-position

import sr_phantoms  # noqa: F401  (puts the repository on sys.path)

import nrrd
import numpy as np
import pytest
from PIL import Image

from data_io import load_nrrd  # segment_mandible's loader

from super_resolution.volume_io import (
    Geometry,
    IntensityScale,
    close_memmap,
    create_nrrd_memmap,
    discover_samples,
    load_scan,
    specimen_bounding_box,
    write_nrrd,
)

OBLIQUE = Geometry(
    directions=np.array([[0.0, 0.1, 0.0], [0.0, 0.0, 0.05], [0.02, 0.03, 0.0]]),
    origin=np.array([1.5, -2.0, 30.0]),
)


def _corner(geometry: Geometry, index) -> np.ndarray:
    """Physical position of a voxel-grid point (voxel centres at integers)."""
    return geometry.origin + np.asarray(index, dtype=float) @ geometry.directions


@pytest.mark.parametrize("scale", [(2, 2, 2), (2, 3, 1)])
def test_upscaled_grid_shares_the_original_outer_boundary(scale):
    fine = OBLIQUE.upscaled(scale)
    shape = np.array([5, 6, 7])
    assert np.allclose(_corner(OBLIQUE, [-0.5] * 3), _corner(fine, [-0.5] * 3))
    assert np.allclose(_corner(OBLIQUE, shape - 0.5), _corner(fine, shape * np.array(scale) - 0.5))
    assert np.allclose(fine.spacing, np.array(OBLIQUE.spacing) / np.array(scale))


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.uint8, np.float32])
def test_memmapped_nrrd_reads_back_through_segment_mandible(tmp_path, dtype):
    rng = np.random.default_rng(0)
    data = (rng.random((5, 6, 7)) * 200).astype(dtype)
    path = str(tmp_path / "volume.nrrd")

    out = create_nrrd_memmap(path, data.shape, dtype, OBLIQUE)
    out[:] = data
    close_memmap(out)
    del out

    loaded, spacing = load_nrrd(path, return_spacing=True)
    assert loaded.shape == data.shape
    assert np.array_equal(loaded, data)
    assert np.allclose(spacing, OBLIQUE.spacing)
    header = Geometry.from_nrrd_header(nrrd.read_header(path))
    assert np.allclose(header.directions, OBLIQUE.directions)
    assert np.allclose(header.origin, OBLIQUE.origin)


def test_write_nrrd_round_trip(tmp_path):
    labels = np.random.default_rng(1).integers(0, 4, size=(4, 5, 6)).astype(np.uint8)
    path = str(tmp_path / "labels.nrrd")
    write_nrrd(path, labels, OBLIQUE)
    assert np.array_equal(load_nrrd(path), labels)
    header = Geometry.from_nrrd_header(nrrd.read_header(path))
    assert np.allclose(header.directions, OBLIQUE.directions)


def test_load_scan_keeps_geometry_and_applies_an_override(tmp_path):
    data = np.random.default_rng(2).integers(0, 1000, size=(6, 7, 8)).astype(np.uint16)
    path = str(tmp_path / "scan.nrrd")
    write_nrrd(path, data, OBLIQUE)

    scan = load_scan(path)
    assert scan.name == "scan" and scan.spacing_from_file
    assert np.array_equal(scan.data, data)
    assert np.allclose(scan.geometry.directions, OBLIQUE.directions)

    overridden = load_scan(path, voxel_size=(0.2, 0.2, 0.2))
    assert np.allclose(overridden.geometry.spacing, (0.2, 0.2, 0.2))
    unit = OBLIQUE.directions / np.linalg.norm(OBLIQUE.directions, axis=1, keepdims=True)
    assert np.allclose(overridden.geometry.directions / 0.2, unit)


def test_bmp_stack_loads_without_spacing(tmp_path):
    folder = tmp_path / "Sample_01"
    folder.mkdir()
    for i in range(3):
        Image.fromarray(np.full((4, 5), 10 * i, dtype=np.uint8)).save(folder / f"slice_{i:04d}.bmp")
    scan = load_scan(str(folder))
    assert scan.data.shape == (3, 4, 5)
    assert not scan.spacing_from_file
    assert scan.geometry.spacing == (1.0, 1.0, 1.0)
    assert scan.name == "Sample_01"


def test_intensity_scale_round_trip_and_robustness():
    volume = np.random.default_rng(3).normal(1000, 100, size=(20, 20, 20)).astype(np.float32)
    volume[0, 0, 0] = 1e6  # one saturated voxel must not set the range
    scale = IntensityScale.fit(volume)
    assert scale.high < 2000
    unit = scale.to_unit(volume)
    assert np.allclose(scale.from_unit(unit), volume, rtol=1e-5)


def test_specimen_box_ignores_noise_specks_but_keeps_the_specimen():
    volume = np.zeros((60, 60, 60), dtype=np.uint16)
    volume[20:40, 15:35, 25:45] = 100
    volume[2, 2, 2] = 100
    box = specimen_bounding_box(volume, threshold=50, margin=3)
    assert (box[0].start, box[0].stop) == (16, 44)
    assert (box[1].start, box[1].stop) == (11, 39)
    assert (box[2].start, box[2].stop) == (21, 49)


def test_discover_samples_follows_main_py(tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "Compressed").mkdir()
    (tmp_path / ".hidden").mkdir()
    write_nrrd(str(tmp_path / "B.nrrd"), np.zeros((2, 2, 2), np.uint8), OBLIQUE)
    assert discover_samples(str(tmp_path)) == [("A", str(tmp_path / "A")), ("B", str(tmp_path / "B.nrrd"))]
    assert discover_samples(str(tmp_path / "B.nrrd")) == [("B", str(tmp_path / "B.nrrd"))]
