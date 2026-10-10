"""
Tests for tiled inference.

Tiling must be invisible: a volume processed in small tiles has to match the
same volume processed in one pass. The output must keep the input's exact
minimum and maximum (segment_mandible min-max normalises, so anything else
would move every threshold), and all-air tiles must be exact replicas.

Run with:  python -m pytest super_resolution/tests
"""

# pylint: disable=wrong-import-position

import logging

import sr_phantoms  # noqa: F401  (puts the repository on sys.path)

import nrrd
import numpy as np
import pytest
import torch
from scipy import ndimage
from torch import nn

from data_io import load_nrrd  # segment_mandible's loader

from super_resolution import upscale as U
from super_resolution.model import SRGenerator3d
from super_resolution.volume_io import Geometry

CPU = torch.device("cpu")


def _generator(scale=2, blocks=0, detail=0.2) -> SRGenerator3d:
    torch.manual_seed(0)
    generator = SRGenerator3d(scale=scale, features=6, growth=4, blocks=blocks)
    nn.init.normal_(generator.tail.weight, std=detail)
    return generator.eval()


def _smooth_volume(shape=(20, 18, 22), seed=0, dtype=np.float32) -> np.ndarray:
    noise = np.random.default_rng(seed).random(shape)
    volume = ndimage.gaussian_filter(noise, 1.5)
    volume = (volume - volume.min()) / (volume.max() - volume.min()) * 1000
    return volume.astype(dtype)


def _run(volume, generator, tile, halo, dtype=np.float32, skip_below=None):
    shape = tuple(n * s for n, s in zip(volume.shape, generator.scale))
    out = np.zeros(shape, dtype=dtype)
    stats = U.upscale_volume(
        volume, generator, out, device=CPU, tile=tile, halo=halo,
        skip_below=skip_below, progress=False,
    )
    return out, stats


@pytest.mark.parametrize("scale", [(2, 2, 2), (1, 2, 3)])
def test_tiles_match_a_single_pass(scale):
    generator = _generator(scale)
    volume = _smooth_volume()
    whole, stats_whole = _run(volume, generator, tile=1000, halo=0)
    tiled, stats_tiled = _run(volume, generator, tile=7, halo=generator.receptive_radius)
    assert stats_whole["tiles"] == 1 and stats_tiled["tiles"] > 8
    assert np.allclose(tiled, whole, atol=1e-3)


def test_output_keeps_the_input_range_exactly_and_block_means_where_unclipped():
    generator = _generator(detail=0.01)
    volume = _smooth_volume(dtype=np.float32)
    out, _ = _run(volume, generator, tile=8, halo=generator.receptive_radius)
    assert out.min() == volume.min() and out.max() == volume.max()

    z, y, x = volume.shape
    blocks = out.reshape(z, 2, y, 2, x, 2)
    clipped = ((blocks <= volume.min()) | (blocks >= volume.max())).any(axis=(1, 3, 5))
    means = blocks.mean(axis=(1, 3, 5))
    assert (~clipped).mean() > 0.9
    assert np.allclose(means[~clipped], volume[~clipped], atol=1e-2)


def test_integer_output_is_rounded_and_keeps_the_range():
    generator = _generator()
    volume = _smooth_volume(dtype=np.uint16) + 100
    out, _ = _run(volume, generator, tile=8, halo=4, dtype=np.uint16)
    assert out.dtype == np.uint16
    assert out.min() == volume.min() and out.max() == volume.max()


def test_air_tiles_are_replicated_exactly():
    generator = _generator()
    rng = np.random.default_rng(1)
    volume = (rng.normal(100, 5, size=(16, 16, 32))).astype(np.float32)
    volume[:, :, 16:] = 1000 + rng.normal(0, 5, size=(16, 16, 16))  # tissue in one half
    ceiling = U.air_ceiling(volume)
    assert ceiling == pytest.approx(550, abs=5)

    out, stats = _run(volume, generator, tile=8, halo=2, skip_below=ceiling)
    assert 0 < stats["tiles_skipped_as_air"] < stats["tiles"]
    air_tile = volume[0:8, 0:8, 0:8]
    assert np.array_equal(out[0:16, 0:16, 0:16], np.repeat(np.repeat(np.repeat(air_tile, 2, 0), 2, 1), 2, 2))


def test_a_sliver_of_tissue_keeps_a_tile_on_the_network():
    generator = _generator()
    volume = np.full((16, 16, 16), 100.0, dtype=np.float32)
    volume[:, :, 8:] = 1000.0
    volume[2, 2, 2] = 1000.0  # one bright voxel inside an otherwise-air tile
    _, stats = _run(volume, generator, tile=8, halo=0, skip_below=U.air_ceiling(volume))
    assert stats["tiles_skipped_as_air"] == 3  # of the 4 air tiles, not the one with the voxel


def test_upscale_to_nrrd_overlays_the_original(tmp_path):
    generator = _generator()
    volume = _smooth_volume(dtype=np.uint16)
    geometry = Geometry(np.diag([0.02, 0.01, 0.01]), np.array([5.0, 6.0, 7.0]))
    path = str(tmp_path / "fine.nrrd")
    stats = U.upscale_to_nrrd(volume, geometry, generator, path, device=CPU, tile=8, halo=4, progress=False)

    loaded, spacing = load_nrrd(path, return_spacing=True)
    assert loaded.shape == tuple(2 * n for n in volume.shape)
    assert np.allclose(spacing, (0.01, 0.005, 0.005))
    header = Geometry.from_nrrd_header(nrrd.read_header(path))
    assert np.allclose(header.origin, geometry.upscaled((2, 2, 2)).origin)
    assert stats["bytes"] == loaded.nbytes


def test_upscale_to_nrrd_refuses_without_disk_space(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "free_disk_bytes", lambda _path: 0)
    path = tmp_path / "fine.nrrd"
    with pytest.raises(OSError, match="GB free"):
        U.upscale_to_nrrd(_smooth_volume(), Geometry.from_spacing((1, 1, 1)), _generator(), str(path),
                          device=CPU, progress=False)
    assert not path.exists()


def test_a_failed_upscale_removes_its_partial_file(tmp_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(U, "upscale_volume", explode)
    path = tmp_path / "fine.nrrd"
    with pytest.raises(RuntimeError):
        U.upscale_to_nrrd(_smooth_volume(), Geometry.from_spacing((1, 1, 1)), _generator(), str(path),
                          device=CPU, progress=False)
    assert not path.exists()


def test_resolution_mismatch_is_reported(caplog):
    checkpoint = {"trained_on": [{"spacing_mm_zyx": [0.008, 0.008, 0.008]}]}
    with caplog.at_level(logging.WARNING):
        U.check_training_resolution(checkpoint, (0.0081, 0.008, 0.008))
    assert not caplog.records
    with caplog.at_level(logging.WARNING):
        U.check_training_resolution(checkpoint, (0.016, 0.016, 0.016))
    assert "extrapolating" in caplog.text
