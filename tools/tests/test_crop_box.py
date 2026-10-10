"""
Tests for ``--crop-box``: an explicit crop region passed to the converters.

The studio UI draws the crop on a strided preview and hands the converter the
box in source voxel indices, so the box has to mean exactly what a numpy slice
means, survive an edge that overshoots the array, move the NRRD origin with
the crop, and -- the easy one to get wrong -- not be silently replaced by an
older crop that a ``--save-raw`` resume cache happens to hold.

Run with:  python -m pytest tools/tests/test_crop_box.py
"""

# pylint: disable=wrong-import-position

import argparse
import json
import os
import sys

import nrrd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volume_pipeline import (  # noqa: E402
    PipelineOptions,
    convert_volume,
    crop_box_slices,
    directions_from_spacing,
    parse_crop_box,
)

SHAPE = (40, 30, 20)
SPACING = (0.02, 0.01, 0.01)


def _reader(volume):
    def read():
        return volume, directions_from_spacing(*SPACING), np.zeros(3), {"source": "test"}
    return read


def _ramp():
    """A volume whose value encodes its own (z, y, x) index."""
    z, y, x = np.indices(SHAPE)
    return (z * 10000 + y * 100 + x).astype(np.float32)


def test_parse_matches_numpy_slice_semantics():
    assert parse_crop_box("2:10,0:30,5:6") == (2, 10, 0, 30, 5, 6)
    for bad in ("1:2,3:4", "5:5,0:1,0:1", "a:b,0:1,0:1", "-1:4,0:1,0:1"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_crop_box(bad)


def test_box_past_the_edge_is_clamped_not_rejected():
    z, y, x = crop_box_slices((30, 99, 0, 30, 10, 25), SHAPE)
    assert (z.start, z.stop, x.stop) == (30, 40, 20)
    with pytest.raises(ValueError):
        crop_box_slices((45, 50, 0, 1, 0, 1), SHAPE)


def test_convert_crops_and_moves_the_origin(tmp_path):
    opts = PipelineOptions(nlm_h=0.0, crop_box=(5, 15, 3, 23, 2, 12), compression_level=0)
    outputs = convert_volume(output_dir=tmp_path, stem="s", read_source=_reader(_ramp()), opts=opts)

    data, header = nrrd.read(str(outputs["filtered"]))
    data = np.transpose(data, (2, 1, 0))            # NRRD stores X, Y, Z
    assert data.shape == (10, 20, 10)
    assert data[0, 0, 0] == 5 * 10000 + 3 * 100 + 2
    # Voxel [0,0,0] of the output is voxel [5,3,2] of the source, so the
    # origin moves by that many voxels along each axis's direction.
    origin = np.asarray(header["space origin"], dtype=float)
    expected = np.array([5, 3, 2], dtype=float) @ directions_from_spacing(*SPACING)
    assert np.allclose(origin, expected)

    meta = json.loads((tmp_path / "s_meta.json").read_text())
    assert meta["_filter_params"]["crop_box"] == [5, 15, 3, 23, 2, 12]
    assert meta["_filter_params"]["cropped"] is True


def test_a_new_box_is_not_hidden_by_an_old_resume_cache(tmp_path):
    base = dict(nlm_h=0.0, save_raw=True, compression_level=0)
    convert_volume(output_dir=tmp_path, stem="s", read_source=_reader(_ramp()),
                   opts=PipelineOptions(crop_box=(0, 10, 0, 10, 0, 10), **base))
    outputs = convert_volume(output_dir=tmp_path, stem="s", read_source=_reader(_ramp()),
                             opts=PipelineOptions(crop_box=(20, 30, 5, 25, 0, 20), **base))
    data, _ = nrrd.read(str(outputs["filtered"]))
    data = np.transpose(data, (2, 1, 0))
    assert data.shape == (10, 20, 20)
    assert data[0, 0, 0] == 20 * 10000 + 5 * 100


def test_the_same_box_reuses_the_cache(tmp_path):
    calls = []

    def counting_reader():
        calls.append(1)
        return _reader(_ramp())()

    opts = PipelineOptions(crop_box=(0, 10, 0, 10, 0, 10), nlm_h=0.0, save_raw=True,
                           compression_level=0)
    convert_volume(output_dir=tmp_path, stem="s", read_source=counting_reader, opts=opts)
    convert_volume(output_dir=tmp_path, stem="s", read_source=counting_reader, opts=opts)
    assert len(calls) == 1
