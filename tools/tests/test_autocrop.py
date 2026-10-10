"""
Regression tests for how ``--autocrop`` picks the specimen out of its mask.

These pin the fix for autocrop cutting ~1 mm off the incisal tip of P22_TGM2_KO_M_4
(f0004034). At the raw-scan Otsu level the incisor separates from the mandible
across the periodontal ligament, so it was a connected component of its own,
and "keep the largest component" kept only the mandible. The part of the
incisor inside the mandible's bounding box survived by coincidence; the
erupted tip, which reaches past the bone, was cropped away before the
reorientation stage (which already has its own fix for this, see
test_reorient_mask_smooth.py) ever saw it.

Measured on f0004034 (412 x 3168 x 3168 at 6 um, stride 3 x 17 x 17):

                     components   kept of mask   box X start   fitted long extent
    largest only            540          75.3%           714             11.07 mm
    grouped 0.3 mm          540          99.9%           544             12.17 mm

The second thing pinned here is the guard on that grouping. f0004038 is
scanned in a fluid-filled tube, so Otsu splits the out-of-view corners from
the tube contents rather than air from tissue; the fluid is the largest
component, and the tube wall is a broken ring 0.1-0.2 mm outside it. Grouping
merged the ring, the box grew to the full field of view, and the reorientation
stage then fit a 15.3 mm box to an 11.7 mm specimen. The ring reaches the edge
of the array where the fluid does not, and that is what the guard keys on.

Run with:  python -m pytest tools/tests/test_autocrop.py
"""

# pylint: disable=wrong-import-position

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volume_pipeline import (  # noqa: E402
    AUTOCROP_VERSION,
    PipelineOptions,
    autocrop_bbox,
    convert_volume,
    directions_from_spacing,
    largest_specimen_group,
)

SPACING_MM = 0.05


def _mandible_with_detached_incisor(seed=0):
    """A bone block, an incisor bar 0.1 mm below it, and debris 1.35 mm away.

    The bar runs past the block's low-X end, the way an erupted incisor tip
    runs past the alveolar bone.
    """
    rng = np.random.default_rng(seed)
    vol = np.full((40, 120, 160), 1000.0, dtype=np.float32)
    vol[10:30, 40:90, 50:140] = 10000.0   # mandible: Y 40-89, X 50-139
    vol[15:25, 92:98, 10:120] = 14000.0   # incisor: two empty rows below, tip at X 10
    vol[15:18, 10:13, 100:103] = 10000.0  # debris, well clear of both
    vol += rng.normal(0.0, 150.0, size=vol.shape).astype(np.float32)
    return vol


def _box(volume, **kwargs):
    slices = autocrop_bbox(
        volume, margin_mm=0.0, spacing=np.full(3, SPACING_MM), max_dim=200, **kwargs,
    )
    return [(s.start, s.stop) for s in slices]


def test_largest_component_alone_cuts_the_tip():
    """Guards the premise, so the recovery test below cannot pass vacuously."""
    _, (_, y1), (x0, _) = _box(_mandible_with_detached_incisor(), bridge_mm=0.0)
    assert x0 == 50 and y1 == 90


def test_a_detached_incisor_stays_in_the_box():
    _, (_, y1), (x0, x1) = _box(_mandible_with_detached_incisor())
    assert x0 == 10 and y1 == 98 and x1 == 140


def test_distant_debris_still_drops_out():
    _, (y0, _), _ = _box(_mandible_with_detached_incisor())
    assert y0 == 40


def _fluid_disk_in_a_broken_tube_wall(shape=(6, 101, 101)):
    """Fluid filling a tube, and the tube wall as arcs just outside it.

    The wall reaches the edge of the array at the four axis points and is cut
    into arcs, so the diagonal arcs never touch the edge themselves -- which is
    what defeats a guard that looks at each piece on its own.
    """
    yy, xx = np.mgrid[:shape[1], :shape[2]] - 50
    r = np.hypot(yy, xx)
    theta = (np.degrees(np.arctan2(yy, xx)) + 15) % 30  # cuts at 15, 45, 75... deg
    disk = r <= 40
    wall = (r >= 42) & (r <= 50) & (theta > 3)
    return np.broadcast_to(disk, shape).copy(), np.broadcast_to(disk | wall, shape).copy()


def test_a_tube_wall_does_not_widen_the_box():
    disk, mask = _fluid_disk_in_a_broken_tube_wall()
    kept, n = largest_specimen_group(mask, np.full(3, SPACING_MM))
    assert n > 2
    assert np.array_equal(kept, disk)


def test_zero_bridge_keeps_exactly_the_largest_component():
    mask = _mandible_with_detached_incisor() > 5000
    labels, _ = label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    kept, _ = largest_specimen_group(mask, np.full(3, SPACING_MM), bridge_mm=0.0)
    assert np.array_equal(kept, labels == sizes.argmax())


def test_one_component_is_returned_untouched():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:8, 2:8, 2:8] = True
    kept, n = largest_specimen_group(mask, np.full(3, SPACING_MM))
    assert n == 1 and np.array_equal(kept, mask)


# --- the resume cache -------------------------------------------------------
#
# The --save-raw copy is reused whenever the settings match, and the fix above
# changed the crop without changing any setting. P22_TGM2_KO_M_4 was re-run
# after the fix and came out exactly as truncated as before, because the run
# reloaded the copy the old autocrop had cut and never read the scan.


def _counting_run(tmp_path, calls):
    def read():
        calls.append(1)
        vol = _mandible_with_detached_incisor()
        return vol, directions_from_spacing(*([SPACING_MM] * 3)), np.zeros(3), {"source": "test"}

    opts = PipelineOptions(autocrop=True, autocrop_margin_mm=0.1, save_raw=True,
                           nlm_h=0.0, compression_level=0)
    convert_volume(output_dir=tmp_path, stem="s", read_source=read, opts=opts)
    return json.loads((tmp_path / "s_meta.json").read_text())["_filter_params"]


def test_a_copy_cut_by_an_older_autocrop_is_read_again(tmp_path):
    calls = []
    _counting_run(tmp_path, calls)
    # What every copy made before AUTOCROP_VERSION existed looks like.
    meta_path = tmp_path / "s_meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["_filter_params"]["autocrop_version"]
    meta_path.write_text(json.dumps(meta))

    params = _counting_run(tmp_path, calls)
    assert len(calls) == 2
    assert params["autocrop_version"] == AUTOCROP_VERSION


def test_a_copy_cut_by_the_current_autocrop_is_reused_and_keeps_its_record(tmp_path):
    calls = []
    first = _counting_run(tmp_path, calls)
    second = _counting_run(tmp_path, calls)
    assert len(calls) == 1
    assert first["autocrop_version"] == second["autocrop_version"] == AUTOCROP_VERSION


# --- the real scans ---------------------------------------------------------

TGM2 = Path("/Volumes/T7/RK-TGM2")
INCISOR_SCAN = TGM2 / "KO-M-1" / "f0004034.isq"   # P22_TGM2_KO_M_4
FLUID_SCAN = TGM2 / "KO-F-2" / "f0004038.isq"
SCAN_SPACING = np.full(3, 0.006)


class _ClampedScan:
    """An ISQ as autocrop reads it, without loading 8 GB.

    Memory-mapped, clamped at 0 the way the studio's --clamp-negative
    conversion is, and cached per slicing, so each test reads only the strided
    subsample, and only once.
    """

    def __init__(self, path: Path):
        from convert_ISQ_NRRD import _HEADER_SIZE, _parse_isq_header  # noqa: PLC0415

        with open(path, "rb") as fh:
            header = _parse_isq_header(fh.read(_HEADER_SIZE))
        shape = (header["dim_z_pixels"], header["dim_y_pixels"], header["dim_x_pixels"])
        offset = path.stat().st_size - 2 * int(np.prod(shape))
        self._data = np.memmap(path, dtype="<i2", mode="r", offset=offset, shape=shape)
        self.shape = shape
        self._cache = {}

    def __getitem__(self, key):
        if repr(key) not in self._cache:
            self._cache[repr(key)] = np.maximum(np.asarray(self._data[key]), 0)
        return self._cache[repr(key)]


def _scan(path: Path) -> _ClampedScan:
    if not path.is_file():
        pytest.skip(f"scan not mounted: {path}")
    return _ClampedScan(path)


@pytest.fixture(scope="module")
def incisor_scan():
    return _scan(INCISOR_SCAN)


@pytest.fixture(scope="module")
def fluid_scan():
    return _scan(FLUID_SCAN)


def test_real_scan_largest_component_alone_cuts_the_tip(incisor_scan):
    _, y, x = autocrop_bbox(
        incisor_scan, margin_mm=0.0, spacing=SCAN_SPACING, bridge_mm=0.0,
    )
    assert (y.start, x.start) == (1241, 714)


def test_real_scan_keeps_the_incisor_tip(incisor_scan):
    """The tip reaches X 544 and Y 1173 at the 17-voxel stride."""
    _, y, x = autocrop_bbox(incisor_scan, margin_mm=0.0, spacing=SCAN_SPACING)
    assert x.start <= 544 and y.start <= 1173


def test_real_fluid_scan_box_is_unchanged(fluid_scan):
    """The guard's case: grouping must fall back to the old box exactly."""
    grouped = autocrop_bbox(fluid_scan, margin_mm=0.1, spacing=SCAN_SPACING)
    alone = autocrop_bbox(fluid_scan, margin_mm=0.1, spacing=SCAN_SPACING, bridge_mm=0.0)
    assert grouped == alone
