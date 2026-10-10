"""
Tests for the previews the viewer streams.

A preview is only useful if the numbers the viewer shows are the scan's own:
quantization must round-trip to within one step of the recorded range,
masked volumes must keep background distinguishable from tissue, block
averaging must not darken tissue at a mask's edge, and the full-resolution
histogram must describe the full volume rather than the preview. The unwrap
test pins the geometry the CMPR view relies on: a bright band on one side of a
centred tube shows up at the matching angle.

Run with:  python -m pytest studio/tests/test_volumes.py
"""

import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio.volumes import (  # noqa: E402
    block_reduce,
    build_view,
    choose_factors,
    load_volume,
    nonzero_bbox,
    quantize,
    read_gz,
    suggest_crop,
    unwrap_cmpr,
)
from studio.tests.phantom import SPACING, write_isq, write_segmentation  # noqa: E402


def test_factors_meet_the_budget():
    assert choose_factors((100, 100, 100), 2_000_000) == (1, 1, 1)
    f = choose_factors((2000, 1000, 700), 24_000_000)
    assert f[0] == f[1] == f[2]
    assert math.prod(math.ceil(n / k) for n, k in zip((2000, 1000, 700), f)) <= 24_000_000
    fz = choose_factors((1500, 150, 150), 16_000_000, z_first=True)
    assert fz[1] == fz[2] == 1 and fz[0] == 3


def test_block_reduce_modes():
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    vol[:2, :2, :2] = 10.0
    vol[0, 0, 0] = 0.0                                  # one hole in the tissue block
    mean, _ = block_reduce(vol, (2, 2, 2), "mean")
    nz, has = block_reduce(vol, (2, 2, 2), "nzmean")
    frac, _ = block_reduce(vol, (2, 2, 2), "frac")
    assert mean[0, 0, 0] == 70.0 / 8
    assert nz[0, 0, 0] == 10.0 and has[0, 0, 0] and not has[1, 1, 1]
    assert frac[0, 0, 0] == 7 / 8
    # A partial block at the far edge averages over its own voxels only.
    edge, _ = block_reduce(np.ones((3, 3, 3), dtype=np.float32), (2, 2, 2), "mean")
    assert np.allclose(edge, 1.0)


def test_quantize_round_trip_and_background():
    values = np.linspace(-120.0, 3000.0, 1000).astype(np.float32)
    q = quantize(values, -120.0, 3000.0)
    step = 3120.0 / 65535
    back = -120.0 + q.astype(np.float64) * step
    assert np.max(np.abs(back - values)) <= step
    has = np.ones(values.shape, dtype=bool)
    has[::7] = False
    qm = quantize(values, -120.0, 3000.0, has)
    assert (qm[~has] == 0).all() and (qm[has] >= 1).all()


def test_nonzero_bbox_pads_and_clamps():
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    vol[5:8, 0:3, 10:19] = 1
    box = nonzero_bbox(vol, margin=2)
    assert box == (slice(3, 10), slice(0, 5), slice(8, 20))
    assert nonzero_bbox(np.zeros((4, 4, 4))) is None


def test_unwrap_finds_the_bright_side():
    depth, size = 6, 41
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2
    r = np.hypot(yy, xx)
    ring = ((r > 8) & (r < 14)).astype(np.float32) * 100
    ring[(r > 8) & (r < 14) & (xx > 5)] = 300          # bright wall on the +x side
    vol = np.repeat(ring[None], depth, axis=0)
    u = unwrap_cmpr(vol, n_angles=36)
    assert u.shape == (depth, 36)
    assert u[0, 0] == 300 and u[0, 18] == 100            # 0 deg is +x, 180 deg is -x


def test_build_view_masked_and_overlays(tmp_path):
    folder = write_segmentation(tmp_path / "seg", "S1", seed=3)
    out = tmp_path / "preview"
    meta = build_view(out / "molar", view="molar",
                      base={"path": str(folder / "volumes" / "S1_molar_volume.nrrd"), "key": "molar_volume"},
                      overlays=[{"name": "molar", "path": str(folder / "masks" / "S1_molar_mask.nrrd")}],
                      budget=5_000, masked=True, log=lambda *_: None)
    saved = json.loads((out / "molar" / "view.json").read_text())
    assert saved["shape"] == meta["shape"] and math.prod(meta["shape"]) <= 5_000
    assert meta["spacing_known"] and meta["spacing_mm"] == list(SPACING)
    q = read_gz(out / "molar" / "base.u16.gz", "<u2", meta["shape"])
    # Tissue values read back inside the true range; background stays 0.
    lo, hi = meta["quant"]["lo"], meta["quant"]["hi"]
    tissue = q[q > 0].astype(np.float64)
    values = lo + (tissue - 1) * (hi - lo) / 65534
    assert values.min() >= lo - 1e-6 and values.max() <= hi + 1e-6
    ov = read_gz(out / "molar" / "ov_molar.u8.gz", "u1", meta["shape"])
    assert ov.max() == 255
    # Voxel counts are full resolution, not preview.
    assert meta["counts"]["molar"] == meta["counts"]["molar_volume"] > math.prod(meta["shape"])
    assert sum(meta["histogram"]["counts"]) > 0 and (out / "molar" / "thumb.png").exists()


def test_source_view_reads_isq_strided_and_suggests_a_crop(tmp_path):
    isq = write_isq(tmp_path / "scan.isq")
    meta = build_view(tmp_path / "src", view="source", base={"path": str(isq), "key": "raw"},
                      budget=40_000, source_kind="isq", log=lambda *_: None)
    assert meta["sampling"] == "strided" and meta["full_shape"] == [180, 90, 72]
    q = read_gz(tmp_path / "src" / "base.u16.gz", "<u2", meta["shape"])
    box = suggest_crop(q, meta, margin_mm=0.0)
    z0, z1, y0, y1, x0, x1 = box
    # The phantom's tissue spans y ~ 16-78 and x ~ 10-62; the box must hold it
    # and drop most of the air around it.
    assert y0 <= 17 and y1 >= 77 and x0 <= 11 and x1 >= 61
    assert (z1 - z0) * (y1 - y0) * (x1 - x0) < 180 * 90 * 72


def test_a_bmp_preview_skips_macos_sidecars(tmp_path):
    from PIL import Image  # noqa: PLC0415
    src = tmp_path / "bmp"
    src.mkdir()
    for i in range(12):
        Image.fromarray(np.full((20, 16), 40 + i, np.uint8)).save(src / f"scan_rec_Tra{i:08d}.bmp")
        # What macOS writes beside each file on an exFAT drive: not an image.
        (src / f"._scan_rec_Tra{i:08d}.bmp").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 60)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(src / "scan_spr.bmp")
    meta = build_view(tmp_path / "src", view="source", base={"path": str(src), "key": "raw"},
                      budget=40_000, source_kind="bmp", log=lambda *_: None)
    assert meta["full_shape"] == [12, 20, 16]


def test_a_preview_survives_its_drive_being_mounted_elsewhere(tmp_path):
    from studio.volumes import file_identity, view_is_current, PREVIEW_VERSION  # noqa: PLC0415
    a = tmp_path / "mac" / "scan.nrrd"
    a.parent.mkdir()
    a.write_bytes(b"x" * 1000)
    meta = {"version": PREVIEW_VERSION, "base": file_identity(a), "overlays": []}
    b = tmp_path / "pc" / "scan.nrrd"
    b.parent.mkdir()
    shutil.copy2(a, b)  # same name, size and time, another path
    assert view_is_current(meta, b, {})
    stamp = a.stat().st_mtime
    os.utime(b, (stamp + 3600, stamp + 3600))  # read an hour off, as FAT local time can be
    assert view_is_current(meta, b, {})
    os.utime(b, (stamp + 30, stamp + 30))  # a genuinely newer file
    assert not view_is_current(meta, b, {})


def test_a_label_map_overlay_keeps_one_label(tmp_path):
    """The molar step writes its teeth as one label map; each tooth's overlay is the voxels holding its number."""
    from scipy import ndimage as ndi  # noqa: PLC0415

    folder = write_segmentation(tmp_path / "seg", "S1", seed=3)
    from data_io import save_volume  # noqa: PLC0415  (importable once write_segmentation has run)

    molar, _ = load_volume(folder / "masks" / "S1_molar_mask.nrrd")
    teeth, n = ndi.label(molar > 0)
    assert n == 3
    labels = save_volume(teeth.astype(np.int16), "", "S1_tooth_labels", base_dir=str(tmp_path),
                         output_format="nrrd", spacing=SPACING)
    out = tmp_path / "preview" / "molar"
    meta = build_view(out, view="molar",
                      base={"path": str(folder / "volumes" / "S1_molar_volume.nrrd"), "key": "molar_volume"},
                      overlays=[{"name": f"m{k}", "path": labels, "value": k} for k in (1, 2, 3)],
                      budget=5_000, masked=True, log=lambda *_: None)
    assert [meta["counts"][f"m{k}"] for k in (1, 2, 3)] == np.bincount(teeth.ravel())[1:].tolist()
    shares = [read_gz(out / f"ov_m{k}.u8.gz", "u1", meta["shape"]).astype(int) for k in (1, 2, 3)]
    assert all(share.max() == 255 for share in shares)
    assert (sum(shares) <= 255 + 2).all(), "the teeth do not overlap"


def test_the_molar_view_shows_each_tooth(tmp_path):
    """With the molar step's tooth labels there, the molar view overlays M1-M3, ahead of the enamel."""
    from studio import steps  # noqa: PLC0415

    class Workspace:
        files = {"molar_volume": tmp_path / "S1_molar_volume.nrrd",
                 "tooth_labels": tmp_path / "S1_tooth_labels.nrrd",
                 "enamel_mask": tmp_path / "S1_enamel_mask.nrrd"}

        def artifact(self, _record, key):
            return self.files.get(key)

        def settings(self):
            return {}

        def sample_dir(self, name):
            return tmp_path / name

    spec = steps.view_spec(Workspace(), {"name": "S1"}, "molar")
    assert [(o["name"], o["label"], o["value"]) for o in spec["overlays"]] == [
        ("m1", "M1", 1), ("m2", "M2", 2), ("m3", "M3", 3), ("enamel", "Enamel", None)]
    assert {o["path"] for o in spec["overlays"][:3]} == {str(tmp_path / "S1_tooth_labels.nrrd")}
