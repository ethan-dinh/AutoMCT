"""
End-to-end tests for segmenting on the fine grid and writing to the original.

segment_mandible itself is replaced by a small stand-in that does what matters
here the same way: it reads the upscaled NRRD with segment_mandible's own
loader, thresholds it, and runs the *real* ``reorient_mandible``, returning
its outputs in the pipeline's five-tuple. Everything around it -- upscaling,
the temporary file, mapping back, applying to the original, the outputs -- is
the code the command line runs.

Run with:  python -m pytest super_resolution/tests
"""

# pylint: disable=wrong-import-position

import json
import os

import sr_phantoms as P

import nrrd
import numpy as np
import pytest
import torch

from data_io import load_nrrd, permute_spacing_for_record  # segment_mandible
from preprocessing import reorient_mandible

from super_resolution import sr_segment as S
from super_resolution.config import TrainConfig
from super_resolution.mask_mapping import LABELS, STRUCTURES, reorient
from super_resolution.model import SRGenerator3d, save_generator
from super_resolution.volume_io import Geometry, write_nrrd

GEOMETRY = Geometry(
    directions=np.array([[0.0, 0.0, 0.016], [0.0, 0.016, 0.0], [0.016, 0.0, 0.0]]),
    origin=np.array([1.0, 2.0, 3.0]),
)


def _scan(tmp_path, name="jaw") -> tuple[str, np.ndarray]:
    volume, _ = P.jaw_phantom(body_value=1000, tip_value=3000, air_value=100, dtype=np.uint16)
    path = str(tmp_path / f"{name}.nrrd")
    write_nrrd(path, volume, GEOMETRY)
    return path, volume


def _model(tmp_path) -> str:
    torch.manual_seed(0)
    generator = SRGenerator3d(scale=2, features=4, growth=4, blocks=1)
    path = str(tmp_path / "model.pt")
    save_generator(path, generator, {"trained_on": [{"name": "jaw", "spacing_mm_zyx": list(GEOMETRY.spacing)}]})
    return path


class FakePipeline:
    """Stands in for segment_mandible: threshold the upscaled scan, reorient for real."""

    def __init__(self):
        self.calls = []
        self.spacings = []

    def __call__(self, path, *, spacing_out, **kwargs):
        self.calls.append((path, kwargs))
        volume, spacing = load_nrrd(path, return_spacing=True)
        self.spacings.append(spacing)
        tissue = volume > 500
        incisor = volume > 2000
        molar = np.zeros_like(tissue)
        molar[8:20, 60:80, 14:26] = True
        molar &= tissue & ~incisor
        bone = tissue & ~incisor & ~molar
        record: dict = {}
        preprocessed, (bmp, inc, bon, mol) = reorient_mandible(
            np.where(tissue, volume, 0).astype(np.float32),
            companion_volumes=[volume, incisor, bone, molar], record=record,
        )
        spacing_out["record"] = dict(record)
        spacing_out["spacing"] = permute_spacing_for_record(spacing, record) if spacing else None
        return bmp, preprocessed, np.where(inc, bmp, 0), np.where(bon, bmp, 0), np.where(mol, bmp, 0)


@pytest.fixture
def fake_pipeline(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(S, "load_classical_pipeline", lambda: fake)
    return fake


def _options(tmp_path, **overrides) -> S.SegmentOptions:
    base = dict(
        out_root=str(tmp_path / "out"), device="cpu", tile=8, halo=4,
        output_format="nrrd", progress=False,
    )
    base.update(overrides)
    return S.SegmentOptions(**base)


def _load(out, *parts):
    return load_nrrd(os.path.join(out, *parts))


def test_masks_land_on_the_original_scan(tmp_path, fake_pipeline):
    scan_path, original = _scan(tmp_path)
    opts = _options(tmp_path, model=_model(tmp_path), save_fractions=True, incisor_margin_shrink_mm=0.01)
    summary = S.segment_sample("jaw", scan_path, opts)
    out = os.path.join(opts.out_root, "jaw")

    # The pipeline saw the upscaled volume and the pass-through options.
    (fine_path, kwargs), = fake_pipeline.calls
    assert kwargs["incisor_margin_shrink_mm"] == 0.01
    assert np.allclose(fake_pipeline.spacings[0], [0.008] * 3)
    assert not os.path.exists(fine_path), "temporary upscaled volume should be deleted"

    # A label map in the scan's own frame and geometry.
    labels, header = nrrd.read(os.path.join(out, "jaw_labels_input_frame.nrrd"))
    labels = np.transpose(labels, (2, 1, 0))
    assert labels.shape == original.shape
    geometry = Geometry.from_nrrd_header(header)
    assert np.allclose(geometry.directions, GEOMETRY.directions)
    assert np.allclose(geometry.origin, GEOMETRY.origin)
    assert (labels[P.TIP] == LABELS["incisor"]).mean() > 0.9
    assert (labels[P.BODY] > 0).mean() > 0.95
    assert labels[0, 0, 0] == 0

    # The reoriented outputs are that label map and the original, reoriented alike.
    record = summary["reorientation_record_original_grid"]
    reoriented = _load(out, "jaw_reoriented_volume.nrrd")
    assert np.array_equal(reoriented, reorient(original, record))
    for structure in STRUCTURES:
        mask = _load(out, "masks", f"jaw_{structure}_mask.nrrd").astype(bool)
        assert np.array_equal(mask, reorient(labels == LABELS[structure], record)), structure
        volume = _load(out, "volumes", f"jaw_{structure}_volume.nrrd")
        assert np.array_equal(volume, np.where(mask, reoriented, 0)), structure
        assert summary["structures"][structure]["voxels"] == int(mask.sum())
        # A kept voxel has some of the structure; a voxel more than half
        # covered by it is always kept.
        fine = _load(out, "sr_fractions", f"jaw_{structure}_fine_voxels.nrrd")
        assert fine.max() <= 8 and np.all(fine[mask] > 0) and np.all(fine[~mask] <= 4)
    assert summary["structures"]["molar"]["voxels"] > 0

    with open(os.path.join(out, "jaw_sr_summary.json"), encoding="utf-8") as handle:
        assert json.load(handle)["scale_zyx"] == [2, 2, 2]
    with open(os.path.join(out, "jaw_voxel_size.json"), encoding="utf-8") as handle:
        assert np.allclose(json.load(handle)["spacing_mm_zyx"], [0.016] * 3)


def test_crop_only_changes_where_the_work_happens(tmp_path, fake_pipeline):
    scan_path, original = _scan(tmp_path)
    opts = _options(tmp_path, model=_model(tmp_path), crop=True, crop_margin=2)
    summary = S.segment_sample("jaw", scan_path, opts)
    out = os.path.join(opts.out_root, "jaw")

    box = tuple(slice(lo, hi) for lo, hi in summary["crop_box_zyx"])
    assert np.prod(summary["upscaled_shape_zyx"]) < 8 * original.size
    labels = np.transpose(nrrd.read(os.path.join(out, "jaw_labels_input_frame.nrrd"))[0], (2, 1, 0))
    outside = np.ones(original.shape, dtype=bool)
    outside[box] = False
    assert not labels[outside].any()

    record = summary["reorientation_record_original_grid"]
    assert np.array_equal(_load(out, "jaw_reoriented_volume.nrrd"), reorient(original[box], record))
    mask = _load(out, "masks", "jaw_bone_mask.nrrd").astype(bool)
    assert np.array_equal(mask, reorient(labels[box] == LABELS["bone"], record))
    assert fake_pipeline.calls


def test_a_scan_without_a_voxel_size_is_not_given_one(tmp_path, fake_pipeline):
    volume, _ = P.jaw_phantom(body_value=1000, tip_value=3000, air_value=100, dtype=np.uint16)
    scan_path = str(tmp_path / "bare.nrrd")
    write_nrrd(scan_path, volume, None)  # like a BMP stack: no spacing anywhere
    opts = _options(tmp_path, model=_model(tmp_path))
    summary = S.segment_sample("bare", scan_path, opts)

    assert fake_pipeline.spacings == [None]
    assert summary["voxel_size_mm_zyx"] is None
    assert summary["structures"]["bone"]["volume_mm3"] is None
    assert summary["structures"]["bone"]["subvoxel_voxels"] > 0
    with open(os.path.join(opts.out_root, "bare", "bare_voxel_size.json"), encoding="utf-8") as handle:
        assert json.load(handle)["is_default"] is True


def test_a_pipeline_failure_returns_none_and_cleans_up(tmp_path, monkeypatch):
    scan_path, _ = _scan(tmp_path)
    calls = []

    def failing(path, **_kwargs):
        calls.append(path)

    monkeypatch.setattr(S, "load_classical_pipeline", lambda: failing)
    assert S.segment_sample("jaw", scan_path, _options(tmp_path, model=_model(tmp_path))) is None
    assert calls and not os.path.exists(calls[0])


def test_without_a_model_one_is_trained_then_reused(tmp_path, fake_pipeline):
    scan_path, _ = _scan(tmp_path)
    tiny = TrainConfig(
        scale=2, features=4, growth=4, blocks=1, lr_patch=4, batch_size=2,
        pretrain_iters=2, gan_iters=1, val_patches=2, val_every=10,
        checkpoint_every=10, log_every=10, critic_features=4,
    )
    opts = _options(tmp_path, train=tiny)
    summary = S.segment_sample("jaw", scan_path, opts)
    model_path = os.path.join(opts.out_root, "jaw", "sr_model", "model.pt")
    assert summary["model"] == os.path.abspath(model_path)
    stamp = os.path.getmtime(model_path)
    S.segment_sample("jaw", scan_path, opts)
    assert os.path.getmtime(model_path) == stamp
    assert len(fake_pipeline.calls) == 2


def test_batch_skips_finished_samples(tmp_path, fake_pipeline):
    samples = tmp_path / "samples"
    samples.mkdir()
    _scan(samples, "A")
    _scan(samples, "B")
    opts = _options(tmp_path, model=_model(tmp_path))

    assert S.segment_samples(str(samples), opts) == 0
    assert len(fake_pipeline.calls) == 2
    assert os.path.exists(os.path.join(opts.out_root, "validation_report.json"))
    assert S.segment_samples(str(samples), opts) == 0
    assert len(fake_pipeline.calls) == 2
