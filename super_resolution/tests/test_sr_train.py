"""
Tests for training data and the training loop.

The data claims: training patches never touch the held-out slab (otherwise
validation would score anatomy the model trained on), the degraded input has
the noise level it is meant to have, and the noise estimate recovers a known
sigma. The loop claims: a run writes every file it promises, a saved state
resumes where it stopped, and resuming with a different architecture is
refused rather than silently mixing two models.

A few iterations of a tiny network on the CPU; this checks the code, it does
not train anything useful.

Run with:  python -m pytest super_resolution/tests
"""

# pylint: disable=wrong-import-position,protected-access

import json
import os
from dataclasses import replace

import sr_phantoms  # noqa: F401  (puts the repository on sys.path)

import numpy as np
import pytest
import torch

from super_resolution.config import TrainConfig
from super_resolution.model import block_mean, load_generator
from super_resolution.patches import PatchSampler, estimate_noise_sigma, prepare_volume
from super_resolution.train import degrade, train_model
from super_resolution.volume_io import Geometry, IntensityScale, Scan

CPU = torch.device("cpu")
TINY = TrainConfig(
    scale=2, features=4, growth=4, blocks=1, lr_patch=8, batch_size=2,
    pretrain_iters=3, gan_iters=3, val_patches=4, val_every=2,
    checkpoint_every=2, log_every=2, critic_features=4,
)


def _scan(shape=(64, 32, 32), seed=0) -> Scan:
    rng = np.random.default_rng(seed)
    volume = np.full(shape, 100.0)
    volume[8:56, 6:26, 6:26] = 1000.0
    volume[20:44, 10:16, 10:16] = 2500.0
    volume += rng.normal(0, 20, size=shape)
    return Scan(
        name="phantom", path="phantom", source="phantom",
        data=np.clip(volume, 0, None).astype(np.uint16),
        geometry=Geometry.from_spacing((0.01, 0.01, 0.01)), spacing_from_file=True,
    )


def test_training_patches_never_touch_the_held_out_slab():
    rng = np.random.default_rng(0)
    patch = (16, 16, 16)
    volume = prepare_volume(_scan(), patch, rng)
    assert volume.holdout == (24, 40)
    sampler = PatchSampler([volume], patch, rng)
    for _ in range(300):
        corner = sampler._train_corner(volume)
        assert not sampler._overlaps_holdout(volume, int(corner[0]))
        assert np.all(corner >= 0) and np.all(corner + patch <= volume.data.shape)
    for _ in range(100):
        corner = sampler._val_corner(volume)
        assert volume.holdout[0] <= corner[0] and corner[0] + patch[0] <= volume.holdout[1]

    batch, sigma = sampler.sample(5)
    assert batch.shape == (5, 1, 16, 16, 16) and batch.dtype == np.float32
    assert sigma.shape == (5,)


def test_most_training_patches_contain_tissue():
    rng = np.random.default_rng(1)
    scan = _scan()
    scan.data[:] = 100
    scan.data[10:14, 14:18, 14:18] = 2000  # a small specimen in a lot of air, clear of the held-out slab
    volume = prepare_volume(scan, (8, 8, 8), rng)
    sampler = PatchSampler([volume], (8, 8, 8), rng)
    batch, _ = sampler.sample(200)
    with_tissue = (batch.reshape(200, -1).max(axis=1) > 0.5).mean()
    assert with_tissue > 0.6


def test_noise_estimate_recovers_a_known_sigma():
    rng = np.random.default_rng(2)
    data = rng.normal(1000.0, 20.0, size=(48, 48, 48)).astype(np.float32)
    intensity = IntensityScale(900.0, 1100.0)
    coords = np.argwhere(np.ones(data.shape, dtype=bool))[::101]
    sigma = estimate_noise_sigma(data, intensity, coords, rng, blocks=24, size=24)
    assert sigma == pytest.approx(20.0 / 200.0, rel=0.1)


def test_degrade_block_averages_and_restores_the_noise_level():
    torch.manual_seed(0)
    high = torch.rand(4, 1, 16, 16, 16)
    clean = degrade(high, (2, 2, 2), torch.zeros(4), (1.0, 1.0))
    assert torch.allclose(clean, block_mean(high, (2, 2, 2)))

    noisy = degrade(high, (2, 2, 2), torch.full((4,), 0.5), (1.0, 1.0))
    measured = float((noisy - clean).std())
    assert measured == pytest.approx(0.5 * np.sqrt(7 / 8), rel=0.05)


def test_training_writes_its_files_and_resumes(tmp_path):
    out = str(tmp_path / "model")
    model_path = train_model([_scan()], out, TINY, CPU)

    for name in ("model.pt", "model_l1.pt", "history.json", "last_state.pt"):
        assert os.path.exists(os.path.join(out, name)), name
    generator, payload = load_generator(model_path, CPU)
    assert generator.scale == (2, 2, 2)
    assert payload["iterations"] == 6
    assert payload["trained_on"][0]["name"] == "phantom"
    assert payload["trained_on"][0]["holdout_slices"] == [24, 40]
    with open(os.path.join(out, "history.json"), encoding="utf-8") as handle:
        history = json.load(handle)["validation"]
    assert [h["iteration"] for h in history] == [2, 3, 4, 6]
    assert all(np.isfinite(h["psnr_model"]) for h in history)

    longer = replace(TINY, gan_iters=5)
    train_model([_scan()], out, longer, CPU, resume=True)
    _, payload = load_generator(model_path, CPU)
    assert payload["iterations"] == 8

    with pytest.raises(ValueError, match="Cannot resume"):
        train_model([_scan()], out, replace(TINY, features=8), CPU, resume=True)


def test_fidelity_only_training(tmp_path):
    out = str(tmp_path / "l1")
    train_model([_scan()], out, replace(TINY, gan_iters=0), CPU)
    _, payload = load_generator(os.path.join(out, "model.pt"), CPU)
    assert payload["train_config"]["gan_iters"] == 0


def test_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(pretrain_iters=0, gan_iters=0)
    with pytest.raises(ValueError):
        TrainConfig(scale=(2, 0, 2))
    assert TrainConfig(scale=3).hr_patch == (96, 96, 96)
