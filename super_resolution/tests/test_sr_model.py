"""
Tests for the generator, critic and checkpoints.

The property everything downstream leans on is the data-consistency
projection: block-averaging the upscaled volume must give back the input
exactly, for every scale and whatever the weights. The rest check shapes,
that an untrained network is plain (projected) trilinear interpolation, and
that a saved model reloads to the same function.

Tiny networks on the CPU only; nothing here trains or touches a scan.

Run with:  python -m pytest super_resolution/tests
"""

# pylint: disable=wrong-import-position

import sr_phantoms  # noqa: F401  (puts the repository on sys.path)

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from super_resolution.model import (
    PatchCritic3d,
    SRGenerator3d,
    block_mean,
    gradient_l1,
    load_generator,
    nearest_up,
    project_to_block_mean,
    relativistic_losses,
    save_generator,
    select_device,
    upsample_stages,
)


def _tiny(scale=2, **kwargs) -> SRGenerator3d:
    torch.manual_seed(0)
    generator = SRGenerator3d(scale=scale, features=8, growth=4, blocks=1, **kwargs)
    # A zero tail makes the network plain interpolation; give it real output.
    nn.init.normal_(generator.tail.weight, std=0.1)
    return generator.eval()


@pytest.mark.parametrize(
    "scale, expected",
    [
        ((2, 2, 2), [(2, 2, 2)]),
        ((4, 4, 4), [(2, 2, 2), (2, 2, 2)]),
        ((1, 2, 2), [(1, 2, 2)]),
        ((4, 2, 1), [(2, 2, 1), (2, 1, 1)]),
        ((3, 3, 3), [(3, 3, 3)]),
        ((1, 1, 1), []),
    ],
)
def test_upsample_stages_factor_each_axis(scale, expected):
    assert upsample_stages(scale) == expected


def test_projection_is_exact_idempotent_and_blockwise():
    torch.manual_seed(1)
    scale = (2, 3, 2)
    low = torch.rand(2, 1, 4, 5, 6)
    high = torch.rand(2, 1, 8, 15, 12)
    projected = project_to_block_mean(high, low, scale)

    assert torch.allclose(block_mean(projected, scale), low, atol=1e-6)
    assert torch.allclose(project_to_block_mean(projected, low, scale), projected, atol=1e-6)
    # The least-squares correction is constant within each block.
    delta = projected - high
    assert torch.allclose(delta, nearest_up(block_mean(delta, scale), scale), atol=1e-6)


@pytest.mark.parametrize("scale", [(2, 2, 2), (1, 2, 2), (3, 3, 3), (2, 1, 3), (4, 4, 4)])
def test_generator_output_shape_and_consistency(scale):
    generator = _tiny(scale)
    low = torch.rand(1, 1, 6, 7, 5)
    with torch.no_grad():
        high = generator(low)
    assert high.shape == (1, 1, 6 * scale[0], 7 * scale[1], 5 * scale[2])
    assert torch.allclose(block_mean(high, scale), low, atol=1e-5)


def test_untrained_generator_is_projected_trilinear():
    torch.manual_seed(0)
    generator = SRGenerator3d(scale=2, features=8, growth=4, blocks=1).eval()
    low = torch.rand(1, 1, 5, 6, 7)
    trilinear = F.interpolate(low, scale_factor=(2, 2, 2), mode="trilinear", align_corners=False)
    with torch.no_grad():
        assert torch.allclose(generator(low), project_to_block_mean(trilinear, low, (2, 2, 2)), atol=1e-6)


def test_consistency_flag_is_what_enforces_block_means():
    generator = _tiny(2, consistency=False)
    low = torch.rand(1, 1, 5, 5, 5)
    with torch.no_grad():
        drift = (block_mean(generator(low), (2, 2, 2)) - low).abs().max()
    assert drift > 1e-3


def test_critic_scores_patches_and_losses_are_finite():
    critic = PatchCritic3d(features=4)
    scores = critic(torch.rand(2, 1, 16, 16, 16))
    assert scores.shape == (2, 1, 2, 2, 2)
    critic_loss, generator_loss = relativistic_losses(torch.randn(3, 1, 2, 2, 2), torch.randn(3, 1, 2, 2, 2))
    assert critic_loss.ndim == 0 and generator_loss.ndim == 0
    assert torch.isfinite(critic_loss) and torch.isfinite(generator_loss)


def test_gradient_loss_ignores_a_constant_offset():
    a = torch.rand(1, 1, 6, 6, 6)
    assert gradient_l1(a, a).item() == 0.0
    assert gradient_l1(a, a + 0.3).item() < 1e-6
    assert gradient_l1(a, torch.rand_like(a)).item() > 0.0


def test_checkpoint_reloads_the_same_function(tmp_path):
    generator = _tiny((1, 2, 2))
    path = str(tmp_path / "model.pt")
    save_generator(path, generator, {"trained_on": [{"name": "phantom", "spacing_mm_zyx": [0.01] * 3}]})

    loaded, payload = load_generator(path, torch.device("cpu"))
    low = torch.rand(1, 1, 4, 5, 6)
    with torch.no_grad():
        assert torch.equal(generator(low), loaded(low))
    assert loaded.scale == (1, 2, 2)
    assert payload["trained_on"][0]["name"] == "phantom"
    assert not (tmp_path / "model.pt.tmp").exists()


def test_loading_something_else_is_a_clear_error(tmp_path):
    path = str(tmp_path / "other.pt")
    torch.save({"weights": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="not a super_resolution model"):
        load_generator(path, torch.device("cpu"))


def test_select_device():
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in ("cuda", "mps", "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            select_device("cuda")
