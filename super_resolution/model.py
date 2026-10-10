"""
3-D generative super-resolution network for micro-CT.

The generator is a volumetric ESRGAN (Wang et al. 2018): residual-in-residual
dense blocks, nearest-neighbour upsampling followed by convolution (which does
not produce the checkerboard pattern transposed convolutions and voxel shuffle
leave behind), and an adversarial critic trained alongside it. Everything is
3-D, not slice-wise, so the added detail is consistent through Z rather than
being hallucinated independently on every slice.

Two design choices differ from the 2-D original, both for the same reason: the
upscaled volume is only ever used to *segment* the scan, so being faithful to
the measured data matters more than looking sharp.

1. **Global residual.** The network predicts detail on top of a trilinear
   upsample of its input instead of the whole output. An untrained or
   undertrained network therefore degrades to plain interpolation rather than
   to noise.

2. **Data-consistency projection.** The last layer projects the output onto
   the set of volumes whose s x s x s block means equal the input exactly
   (:func:`project_to_block_mean`). A micro-CT voxel is, to first order, the
   mean attenuation over its cell -- the partial-volume model -- so this is the
   physical statement "the fine voxels inside one scanned voxel must average
   to what the scanner measured there". The generator can redistribute
   intensity *within* a voxel, which is where sub-voxel boundaries live, but it
   cannot add or remove density at the scanned resolution. Block-averaging the
   upscaled volume back down reproduces the original scan.
"""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.nn import functional as F

from .config import Scale, as_scale

CHECKPOINT_FORMAT = "microct-super-resolution/1"


def _prime_factors(n: int) -> list[int]:
    factors, p = [], 2
    while n > 1:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1
    return factors


def upsample_stages(scale: Scale) -> list[Scale]:
    """
    Split a per-axis scale into a sequence of small upsampling steps.

    Each axis is factored into primes and the steps are aligned across axes,
    so (4, 4, 4) becomes two (2, 2, 2) stages and an anisotropic (1, 2, 2)
    becomes a single in-plane stage. Every stage is followed by a convolution,
    so a large factor is reached through intermediate resolutions rather than
    one blocky jump.
    """
    per_axis = [_prime_factors(s) for s in scale]
    depth = max((len(f) for f in per_axis), default=0)
    stages = []
    for i in range(depth):
        stages.append(tuple(f[i] if i < len(f) else 1 for f in per_axis))
    return stages  # type: ignore[return-value]


def block_mean(volume: torch.Tensor, scale: Scale) -> torch.Tensor:
    """Mean over non-overlapping s_z x s_y x s_x blocks (the degradation operator)."""
    return F.avg_pool3d(volume, kernel_size=scale, stride=scale)


def nearest_up(volume: torch.Tensor, scale: Scale) -> torch.Tensor:
    """Replicate every voxel into an s_z x s_y x s_x block (the adjoint of block_mean, up to 1/s^3)."""
    for axis, s in zip((2, 3, 4), scale):
        if s > 1:
            volume = volume.repeat_interleave(s, dim=axis)
    return volume


def project_to_block_mean(high: torch.Tensor, low: torch.Tensor, scale: Scale) -> torch.Tensor:
    """
    Orthogonal projection of ``high`` onto {x : block_mean(x) == low}.

    The block-mean operator A has A A^T = I / s^3, so the least-squares
    correction is x + A^T (A A^T)^-1 (y - A x) = x + nearest_up(y - A x): add
    each block's residual uniformly to its fine voxels. It changes ``high`` as
    little as possible (in L2) while making the constraint hold exactly.
    """
    return high + nearest_up(low - block_mean(high, scale), scale)


class ResidualDenseBlock3d(nn.Module):
    """Five densely connected 3x3x3 convolutions with a scaled residual (ESRGAN RDB)."""

    def __init__(self, features: int, growth: int):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv3d(features + i * growth, growth if i < 4 else features, 3, padding=1)
            for i in range(5)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        # Zero-initialising the last convolution makes every block start as the
        # identity, which is what lets a deep residual stack train stably
        # without batch norm (which ESRGAN drops for SR because it adds
        # artefacts at inference-time batch statistics).
        nn.init.zeros_(self.convs[-1].weight)
        nn.init.zeros_(self.convs[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for conv in self.convs[:-1]:
            features.append(self.act(conv(torch.cat(features, dim=1))))
        return x + 0.2 * self.convs[-1](torch.cat(features, dim=1))


class RRDB3d(nn.Module):
    """Residual-in-residual dense block: three RDBs under one more scaled skip."""

    def __init__(self, features: int, growth: int):
        super().__init__()
        self.blocks = nn.Sequential(*(ResidualDenseBlock3d(features, growth) for _ in range(3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.blocks(x)


class SRGenerator3d(nn.Module):
    """
    Volumetric ESRGAN generator with a trilinear global residual and an
    optional hard data-consistency projection.

    Input and output are single-channel intensity volumes normalised to
    roughly [0, 1] (see ``volume_io.IntensityScale``), shaped (N, 1, Z, Y, X).
    """

    def __init__(
        self,
        scale: int | Scale = 2,
        features: int = 32,
        growth: int = 16,
        blocks: int = 4,
        consistency: bool = True,
    ):
        super().__init__()
        self.scale = as_scale(scale)
        self.consistency = consistency
        self.config = {
            "scale": list(self.scale), "features": features, "growth": growth,
            "blocks": blocks, "consistency": consistency,
        }

        self.head = nn.Conv3d(1, features, 3, padding=1)
        self.trunk = nn.Sequential(*(RRDB3d(features, growth) for _ in range(blocks)))
        self.trunk_tail = nn.Conv3d(features, features, 3, padding=1)
        self.stages = upsample_stages(self.scale)
        self.up_convs = nn.ModuleList(
            nn.Conv3d(features, features, 3, padding=1) for _ in self.stages
        )
        self.hr_conv = nn.Conv3d(features, features, 3, padding=1)
        self.tail = nn.Conv3d(features, 1, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        # The tail starts at zero so the untrained network is exactly the
        # trilinear baseline (plus the projection), not random texture.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, low: torch.Tensor) -> torch.Tensor:
        feat = self.head(low)
        body = feat + self.trunk_tail(self.trunk(feat))
        for stage, conv in zip(self.stages, self.up_convs):
            body = self.act(conv(nearest_up(body, stage)))
        detail = self.tail(self.act(self.hr_conv(body)))
        base = F.interpolate(low, scale_factor=self.scale, mode="trilinear", align_corners=False)
        high = base + detail
        if self.consistency:
            high = project_to_block_mean(high, low, self.scale)
        return high

    @property
    def receptive_radius(self) -> int:
        """
        Upper bound on the receptive-field radius, in *input* voxels.

        Every 3x3x3 convolution at the input resolution reaches one voxel
        further (the head, 15 per RRDB, and the trunk tail). A convolution
        after upsampling reaches less than one input voxel, but can still cross
        into the next one, so each of those counts as one too. The *effective*
        field is far smaller than this bound, which is why tiled inference uses
        a much smaller halo by default; the tests use the bound to show tiling
        is otherwise exact.
        """
        return 2 + 15 * len(self.trunk) + len(self.stages) + 2


class PatchCritic3d(nn.Module):
    """
    3-D PatchGAN discriminator with spectral normalisation.

    It scores overlapping HR sub-volumes rather than whole patches, so the
    adversarial signal is about local texture and edge sharpness -- the thing
    L1 training blurs -- not about global anatomy, which the consistency
    projection already pins to the input.
    """

    def __init__(self, features: int = 32):
        super().__init__()
        sn = nn.utils.spectral_norm
        widths = [features, features * 2, features * 4, features * 4]
        layers: list[nn.Module] = [sn(nn.Conv3d(1, widths[0], 3, padding=1)), nn.LeakyReLU(0.2, True)]
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            layers += [sn(nn.Conv3d(c_in, c_out, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, True)]
        layers.append(sn(nn.Conv3d(widths[-1], 1, 3, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def relativistic_losses(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Relativistic-average GAN losses (Jolicoeur-Martineau 2018, used by ESRGAN).

    Returns (critic_loss, generator_loss). The critic estimates whether a real
    patch is *more realistic than the average fake* rather than whether it is
    real in absolute terms, which gives the generator gradient from both real
    and fake batches and trains more stably than the standard GAN loss.
    """
    real_rel = real_logits - fake_logits.mean()
    fake_rel = fake_logits - real_logits.mean()
    ones, zeros = torch.ones_like(real_rel), torch.zeros_like(real_rel)
    bce = F.binary_cross_entropy_with_logits
    critic = 0.5 * (bce(real_rel, ones) + bce(fake_rel, zeros))
    generator = 0.5 * (bce(real_rel, zeros) + bce(fake_rel, ones))
    return critic, generator


def gradient_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """L1 distance between forward differences along Z, Y and X -- an edge-fidelity term."""
    loss = a.new_zeros(())
    for dim in (2, 3, 4):
        loss = loss + (torch.diff(a, dim=dim) - torch.diff(b, dim=dim)).abs().mean()
    return loss / 3


def build_generator(config: dict) -> SRGenerator3d:
    """Rebuild a generator from the ``config`` dict stored in a checkpoint."""
    return SRGenerator3d(
        scale=tuple(config["scale"]),
        features=int(config["features"]),
        growth=int(config["growth"]),
        blocks=int(config["blocks"]),
        consistency=bool(config["consistency"]),
    )


# ---------------------------------------------------------------------------
# Devices and checkpoints
# ---------------------------------------------------------------------------

def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def select_device(name: str = "auto") -> torch.device:
    """
    Resolve ``--device``: "auto" picks CUDA, then Apple MPS, then the CPU.

    An explicitly requested accelerator that is not usable is an error rather
    than a silent fall back to the CPU, which would turn a GPU run into a
    days-long one without saying so.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. Install a "
            "CUDA build of PyTorch (see env/README.md), or run the Docker image "
            "with --gpus all."
        )
    if device.type == "mps" and not _mps_available():
        raise RuntimeError("MPS was requested but is not available on this machine.")
    return device


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return f"cuda:{index} ({props.name}, {props.total_memory / 1e9:.1f} GB)"
    return str(device)


def save_generator(path: str, generator: SRGenerator3d, metadata: dict) -> None:
    """
    Save an inference checkpoint: weights, architecture, and provenance.

    Only plain Python values go into ``metadata`` so the file loads under
    ``torch.load(weights_only=True)`` -- the default from PyTorch 2.6 on, and
    the safe way to open a checkpoint someone else sent you. Written to a
    temporary name and renamed, so an interrupted save never leaves a
    truncated model where a good one was.
    """
    payload = {
        "format": CHECKPOINT_FORMAT,
        "generator": {k: v.detach().cpu() for k, v in generator.state_dict().items()},
        "generator_config": dict(generator.config),
        **metadata,
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_generator(path: str, device: torch.device) -> tuple[SRGenerator3d, dict]:
    """Load an inference checkpoint written by :func:`save_generator`, ready for eval."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a super_resolution model checkpoint. Pass the model.pt "
            "written by `python -m super_resolution train`, not last_state.pt."
        )
    generator = build_generator(payload["generator_config"])
    generator.load_state_dict(payload["generator"])
    generator.to(device).eval()
    return generator, payload
