"""
Training settings, kept free of torch so the CLI can build its help text and
`check` can report a broken torch install instead of crashing on import.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

Scale = tuple[int, int, int]


def as_scale(scale) -> Scale:
    """Normalise a scale given as one integer or a (z, y, x) triple."""
    if isinstance(scale, int):
        scale = (scale, scale, scale)
    values = tuple(int(s) for s in scale)
    if len(values) != 3 or any(s < 1 for s in values):
        raise ValueError(f"scale must be a positive int or a 3-tuple of them, got {scale!r}")
    return values  # type: ignore[return-value]


@dataclass
class TrainConfig:
    """Everything that defines a training run; saved into each checkpoint."""

    scale: Scale = (2, 2, 2)
    features: int = 32
    growth: int = 16
    blocks: int = 4
    consistency: bool = True
    lr_patch: int = 32
    batch_size: int = 4
    pretrain_iters: int = 5000
    gan_iters: int = 5000
    lr_generator: float = 2e-4
    lr_critic: float = 1e-4
    l1_weight: float = 1.0
    gradient_weight: float = 0.5
    adversarial_weight: float = 1e-3
    critic_features: int = 32
    ema_decay: float = 0.999
    noise_range: tuple[float, float] = (0.5, 1.5)
    foreground_fraction: float = 0.9
    holdout_fraction: float = 0.1
    val_patches: int = 32
    val_every: int = 250
    checkpoint_every: int = 500
    log_every: int = 50
    seed: int = 0

    def __post_init__(self):
        self.scale = as_scale(self.scale)
        if self.lr_patch < 4:
            raise ValueError("lr_patch must be at least 4 voxels")
        if self.pretrain_iters < 0 or self.gan_iters < 0:
            raise ValueError("iteration counts must be non-negative")
        if self.pretrain_iters + self.gan_iters == 0:
            raise ValueError("nothing to train: pretrain_iters + gan_iters is 0")

    @property
    def hr_patch(self) -> tuple[int, int, int]:
        return tuple(self.lr_patch * s for s in self.scale)  # type: ignore[return-value]

    @property
    def total_iters(self) -> int:
        return self.pretrain_iters + self.gan_iters

    def to_json(self) -> dict:
        data = asdict(self)
        data["scale"] = list(self.scale)
        data["noise_range"] = list(self.noise_range)
        return data
