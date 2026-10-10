"""
Training the generator: a fidelity phase, then an adversarial phase (ESRGAN's recipe).

1. **Pretrain** on L1 plus an L1 on spatial gradients. This alone gives a
   faithful but somewhat smooth upscaler; it is saved as ``model_l1.pt``.
2. **Adversarial fine-tune**: the same losses plus a small relativistic GAN
   term from a 3-D patch critic, which restores the edge sharpness L1 averages
   away. Saved as ``model.pt``.

The adversarial weight is deliberately small (1e-3 against an L1 weight of 1),
and the data-consistency projection pins every original voxel's mean, so the
critic can sharpen boundaries but cannot move them far or invent structure.
Set ``--gan-iters 0`` for a purely fidelity-trained model.

Both saved models hold an exponential moving average of the generator's
weights rather than the last step's, which is smoother and consistently a
little better for SR. Validation compares the model against plain trilinear
interpolation on held-out anatomy; a model that does not beat trilinear there
is not worth segmenting with, and the log says so.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from .config import Scale, TrainConfig
from .model import (
    PatchCritic3d,
    SRGenerator3d,
    block_mean,
    describe_device,
    gradient_l1,
    relativistic_losses,
    save_generator,
)
from .patches import PatchSampler, prepare_volume
from .volume_io import Scan

logger = logging.getLogger(__name__)

STATE_FILE = "last_state.pt"
MODEL_FILE = "model.pt"
PRETRAIN_MODEL_FILE = "model_l1.pt"
HISTORY_FILE = "history.json"


def degrade(
    high: torch.Tensor,
    scale: Scale,
    sigma: torch.Tensor,
    noise_range: tuple[float, float],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Simulate the scanner one scale down: block-average, then restore the noise.

    Averaging s^3 voxels divides white noise variance by s^3, so noise of
    variance sigma^2 (1 - 1/s^3) is added back to return it to the scan's own
    level, scaled per patch by a random factor from ``noise_range``.
    """
    low = block_mean(high, scale)
    keep = 1.0 - 1.0 / math.prod(scale)
    lo, hi = noise_range
    factor = lo + (hi - lo) * torch.rand(
        (high.shape[0], 1, 1, 1, 1), generator=generator, device="cpu",
    ).to(high.device)
    std = sigma.view(-1, 1, 1, 1, 1).to(high.device) * math.sqrt(keep) * factor
    noise = torch.randn(low.shape, generator=generator, device="cpu").to(high.device)
    return low + std * noise


def _psnr(mse: float) -> float:
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


class _Trainer:  # pylint: disable=too-many-instance-attributes
    """State for one run; split out so resume can rebuild it from a checkpoint."""

    def __init__(self, scans: list[Scan], out_dir: str, config: TrainConfig, device: torch.device):
        self.config = config
        self.out_dir = out_dir
        self.device = device
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        self.volumes = [
            prepare_volume(scan, config.hr_patch, self.rng, holdout_fraction=config.holdout_fraction)
            for scan in scans
        ]
        self.sampler = PatchSampler(
            self.volumes, config.hr_patch, self.rng,
            foreground_fraction=config.foreground_fraction,
        )

        self.generator = SRGenerator3d(
            scale=config.scale, features=config.features, growth=config.growth,
            blocks=config.blocks, consistency=config.consistency,
        ).to(device)
        self.ema = copy.deepcopy(self.generator).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.critic = PatchCritic3d(config.critic_features).to(device)

        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=config.lr_generator, betas=(0.9, 0.99))
        self.opt_d = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic, betas=(0.9, 0.99))

        self.iteration = 0
        self.history: list[dict] = []

        # A fixed validation set, degraded once with a fixed noise draw, so
        # every evaluation during the run scores exactly the same inputs.
        val_rng = np.random.default_rng(config.seed + 1)
        saved_rng, self.sampler.rng = self.sampler.rng, val_rng
        val_high, val_sigma = self.sampler.validation_set(config.val_patches)
        self.sampler.rng = saved_rng
        noise_gen = torch.Generator().manual_seed(config.seed + 2)
        high = torch.from_numpy(val_high)
        low = degrade(high, config.scale, torch.from_numpy(val_sigma), (1.0, 1.0), noise_gen)
        self.val_high, self.val_low = high, low

    # -- schedule -----------------------------------------------------------

    def _phase(self, iteration: int) -> str:
        return "pretrain" if iteration < self.config.pretrain_iters else "gan"

    def _learning_rate(self, base: float, iteration: int) -> float:
        """Halve at 50% and again at 75% of each phase (ESRGAN's step schedule)."""
        if self._phase(iteration) == "pretrain":
            start, length = 0, self.config.pretrain_iters
        else:
            start, length = self.config.pretrain_iters, self.config.gan_iters
        progress = (iteration - start) / max(length, 1)
        return base * (0.5 ** sum(progress >= m for m in (0.5, 0.75)))

    # -- one step -----------------------------------------------------------

    def _update_ema(self) -> None:
        # Short horizon early on so the average is not dominated by the random
        # initialisation; it settles to ema_decay after a few hundred steps.
        decay = min(self.config.ema_decay, (1 + self.iteration) / (10 + self.iteration))
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), self.generator.parameters()):
                ema_p.lerp_(p.detach(), 1.0 - decay)

    def step(self) -> dict:
        cfg = self.config
        phase = self._phase(self.iteration)
        for group in self.opt_g.param_groups:
            group["lr"] = self._learning_rate(cfg.lr_generator, self.iteration)
        for group in self.opt_d.param_groups:
            group["lr"] = self._learning_rate(cfg.lr_critic, self.iteration)

        high_np, sigma_np = self.sampler.sample(cfg.batch_size)
        high = torch.from_numpy(high_np).to(self.device)
        low = degrade(high, cfg.scale, torch.from_numpy(sigma_np), cfg.noise_range)

        self.generator.train()
        sr = self.generator(low)
        pixel = F.l1_loss(sr, high)
        edges = gradient_l1(sr, high)
        loss = cfg.l1_weight * pixel + cfg.gradient_weight * edges
        stats = {"l1": pixel.item(), "grad": edges.item()}

        if phase == "gan":
            for param in self.critic.parameters():
                param.requires_grad_(False)
            _, adversarial = relativistic_losses(self.critic(high).detach(), self.critic(sr))
            loss = loss + cfg.adversarial_weight * adversarial
            stats["adv"] = adversarial.item()

        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_g.step()
        self._update_ema()

        if phase == "gan":
            for param in self.critic.parameters():
                param.requires_grad_(True)
            critic_loss, _ = relativistic_losses(self.critic(high), self.critic(sr.detach()))
            self.opt_d.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.opt_d.step()
            stats["critic"] = critic_loss.item()

        self.iteration += 1
        return stats

    # -- evaluation ---------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict:
        """PSNR of the EMA model and of trilinear interpolation on the held-out patches."""
        sr_err, tri_err, n = 0.0, 0.0, 0
        for start in range(0, len(self.val_low), self.config.batch_size):
            low = self.val_low[start:start + self.config.batch_size].to(self.device)
            high = self.val_high[start:start + self.config.batch_size].to(self.device)
            sr = self.ema(low)
            tri = F.interpolate(low, scale_factor=self.config.scale, mode="trilinear", align_corners=False)
            count = high.numel()
            sr_err += F.mse_loss(sr, high, reduction="sum").item()
            tri_err += F.mse_loss(tri, high, reduction="sum").item()
            n += count
        sr_psnr, tri_psnr = _psnr(sr_err / n), _psnr(tri_err / n)
        return {
            "iteration": self.iteration,
            "phase": self._phase(max(self.iteration - 1, 0)),
            "psnr_model": sr_psnr,
            "psnr_trilinear": tri_psnr,
            "gain_db": sr_psnr - tri_psnr,
        }

    # -- persistence --------------------------------------------------------

    def metadata(self) -> dict:
        return {
            "trained_on": [
                {
                    "name": v.name,
                    "spacing_mm_zyx": [float(s) for s in v.spacing] if v.spacing_known else None,
                    "intensity_low": float(v.intensity.low),
                    "intensity_high": float(v.intensity.high),
                    "noise_sigma": float(v.noise_sigma),
                    "holdout_slices": list(v.holdout) if v.holdout else None,
                }
                for v in self.volumes
            ],
            "train_config": self.config.to_json(),
            "iterations": int(self.iteration),
            "validation": self.history[-1] if self.history else None,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "torch_version": str(torch.__version__),  # TorchVersion is not weights_only-safe
        }

    def save_state(self) -> None:
        state = {
            "generator": self.generator.state_dict(),
            "ema": self.ema.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "iteration": self.iteration,
            "history": self.history,
            "config": self.config.to_json(),
            "numpy_rng": self.rng.bit_generator.state,
        }
        path = os.path.join(self.out_dir, STATE_FILE)
        torch.save(state, f"{path}.tmp")
        os.replace(f"{path}.tmp", path)

    def load_state(self, path: str) -> None:
        # Tensors, numbers, strings and lists only, so the safe loader suffices.
        state = torch.load(path, map_location=self.device, weights_only=True)
        saved = dict(state["config"])
        current = self.config.to_json()
        for key in ("scale", "features", "growth", "blocks", "consistency", "lr_patch"):
            if saved.get(key) != current.get(key):
                raise ValueError(
                    f"Cannot resume: {key} was {saved.get(key)!r} in the saved run "
                    f"but is {current.get(key)!r} now. Use the original settings or a "
                    "new output directory."
                )
        self.generator.load_state_dict(state["generator"])
        self.ema.load_state_dict(state["ema"])
        self.critic.load_state_dict(state["critic"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.iteration = int(state["iteration"])
        self.history = list(state["history"])
        self.rng.bit_generator.state = state["numpy_rng"]

    def write_history(self) -> None:
        with open(os.path.join(self.out_dir, HISTORY_FILE), "w", encoding="utf-8") as handle:
            json.dump({"config": self.config.to_json(), "validation": self.history}, handle, indent=2)


def train_model(
    scans: list[Scan],
    out_dir: str,
    config: TrainConfig,
    device: torch.device,
    *,
    resume: bool = False,
) -> str:
    """
    Train a generator on ``scans`` and return the path of the model to use.

    ``out_dir`` receives ``model.pt`` (the final EMA generator), ``model_l1.pt``
    (the end of the fidelity phase), ``history.json`` (validation scores) and
    ``last_state.pt`` (everything needed to resume with ``resume=True``).
    """
    os.makedirs(out_dir, exist_ok=True)
    if device.type == "cuda":
        # Fixed patch sizes: let cuDNN pick the fastest 3-D convolution once.
        torch.backends.cudnn.benchmark = True

    trainer = _Trainer(scans, out_dir, config, device)
    state_path = os.path.join(out_dir, STATE_FILE)
    if resume and os.path.exists(state_path):
        trainer.load_state(state_path)
        logger.info("Resumed from %s at iteration %d", state_path, trainer.iteration)
    elif resume:
        logger.info("No %s in %s; starting from scratch", STATE_FILE, out_dir)
    elif os.path.exists(os.path.join(out_dir, MODEL_FILE)):
        logger.warning("%s already holds a model; it will be overwritten", out_dir)

    n_params = sum(p.numel() for p in trainer.generator.parameters())
    logger.info(
        "Training %.2fM-parameter generator on %s: scale %s, LR patch %d^3 (HR %s), "
        "batch %d, %d fidelity + %d adversarial iterations",
        n_params / 1e6, describe_device(device), "x".join(map(str, config.scale)),
        config.lr_patch, "x".join(map(str, config.hr_patch)), config.batch_size,
        config.pretrain_iters, config.gan_iters,
    )
    if trainer.iteration == 0:
        baseline = trainer.validate()
        logger.info(
            "Before training: trilinear %.2f dB, model %.2f dB on held-out patches",
            baseline["psnr_trilinear"], baseline["psnr_model"],
        )

    tick, running, steps = time.perf_counter(), {}, 0
    while trainer.iteration < config.total_iters:
        stats = trainer.step()
        steps += 1
        for key, value in stats.items():
            running[key] = running.get(key, 0.0) + value
        it = trainer.iteration

        if it % config.log_every == 0 or it == config.total_iters:
            elapsed = time.perf_counter() - tick
            logger.info(
                "[%s %d/%d] %s | %.2f it/s",
                trainer._phase(it - 1), it, config.total_iters,  # pylint: disable=protected-access
                "  ".join(f"{k} {v / steps:.4g}" for k, v in running.items()),
                steps / max(elapsed, 1e-9),
            )
            tick, running, steps = time.perf_counter(), {}, 0

        if it % config.val_every == 0 or it in (config.pretrain_iters, config.total_iters):
            result = trainer.validate()
            trainer.history.append(result)
            logger.info(
                "Validation @%d: model %.2f dB vs trilinear %.2f dB (%+.2f dB)",
                it, result["psnr_model"], result["psnr_trilinear"], result["gain_db"],
            )
            trainer.write_history()

        if it == config.pretrain_iters and config.pretrain_iters > 0:
            save_generator(os.path.join(out_dir, PRETRAIN_MODEL_FILE), trainer.ema, trainer.metadata())
            logger.info("Fidelity phase done; saved %s", PRETRAIN_MODEL_FILE)

        if it % config.checkpoint_every == 0 or it == config.total_iters:
            trainer.save_state()

    model_path = os.path.join(out_dir, MODEL_FILE)
    save_generator(model_path, trainer.ema, trainer.metadata())
    trainer.write_history()

    final = trainer.history[-1] if trainer.history else trainer.validate()
    if final["gain_db"] <= 0:
        logger.warning(
            "The model does not beat trilinear interpolation on held-out anatomy "
            "(%+.2f dB). Train longer, or check that the scans are what you expect, "
            "before segmenting with it.", final["gain_db"],
        )
    logger.info("Saved %s (%+.2f dB over trilinear)", model_path, final["gain_db"])
    return model_path
