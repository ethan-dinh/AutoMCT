"""
Training and inference configuration for mandible segmentation.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal
import json


@dataclass
class Config:
    # --- Paths ---
    data_dir: str = "../segment_mandible/tests/segmentation_results"
    output_dir: str = "./runs"

    # --- Data ---
    num_classes: int = 4  # background, incisor, bone, molar
    class_names: list[str] = field(
        default_factory=lambda: ["background", "incisor", "bone", "molar"]
    )
    patch_size: tuple[int, int, int] = (96, 96, 96)
    samples_per_volume: int = 16  # random patches extracted per volume per epoch
    pos_sample_ratio: float = 0.7  # fraction of patches guaranteed to contain foreground

    # --- Model ---
    model: Literal["unet", "segresnet"] = "unet"
    in_channels: int = 1
    channels: tuple[int, ...] = (32, 64, 128, 256)
    strides: tuple[int, ...] = (2, 2, 2)
    dropout: float = 0.2

    # --- Training ---
    max_epochs: int = 300
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: Literal["cosine", "plateau"] = "cosine"
    early_stop_patience: int = 50

    # --- Cross-validation ---
    n_folds: int = 9  # 9 = leave-one-out with 9 samples
    fold: int | None = None  # None = run all folds; int = run a single fold

    # --- Iterative retraining ---
    resume_checkpoint: str | None = None  # path to checkpoint to resume from

    # --- Inference ---
    sw_batch_size: int = 4  # sliding-window batch size
    sw_overlap: float = 0.5  # sliding-window overlap ratio

    # --- Device ---
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = json.load(f)
        # Convert lists back to tuples where needed
        for key in ("patch_size", "channels", "strides"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)
