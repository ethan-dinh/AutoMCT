"""
Shared phantoms for the super_resolution tests.

The jaw phantom is built so that segment_mandible's ``reorient_mandible``
exercises *every* step it can record -- a crop, an axis permutation, a flip of
axis 0, a swap of axes 1/2 and a flip of axis 1. A mapping that got any one of
them backwards would then fail the tests rather than pass by coincidence.
"""

# pylint: disable=wrong-import-position

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import super_resolution  # noqa: E402,F401  (sets the torch environment variables first)
from super_resolution.classical import add_segment_mandible_to_path  # noqa: E402

add_segment_mandible_to_path()

# Coarse-grid layout (Z, Y, X). The body's bounding box is 20 x 50 x 12, so
# its long axis is Y (-> permutation); the tip sits at low Y (-> flip0); after
# the permutation the old Z extent (20) exceeds the old X extent (12) (->
# swap12); and the tip is at high X, which lands above the midline (-> flip1).
SHAPE = (24, 60, 30)
BODY = (slice(2, 22), slice(5, 55), slice(5, 17))
TIP = (slice(18, 21), slice(6, 9), slice(13, 16))


def jaw_phantom(body_value: float = 0.5, tip_value: float = 1.0, air_value: float = 0.0,
                dtype=np.float32) -> tuple[np.ndarray, dict]:
    """A hemimandible-like block with a small bright 'incisor tip' near one end."""
    volume = np.full(SHAPE, air_value, dtype=dtype)
    body = np.zeros(SHAPE, dtype=bool)
    body[BODY] = True
    tip = np.zeros(SHAPE, dtype=bool)
    tip[TIP] = True
    volume[body] = body_value
    volume[tip] = tip_value
    return volume, {"body": body, "tip": tip}


def upsample_nearest(volume: np.ndarray, scale) -> np.ndarray:
    for axis, s in enumerate(scale):
        volume = np.repeat(volume, s, axis=axis)
    return volume


def block_sum(mask: np.ndarray, scale) -> np.ndarray:
    sz, sy, sx = scale
    z, y, x = (n // s for n, s in zip(mask.shape, scale))
    return mask.reshape(z, sz, y, sy, x, sx).sum(axis=(1, 3, 5))


def ellipsoid(shape, centre, radii) -> np.ndarray:
    grids = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return sum(((g - c) / r) ** 2 for g, c, r in zip(grids, centre, radii)) <= 1.0
