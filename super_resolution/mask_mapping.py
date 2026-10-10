"""
Carrying masks from the upscaled grid back onto the original scan.

The classical pipeline hands back masks in its *reoriented* frame: cropped to
the specimen, axes permuted and flipped (see ``reorient_mandible``). Every one
of those steps is lossless and recorded, so the way back is exact:

1. Undo the flips and permutation. This yields the mask over the pipeline's
   crop box, in the upscaled volume's own axes -- which are the original
   scan's axes, each subdivided ``s`` times.
2. Count, for every original voxel, how many of its s_z * s_y * s_x fine
   voxels each structure covers.
3. Vote: an original voxel is tissue when at least ``min_fraction`` of its
   fine voxels are, and it goes to whichever structure covers most of it.
4. Re-apply the same crop, permutation and flips at the original resolution,
   so the result lines up with the original scan reoriented the same way --
   the layout ``segment_mandible`` writes its own outputs in.

The counts are kept, not just the vote: divided by s^3 they are the fraction
of each original voxel that the fine-grid segmentation assigns to a structure,
which gives a volume estimate finer than whole voxels.
"""

from __future__ import annotations

import numpy as np

STRUCTURES = ("incisor", "bone", "molar")

# Label values for a single label map, matching ML-Segmentation's encoding.
LABELS = {"incisor": 1, "bone": 2, "molar": 3}

# When two structures cover equally many fine voxels of one original voxel,
# the earlier one here wins. Molar over bone over incisor follows
# ML-Segmentation's precedence for overlapping masks. Such ties are rare: the
# pipeline carves a gap between the incisor and the bone around it.
TIE_PRIORITY = ("molar", "bone", "incisor")

Box = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def record_bbox(record: dict | None, shape: tuple[int, ...]) -> Box:
    """The crop box a reorientation record applied, or the whole volume when it cropped nothing."""
    bbox = (record or {}).get("bbox")
    if bbox is None:
        return tuple((0, int(n)) for n in shape)  # type: ignore[return-value]
    return tuple((int(lo), int(hi)) for lo, hi in bbox)  # type: ignore[return-value]


def reorient(volume: np.ndarray, record: dict | None) -> np.ndarray:
    """
    Apply a ``reorient_mandible`` record to another array in the same input frame.

    Replays the recorded steps in the order ``reorient_mandible`` performs them
    -- crop, permute, flip axis 0, swap axes 1/2, flip axis 1 -- so an array
    given the same record lands voxel-for-voxel on the pipeline's output.
    Returns a view where possible; copy it if a contiguous array is needed.
    """
    record = record or {}
    bbox = record.get("bbox")
    if bbox is not None:
        volume = volume[tuple(slice(int(lo), int(hi)) for lo, hi in bbox)]
    perm = record.get("perm")
    if perm is not None:
        volume = np.transpose(volume, tuple(int(a) for a in perm))
    if record.get("flip0"):
        volume = np.flip(volume, axis=0)
    if record.get("swap12"):
        volume = np.swapaxes(volume, 1, 2)
    if record.get("flip1"):
        volume = np.flip(volume, axis=1)
    return volume


def unreorient(reoriented: np.ndarray, record: dict | None) -> np.ndarray:
    """
    Undo everything but the crop: the reoriented array back in input-frame axis order.

    The result covers the record's crop box. It is a view; nothing is copied.
    """
    record = record or {}
    volume = reoriented
    if record.get("flip1"):
        volume = np.flip(volume, axis=1)
    if record.get("swap12"):
        volume = np.swapaxes(volume, 1, 2)
    if record.get("flip0"):
        volume = np.flip(volume, axis=0)
    perm = record.get("perm")
    if perm is not None:
        volume = np.transpose(volume, tuple(int(a) for a in np.argsort(perm)))
    return volume


def coarse_box(fine_box: Box, scale: tuple[int, int, int]) -> Box:
    """The smallest box of original voxels that contains a box of fine voxels."""
    return tuple(
        (lo // s, -(-hi // s)) for (lo, hi), s in zip(fine_box, scale)
    )  # type: ignore[return-value]


def coarse_record(record: dict | None, scale: tuple[int, int, int], fine_shape) -> dict:
    """
    The fine-grid record restated for the original grid.

    Only the crop box changes; the permutation and flips are properties of the
    specimen's orientation, not of the voxel size, and are reused unchanged.
    """
    record = dict(record or {})
    record["bbox"] = coarse_box(record_bbox(record, fine_shape), scale)
    return record


def count_fine_voxels(
    fine_region: np.ndarray,
    fine_box: Box,
    scale: tuple[int, int, int],
    *,
    slab: int = 16,
) -> np.ndarray:
    """
    For each original voxel in ``coarse_box(fine_box)``, how many of its fine voxels are set.

    ``fine_region`` is a mask over ``fine_box`` of the fine grid (any array
    layout -- typically the non-contiguous view from :func:`unreorient`). The
    box need not align with the scale: a partial block at its edge counts only
    the fine voxels inside it, the rest being outside the pipeline's crop and
    therefore not tissue. Works through the volume in slabs, so memory stays
    small however large the fine grid is.
    """
    sz, sy, sx = scale
    box = coarse_box(fine_box, scale)
    coarse_shape = tuple(hi - lo for lo, hi in box)
    # Where the fine region starts inside the padded, block-aligned grid.
    pad = tuple(f_lo - c_lo * s for (f_lo, _), (c_lo, _), s in zip(fine_box, box, scale))
    fine_len = fine_region.shape
    if tuple(hi - lo for lo, hi in fine_box) != tuple(fine_len):
        raise ValueError(f"fine_region shape {fine_len} does not match box {fine_box}")

    per_block = sz * sy * sx
    dtype = np.uint8 if per_block <= np.iinfo(np.uint8).max else np.uint16
    counts = np.zeros(coarse_shape, dtype=dtype)
    _, cy, cx = coarse_shape

    for c0 in range(0, coarse_shape[0], slab):
        c1 = min(c0 + slab, coarse_shape[0])
        # Fine rows covered by coarse rows [c0, c1), in fine_region's coordinates.
        f0 = max(c0 * sz - pad[0], 0)
        f1 = min(c1 * sz - pad[0], fine_len[0])
        if f1 <= f0:
            continue
        buffer = np.zeros(((c1 - c0) * sz, cy * sy, cx * sx), dtype=np.uint8)
        z_at = f0 + pad[0] - c0 * sz
        buffer[
            z_at:z_at + (f1 - f0),
            pad[1]:pad[1] + fine_len[1],
            pad[2]:pad[2] + fine_len[2],
        ] = fine_region[f0:f1]
        counts[c0:c1] = buffer.reshape(c1 - c0, sz, cy, sy, cx, sx).sum(
            axis=(1, 3, 5), dtype=dtype,
        )
    return counts


def vote(
    counts: dict[str, np.ndarray],
    fine_per_voxel: int,
    min_fraction: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Binary masks on the original grid from per-structure fine-voxel counts.

    Tissue versus background is decided on the *total* tissue coverage, and
    only then is the voxel given to the structure covering most of it. Voting
    each structure against background separately would drop a voxel split
    3/3 between incisor and bone at 2x (6 of 8 fine voxels are tissue) --
    punching holes along every interface between structures.

    A voxel exactly half covered counts as tissue at the default
    ``min_fraction`` of 0.5, the usual partial-volume convention.
    """
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction}")
    names = [name for name in TIE_PRIORITY if name in counts]
    names += [name for name in counts if name not in names]
    stack = np.stack([counts[name] for name in names])
    total = stack.sum(axis=0, dtype=np.uint32)
    tissue = total >= min_fraction * fine_per_voxel
    winner = np.argmax(stack, axis=0)
    return {name: tissue & (winner == i) for i, name in enumerate(names)}
