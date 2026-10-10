"""
Tests for carrying masks from the upscaled grid back to the original.

The claims under test: replaying a reorientation record lands exactly on
``reorient_mandible``'s own output (so masks and volume share one frame);
undoing it is exact; fine-voxel counts are exact even when the pipeline's crop
box does not align with the scale; and the vote keeps voxels split between two
structures instead of punching holes along their interface.

Run with:  python -m pytest super_resolution/tests
"""

# pylint: disable=wrong-import-position

import sr_phantoms as P

import numpy as np
import pytest

from preprocessing import reorient_mandible  # segment_mandible, on sys.path via sr_phantoms

from super_resolution.mask_mapping import (
    STRUCTURES,
    coarse_box,
    coarse_record,
    count_fine_voxels,
    record_bbox,
    reorient,
    unreorient,
    vote,
)


def _pipeline_reorient(volume, companions):
    record: dict = {}
    reoriented, outputs = reorient_mandible(volume, companion_volumes=companions, record=record)
    return reoriented, outputs, record


def test_phantom_exercises_every_reorientation_step():
    volume, _ = P.jaw_phantom()
    _, _, record = _pipeline_reorient(volume, [])
    assert record["bbox"] is not None
    assert record["perm"] is not None
    assert record["flip0"] and record["swap12"] and record["flip1"]


def test_replaying_a_record_matches_reorient_mandible():
    volume, parts = P.jaw_phantom()
    labels = parts["tip"].astype(np.uint8) * 2 + parts["body"].astype(np.uint8)
    reoriented, (labels_out,), record = _pipeline_reorient(volume, [labels])

    assert np.array_equal(reorient(volume, record), reoriented)
    assert np.array_equal(reorient(labels, record), labels_out)


def test_unreorient_inverts_everything_but_the_crop():
    volume, _ = P.jaw_phantom()
    reoriented, _, record = _pipeline_reorient(volume, [])
    box = record_bbox(record, volume.shape)
    cropped = volume[tuple(slice(lo, hi) for lo, hi in box)]
    assert np.array_equal(unreorient(reoriented, record), cropped)


def test_empty_record_is_the_identity():
    volume = np.arange(24).reshape(2, 3, 4)
    assert np.array_equal(reorient(volume, {}), volume)
    assert np.array_equal(unreorient(volume, None), volume)
    assert record_bbox({"bbox": None}, volume.shape) == ((0, 2), (0, 3), (0, 4))


def test_coarse_box_is_the_smallest_cover():
    assert coarse_box(((3, 9), (4, 8), (0, 1)), (2, 2, 2)) == ((1, 5), (2, 4), (0, 1))
    assert coarse_box(((5, 7), (0, 9), (2, 3)), (3, 1, 2)) == ((1, 3), (0, 9), (1, 2))


@pytest.mark.parametrize("scale", [(2, 2, 2), (1, 2, 3), (3, 3, 3)])
def test_counts_match_a_direct_block_sum_for_unaligned_boxes(scale):
    rng = np.random.default_rng(3)
    coarse_shape = (7, 6, 5)
    fine = rng.random(tuple(n * s for n, s in zip(coarse_shape, scale))) > 0.5
    fine_box = tuple(
        (int(rng.integers(0, n // 3)), int(rng.integers(2 * n // 3, n + 1))) for n in fine.shape
    )
    region = fine[tuple(slice(lo, hi) for lo, hi in fine_box)]

    counts = count_fine_voxels(region, fine_box, scale, slab=2)

    inside = np.zeros_like(fine)
    inside[tuple(slice(lo, hi) for lo, hi in fine_box)] = region
    expected = P.block_sum(inside, scale)[tuple(slice(lo, hi) for lo, hi in coarse_box(fine_box, scale))]
    assert counts.shape == expected.shape
    assert np.array_equal(counts, expected)


def test_counts_accept_the_non_contiguous_view_unreorient_returns():
    volume, _ = P.jaw_phantom()
    fine = P.upsample_nearest(volume, (2, 2, 2))
    mask = fine > 0.7
    reoriented, (mask_out,), record = _pipeline_reorient(fine, [mask])
    del reoriented
    counts = count_fine_voxels(unreorient(mask_out, record), record_bbox(record, fine.shape), (2, 2, 2))
    box = coarse_box(record_bbox(record, fine.shape), (2, 2, 2))
    expected = P.block_sum(mask, (2, 2, 2))[tuple(slice(lo, hi) for lo, hi in box)]
    assert np.array_equal(counts, expected)


def _counts(**values):
    return {name: np.array([values.get(name, 0)], dtype=np.uint8) for name in STRUCTURES}


def test_vote_keeps_a_voxel_split_between_two_structures():
    masks = vote(_counts(incisor=3, bone=3), 8)
    assert masks["bone"][0] and not masks["incisor"][0]  # tie: bone outranks incisor


def test_vote_half_covered_is_tissue_and_less_is_not():
    assert vote(_counts(incisor=4), 8)["incisor"][0]
    assert not vote(_counts(incisor=3), 8)["incisor"][0]
    assert not vote(_counts(incisor=7), 8, min_fraction=1.0)["incisor"][0]


def test_vote_ties_follow_molar_bone_incisor_and_masks_are_disjoint():
    rng = np.random.default_rng(0)
    counts = {}
    remaining = np.full(500, 8)
    for name in STRUCTURES:
        counts[name] = rng.integers(0, remaining + 1).astype(np.uint8)
        remaining = remaining - counts[name]
    masks = vote(counts, 8)
    overlap = sum(m.astype(int) for m in masks.values())
    assert overlap.max() <= 1
    assert vote(_counts(molar=2, bone=2, incisor=1), 8)["molar"][0]


def test_vote_rejects_a_meaningless_fraction():
    with pytest.raises(ValueError):
        vote(_counts(bone=8), 8, min_fraction=0.0)


def test_round_trip_through_the_real_reorientation():
    """Fine masks -> reorient_mandible -> map back == voting the fine masks directly."""
    scale = (2, 2, 2)
    coarse, _ = P.jaw_phantom()
    fine = P.upsample_nearest(coarse, scale)
    body = fine > 0
    # Boundaries that do not follow the coarse grid, so voting has real work to do.
    incisor = P.ellipsoid(fine.shape, (20.5, 60.3, 21.7), (9.2, 30.1, 5.3)) & body
    molar = np.zeros_like(body)
    molar[9:25, 31:57, 13:28] = True
    molar &= body & ~incisor
    bone = body & ~incisor & ~molar
    fine_masks = {"incisor": incisor, "bone": bone, "molar": molar}

    _, outputs, record = _pipeline_reorient(fine, [fine_masks[s] for s in STRUCTURES])
    fine_box = record_bbox(record, fine.shape)
    counts = {
        s: count_fine_voxels(unreorient(out, record), fine_box, scale)
        for s, out in zip(STRUCTURES, outputs)
    }
    mapped = vote(counts, 8)

    box = tuple(slice(lo, hi) for lo, hi in coarse_box(fine_box, scale))
    direct = vote({s: P.block_sum(m, scale)[box] for s, m in fine_masks.items()}, 8)
    for s in STRUCTURES:
        assert np.array_equal(mapped[s], direct[s]), s
    assert mapped["incisor"].any() and mapped["bone"].any() and mapped["molar"].any()

    # The same record, restated for the coarse grid, reproduces what the
    # pipeline would have done to the coarse scan itself.
    coarse_rec = coarse_record(record, scale, fine.shape)
    reoriented_coarse, _, own_record = _pipeline_reorient(coarse, [])
    assert np.array_equal(reorient(coarse, coarse_rec), reoriented_coarse)
    assert own_record["perm"] == coarse_rec["perm"]
    unboxed = {**coarse_rec, "bbox": None}
    assert reorient(mapped["bone"], unboxed).shape == reoriented_coarse.shape
