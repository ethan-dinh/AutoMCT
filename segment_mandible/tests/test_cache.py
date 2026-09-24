"""
Tests for the filter cache's key.

The cache stores the normalized, denoised volume -- the pipeline's most
expensive intermediate -- so the only thing that really needs guarding is that
its key changes whenever the cached bytes would no longer be right for the
input. A cache that is merely slow is an annoyance; one that silently serves
another scan's denoised data produces a wrong segmentation that looks fine.

Run with:  python -m pytest segment_mandible/tests/test_cache.py
"""

# The package imports below have to follow the sys.path setup that makes
# segment_mandible/ importable when this file is run directly.
# pylint: disable=wrong-import-position

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import _volume_fingerprint  # noqa: E402


def _volume(seed=0, shape=(40, 30, 25)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


def test_same_input_gives_the_same_key():
    """The cache is useless if the key is not stable across runs."""
    vol = _volume()
    assert _volume_fingerprint("/data/a.nrrd", vol) == _volume_fingerprint("/data/a.nrrd", vol)


def test_moved_file_keeps_its_key():
    """
    Moving a scan to another drive must not invalidate its cache.

    The cached volume is a pure function of the array, so the same content
    under a different path has the same correct cached result. Keying on the
    path turned a drive move into a silent full re-denoise.
    """
    vol = _volume()
    assert _volume_fingerprint("/Volumes/A/s.nrrd", vol) == _volume_fingerprint("/Volumes/B/s.nrrd", vol)


def test_edited_contents_change_the_key():
    """
    Re-exporting a scan in place must invalidate its cache.

    The fingerprint samples the volume rather than hashing all of it, so this
    is the property most at risk from that shortcut: the edit has to be one the
    sampling actually sees. A change spread across the volume, as a re-export
    would be, is the realistic case.
    """
    vol = _volume()
    edited = vol.copy()
    edited[::3, ::3, ::3] = (edited[::3, ::3, ::3].astype(int) + 31) % 255
    assert _volume_fingerprint("/data/a.nrrd", vol) != _volume_fingerprint("/data/a.nrrd", edited)


def test_shape_change_changes_the_key():
    """A crop keeps most voxels identical, so shape must be keyed explicitly."""
    vol = _volume()
    assert _volume_fingerprint("/data/a.nrrd", vol) != _volume_fingerprint("/data/a.nrrd", vol[:-1])


def test_dtype_change_changes_the_key():
    """
    The denoised result depends on the input's type, so the key must too.

    Same numbers at a different width normalize differently, and the cached
    volume would no longer correspond to the input.
    """
    vol = _volume()
    assert (
        _volume_fingerprint("/data/a.nrrd", vol)
        != _volume_fingerprint("/data/a.nrrd", vol.astype(np.uint16))
    )


def test_key_is_a_short_filename_safe_string():
    """The key is pasted into a filename, so it must not need escaping."""
    key = _volume_fingerprint("/data/a.nrrd", _volume())
    assert key and len(key) <= 32
    assert all(c in "0123456789abcdef" for c in key)


def test_fingerprint_is_cheap_on_a_large_volume():
    """
    The fingerprint must not undo the saving it enables.

    It samples with a stride chosen from the volume's size, so cost should stay
    roughly flat as the volume grows. A regression to hashing everything would
    show up here as a long runtime on an array that is large but still small
    next to a real scan.
    """
    import time

    big = np.zeros((400, 300, 300), dtype=np.uint8)
    start = time.perf_counter()
    _volume_fingerprint("/data/a.nrrd", big)
    assert time.perf_counter() - start < 2.0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
