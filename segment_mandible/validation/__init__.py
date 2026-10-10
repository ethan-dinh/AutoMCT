"""
Automated sanity checks for the segmentation pipeline.

The pipeline has no ground truth to score against, so every check here is a
*plausibility* test: it encodes something we know about a mouse hemimandible
and flags an output that violates it. The point is to catch the failure modes
that are obvious to a human glancing at a napari window -- the mask swallowed
the whole jaw, the incisor broke into pieces, the seed latched onto a molar --
without anyone having to open that window for every sample.
"""

from .checks import (
    AreaSpike,
    CheckReport,
    CheckResult,
    Severity,
    find_area_spikes,
    check_incisor_mask,
    check_incisor_seed,
    check_isolation_volume,
    check_loaded_volume,
    check_preprocessed_volume,
    check_reorientation,
    check_shrink,
)
from .summary import SampleValidation, ValidationSummary

__all__ = [
    "AreaSpike",
    "CheckReport",
    "CheckResult",
    "SampleValidation",
    "Severity",
    "ValidationSummary",
    "find_area_spikes",
    "check_incisor_mask",
    "check_incisor_seed",
    "check_isolation_volume",
    "check_loaded_volume",
    "check_preprocessed_volume",
    "check_reorientation",
    "check_shrink",
]
