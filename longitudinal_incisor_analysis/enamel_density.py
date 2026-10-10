"""
Longitudinal enamel density profiling for the mouse incisor.

Author: Ethan Dinh
Date:   2026-09-09

The mouse incisor grows continuously, so its length is a developmental time
axis: enamel at the cervical loop is newly deposited and poorly mineralized,
enamel at the incisal tip is fully mature. Measuring enamel density against
position along the tooth therefore measures the mineralization trajectory.

This module deliberately does NOT segment enamel. `segment_enamel/enamel_seg.py`
takes that route and its own README records where it breaks: enamel is brighter
than dentin near the tip and dimmer near the base, so a per-slice intensity
label silently flips partway down z and precision decays 0.77 -> 0.27 with no
warning. Here we ask a purely statistical question of each slice's intensity
histogram -- "is there a mineral population distinguishable from dentin, and
where is it centred?" -- and, critically, we are allowed to answer "no".

    resliced_volume.tif  (CMPR output, z straightened along the tooth)
        |
        v
    [0] Per-slice dentin reference                  trace_dentin()
        |   Median intensity of each cross-section's deep core (>=6 voxels from
        |   the surface). Enamel is an outer cap, so the core is dentin wherever
        |   dentin exists. Dentin drifts along the tooth -- up to -20 units on
        |   Odam-CRE-B/-C -- and role assignment [5] is a nearest-neighbour test
        |   with no margin, so a fixed global reference swaps the two roles
        |   wherever the drift exceeds half the component separation.
        v
    [1] Per-slice foreground, rim-eroded            slice_values()
        |   Two iterations of a 3x3 erosion. Without it the modal intensity of
        |   a tapered cervical slice collapses onto partial-volume rim voxels
        |   and every real voxel reads as "excess" above dentin.
        v
    [2] Validity gate                               slice_validity()
        |   A slice with almost no interior cannot support a distinct interior
        |   population. Gate on rim fraction and max interior depth; a failing
        |   slice is reported `unresolvable`, never given a number.
        v
    [3] Bounded 1- vs 2-component Gaussian fit      fit_one() / fit_two()
        |   Both component centres constrained to a physical tissue range, both
        |   widths to within ~3x the dentin width. Seeded from BOTH sides so the
        |   second component may land above or below dentin.
        v
    [4] F-like statistic vs a self-calibrated null  decide()
        |   D = (chi2_1 - chi2_2) / (chi2_2 / dof), compared against a null
        |   bootstrapped from THIS specimen's own dentin voxels.
        v
    [5] Role assignment by continuity               decompose()
        |   The component closer to THIS slice's traced dentin reference is
        |   dentin, the other is enamel. Which SIDE enamel falls on is an
        |   output, never an assumption.
        v
    [6] Is a low-density component real tissue?     spatial_coherence()
        |   Only the low side can be counterfeited by partial-volume rim, and
        |   the test is THICKNESS: rim is a 1-2 voxel blur and cannot survive
        |   two erosions, while a real matrix layer has an interior. Angular
        |   extent is secondary -- secretory matrix legitimately wraps the whole
        |   perimeter in some specimens.
        v
    [7] Where does enamel actually start?           trim_pre_onset()
        |   A profile-level step, because onset is visible only in the shape of
        |   the curve along z. Enamel begins as its least dense form (matrix)
        |   and mineralizes monotonically, so the profile's MINIMUM is the
        |   onset. Cervical to it the cross-section is dentin plus enamel-organ
        |   soft tissue and holds no enamel at all, so BOTH `resolved_*` and
        |   `coincident` are withdrawn there -- `coincident` is not a null
        |   result, it claims enamel is present at the dentin density and
        |   reports a ratio (~1.03) saying so. Those slices become `pre_onset`
        |   with enamel density and ratio set to ZERO -- no enamel is a
        |   quantity, not a missing measurement.
        |   Just TIPWARD of the onset the matrix is thin and patchy, and where
        |   it fails the solidity test the fit falls back to the dentin and
        |   reports `coincident` at the dentin's level: the profile opens high
        |   and drops to the matrix. Until the matrix has risen halfway to
        |   dentin no enamel can be that dense, so those readings are withdrawn
        |   (`unresolvable`) -- withdraw_opening_fallbacks().
        v
    CSV + PNG profiles, 0% = cervical loop -> 100% = tip


Why this shape, and what was measured
-------------------------------------
Several more obvious designs were built and rejected on measurements:

* One-sided template subtraction (anchor the dentin fit on its left flank,
  look for excess on the right) gives excellent above-dentin sensitivity but
  *structurally cannot* see enamel that is less dense than dentin -- such a
  component hides under the very flank used to anchor the fit.

* A dentin template built by mirroring the molar's own uncontaminated left
  flank leaves a +4.2% systematic residual on a dentin-only null, six times a
  plain Gaussian's +0.67%, and produced a 66% false-positive rate.

* Relative chi2 improvement as the test statistic fails: a second Gaussian
  improves chi2 by ~40% even on pure dentin, because it absorbs
  smoothing/discretization mismatch. The null gate lands at 0.81 and nothing
  real ever clears it.

* An unbounded two-component fit lets the second component run off onto the
  histogram tails, reporting enamel centres of 281.8 and 17.9 on a volume whose
  maximum intensity is 248.

* Suppressing the cervical spike with a per-slice rule does not work, and this
  is why [7] is a profile-level step. The spike slices are not anomalous when
  examined one at a time: measured on P52-Hifla-KO-2 at z=1164, the fit is
  clean (D well over gate, separation 4.8 sigma), the roles are assigned
  correctly -- the component at 119 sits at median depth 8.1 and IS the dentin
  core, the one at 84 sits at depth 4.5 and IS the outer layer -- and the
  outer layer survives the rim gate on thickness (solidity 0.10-0.19). Every
  per-slice test passes because nothing is wrong with the slice. What is wrong
  is that the outer layer there is enamel organ, not enamel, and no single
  cross-section can say which it is. Only the profile can: the tissue that
  precedes the density minimum cannot be enamel, because enamel starts at its
  minimum. A mass-fraction ceiling was tried instead and rejected -- the spike
  runs at 0.77-0.81 of the cross-section but genuine matrix reaches 0.43, and
  three of the seven cohort spikes sit at 0.19-0.36, inside the legitimate
  range.

Measured performance of the adopted method (synthetic, n=6000, 20% enamel):
detection at 2.0 sigma separation is 64-100% above dentin and 48-100% below;
at 3.0 sigma it is 100% both ways. False positives on pure dentin stay at or
below 0.7% at every sample size tested. Recovered centres are accurate to
+/-1.1 intensity units (149.6 vs 150 true; 89.8 vs 90 true) -- accuracy matters
more than raw detection rate here, because the recovered centre *is* the
reported density.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter1d,
    label,
)
from scipy.optimize import least_squares
from tqdm import tqdm

# Import the project's loaders so this analysis reads volumes exactly the way
# the segmentation pipeline does -- same (Z, Y, X) axis order, same spacing
# rules. tools/view_napari.py uses this same bridge; segment_enamel/'s flat
# `from CMPR import ...` style only works when run from inside that directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segment_mandible.data_io import load_nrrd, load_tiff  # noqa: E402
from longitudinal_incisor_analysis.cmpr import derive_sample_name  # noqa: E402

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Tunable constants
#
# Each value records the measurement that justifies it. These are rebound from
# argparse in main(), matching the convention in segment_enamel/enamel_seg.py.
# ------------------------------------------------------------------

# Iterations of a 3x3 erosion applied to each slice's foreground before
# histogramming. Without it the per-slice mode collapses onto the partial-volume
# rim in tapered cervical slices -- measured modes of 81/55/51 at z=78/86/90%
# against a true dentin peak above 120 -- so every real voxel reads as "excess"
# above dentin and up to 40% phantom enamel is fabricated. Two iterations
# restore those modes to 115/93/67. Three empties the smallest cross-sections.
EROSION_ITERATIONS = 2

# Histogram bins over the intensity range. 128 bins across 0-248 is ~1.94
# intensity units per bin, roughly 6 bins per dentin sigma: fine enough to
# localize a centre after the fit, coarse enough that bins are not
# Poisson-dominated on a single sparse slice.
HISTOGRAM_BINS = 128

# Gaussian smoothing of the histogram, in bins. 1.5 bins is ~0.25 dentin sigma:
# it suppresses the comb left by integer-valued intensities without merging a
# genuinely nearby second component into the dentin peak. This sets the floor on
# detectable separation, so raising it trades sensitivity for stability.
SMOOTH_SIGMA = 1.5

# Pooled-voxel target for a single slice's histogram. The z-window grows until
# the pooled count reaches this or MAX_Z_WINDOW is hit. Pooling is safe because
# properties vary slowly along z -- the dentin mode drifts ~26 intensity units
# over ~770 slices, well under one unit per slice.
TARGET_VOXELS = 6000
MAX_Z_WINDOW = 12

# Eroded-foreground voxels below which a slice is reported `insufficient` with
# no value rather than given a fabricated one.
MIN_SLICE_VOXELS = 250

# Validity gate. A slice whose foreground is mostly within RIM_DEPTH_VOXELS of
# the surface has no interior to support a distinct interior population, so the
# honest answer is `unresolvable`. Rim fraction climbs monotonically toward the
# cervical loop as the cross-section tapers.
#
# 0.25 was set from the mineralized range alone and cut straight through the
# secretory zone, where the tooth is genuinely thinner but the tissue is still
# fully resolvable: measured on Odam-CRE-A the secretory cap survives to
# rim=0.47 / depth=10.0 (raw z=1130) with a single coherent component at
# angular concentration 0.52. The component only degenerates into a perimeter
# ring below depth ~8, which MIN_INTERIOR_DEPTH already catches. 0.50 therefore
# admits the secretory stage while still retiring the extreme taper, where
# rim reaches 0.56-0.94 and the cross-section is nothing but rim.
#
# This gate decides only whether a slice CAN be measured. Whether a low-density
# component is real matrix or partial-volume rim is settled separately, by
# MIN_LOW_SIDE_CONCENTRATION.
RIM_DEPTH_VOXELS = 2.0
MAX_RIM_FRACTION = 0.50
MIN_INTERIOR_DEPTH = 8.0

# Saturation is a property of the DATA, not of the fit: it means voxels are
# piling up against the scan's intensity ceiling, so the tissue's true density is
# censored. It is declared only when more than SATURATION_VOXEL_FRACTION of a
# slice's voxels sit within SATURATION_MARGIN of the volume maximum.
#
# Testing instead for "fitted centre near the ceiling" (the previous rule)
# censored 61 perfectly good slices on Odam-CRE-A, because a runaway fit lands
# near the ceiling whether or not the data does. That scan is in fact not clipped
# at all: the incisor histogram declines smoothly from 13887 counts at 235 to 14
# at 248, and the enamel peak at 223 closes its upper half-max at 237.
SATURATION_MARGIN = 3.0
SATURATION_VOXEL_FRACTION = 0.02

# Maximum fitted enamel width, as a multiple of the dentin width. A genuine
# enamel component measures 0.36-2.89 dentin widths on the reference scan.
# Beyond 3.0 the component is no longer tracking a population: it has widened to
# absorb the peak's skewed shoulder, which drags its centre off the peak and onto
# the fit bound (measured sigma 62-77 with centres at the bound, on slices whose
# brightest voxel is 235-239). Such a fit is retried under `refit_narrow`.
MAX_ENAMEL_SIGMA_RATIO = 3.0

# Width cap used by the constrained refit, in dentin widths. Deliberately
# tighter than MAX_ENAMEL_SIGMA_RATIO: that value decides whether a fit has run
# away, while this one has to pull the component back onto the peak. At 3.0 the
# refit still rode the shoulder and returned 234 where the visible mode is 229.
REFIT_SIGMA_RATIO = 2.0

# Spatial-coherence requirements applied ONLY to a candidate component that is
# LESS dense than dentin. Partial-volume blur pulls surface voxels DOWN, never
# up, so a low-density component is the one that can be counterfeited by rim.
# These are deliberately NOT applied to dense components, where no such failure
# mode exists, and as a symmetric gate they rejected genuine enamel at z=30%/40%.
#
# DEPTH IS NOT THE DISCRIMINATOR, and using it as one discarded the secretory
# stage wholesale. Enamel matrix is laid down as a thin OUTER layer over dentin,
# so it is legitimately shallow -- measured median depth 3.0-4.1 voxels through
# the secretory zone, against a 5.0 cutoff that rejected every slice of it.
#
# What actually separates matrix from rim is ANGULAR EXTENT. Enamel is a labial
# cap occupying one sector; partial-volume rim wraps the whole perimeter. Mean
# resultant length measured on Odam-CRE-A:
#   known-good mineralized enamel (z%=34-75)   1 component,  0.85-0.89
#   secretory matrix   (z%=15-24, raw z 1005-1130)  1-2 comps, 0.52-0.84
#   true cervical rim  (z%<13,   raw z >1150)   2-11 comps, 0.22-0.41
# 0.45 sits in the gap. The fragment count stays as a second, looser check.
MAX_LOW_SIDE_COMPONENTS = 12
MIN_LOW_SIDE_CONCENTRATION = 0.45

# Fraction of a low-density component that must survive two 3x3 erosions for the
# component to be called a real tissue layer rather than partial-volume rim.
#
# ANGULAR EXTENT IS NOT SUFFICIENT ON ITS OWN, and on its own it discards the
# secretory stage of any specimen whose enamel organ wraps further round the
# tooth than the reference scan's. Odam-CRE-B and -C lay down matrix around most
# of the perimeter at z%=10-30: measured angular concentration 0.02-0.15 on
# slices where the low component is 27-78% of the cross-section and is plainly a
# thick layer, not a fringe. MIN_LOW_SIDE_CONCENTRATION rejected all of it.
#
# Thickness is what actually separates the two, because partial-volume rim is
# not a layer at all -- it is the one-to-two-voxel blur across the tissue
# boundary, so it cannot have an interior. Two erosions strip 2 voxels from every
# edge, which annihilates rim and leaves the core of anything genuinely thicker.
# Measured fraction surviving, pooled over all three scans:
#
#   z% 0-10  (true cervical rim)   0.00-0.01
#   z% 10-15 (taper, mixed)        0.01-0.07
#   z% 15-20 (secretory matrix)    0.07-0.30
#   z% 20-30 (secretory matrix)    0.10-0.39
#
# 0.10 sits in the gap. Note this is a measure of the component's own geometry,
# not of where it sits in the tooth, so it stays valid wherever the secretory
# zone happens to fall in a given specimen.
MIN_LOW_SIDE_SOLIDITY = 0.10
LOW_SIDE_EROSIONS = 2

# Per-slice dentin reference (see `trace_dentin`).
#
# Role assignment asks which fitted component is dentin. Judging that against
# the specimen's GLOBAL dentin mode assumes dentin holds one density along the
# whole tooth. It does not, and the size of the violation is specimen-specific:
# measured drift of the per-slice mode away from the global anchor, cervical
# half, is -15 units at worst on Odam-CRE-A but reaches -64 on Odam-CRE-B and
# -73 on Odam-CRE-C. Those two scans are simply less mineralized cervically.
#
# On a global reference that is fatal rather than merely noisy, because the
# comparison is a nearest-neighbour test with no margin. At Odam-CRE-C z=880 the
# fit returns components at 116.4 (82% of the mass, one connected region
# covering 74% of the cross-section -- the tooth body) and 142.1 (a thin bright
# band). Against a global anchor of 129.5 the distances are 13.1 and 12.6, so
# the body loses by 0.5 units, is relabelled "enamel", and is then thrown out by
# the low-side rim gate as a 1-component ring of concentration 0.07. The slice
# reports `coincident`. That single inversion accounts for the 38 coincident
# slices on each of B and C against 4 on A.
#
# The fix is to track dentin along z instead of assuming it. The trace starts in
# the anchor band, where dentin is unambiguous, and walks outward one step at a
# time; each slice is judged against its own neighbours, so a gradual drift is
# followed while a jump is not.
#
# Half-width, in slices, of the median filter applied to the raw traced modes.
# 12 spans ~0.1 mm at 8 um spacing -- long against the per-slice noise, short
# against the real drift, which develops over hundreds of slices.
DENTIN_TRACE_SMOOTH = 12

# z-stride between trace points. The reference is interpolated between them.
DENTIN_TRACE_STRIDE = 8

# Depth, in voxels from the tissue surface, at or beyond which a voxel is taken
# to be dentin for the purpose of the reference trace.
#
# This is an ANATOMICAL definition, and it is what makes the trace robust.
# Enamel is an outer cap over a dentin core, so the deepest tissue in a
# cross-section is dentin wherever dentin exists at all. Measured median
# intensity by depth shell, all three reference scans, rises monotonically
# outward-to-inward and flattens from depth 6:
#
#   z%=50   shell [2-4) [4-6) [6-8) [8-12)      z%=15   [2-4) [4-6) [6-8) [8-12)
#     A     132.3 131.4 130.5 131.0              A      112.7 117.7 121.3 108.6
#     B     111.9 124.9 127.1 124.9              B       71.7  97.2 111.4 117.8
#     C     118.4 132.4 134.5 132.3              C       80.8  98.9 114.4 113.7
#
# The plateau beyond 6 is dentin; the rise below it is partial-volume rim and,
# cervically, unmineralized enamel matrix. Note B and C at z%=15: the outer
# shells read 72-99 while the core holds 111-118. A histogram-peak trace has to
# guess which of those populations is dentin and gets it wrong specimen by
# specimen; a depth rule simply reads the core.
DENTIN_CORE_DEPTH = 6.0

# Minimum core voxels for the depth-based estimate to be used for a trace point.
# Below this the cross-section has no core left to measure and the point is
# skipped, the trace interpolating across it.
MIN_CORE_VOXELS = 30

# Ceiling on a trace point, in dentin sigmas above the global anchor.
#
# The core is dentin only while dentin exists. At the incisal tip the dentin
# horn tapers out and the deepest remaining tissue is enamel, so the core
# reading climbs onto the enamel peak and the reference follows it -- which
# re-creates the very role inversion the trace exists to prevent, this time at
# the tip (measured "dentin=217, enamel=157" at z%=95 on Odam-CRE-B).
#
# Real dentin drift is slow and bounded; enamel contamination is not. Measured
# core deviation from the anchor, in sigmas, over the tip end of all three scans:
#
#   z%   70   80   85   88   90   92   94   95   96   97   98
#    A  +0.2 +0.4 +0.6 +0.8 +1.2 +1.6 +2.1 +2.8 +5.4 +7.8 +8.8
#    B  +0.7 +0.7 +0.9 +1.2 +1.7 +2.5 +3.7 +6.4 +8.5 +10.0 +11.6
#    C  +0.7 +0.8 +0.9 +1.3 +1.6 +2.1 +3.9 +6.4 +8.0 +8.4 +8.9
#
# Dentin holds below +2.5 and the jump to enamel clears +3.7 in one step, so 3.0
# separates them on every scan. Points above it are dropped and the trace holds
# its last dentin value, which is the honest reading: beyond that slice there is
# no dentin left to measure.
MAX_CORE_DEVIATION_SIGMA = 3.0

# Fit bounds, in dentin sigmas either side of the specimen's global dentin mode.
# Asymmetric because enamel legitimately reaches ~+6 sigma at the mature tip
# (measured ratio 1.77) while nothing real sits far below dentin. Bounds are
# what stop the second component chasing histogram tails.
FIT_BOUND_LOW_SIGMA = 6.0
FIT_BOUND_HIGH_SIGMA = 9.0
FIT_SIGMA_RATIO = 3.0

# Minimum separation (in dentin sigmas) and minimum mass fraction for the second
# component to count as a distinct population rather than a reshaping of the
# dentin peak.
MIN_SEPARATION_SIGMA = 0.75
MIN_COMPONENT_FRACTION = 0.02

# Safety factor on the bootstrapped null threshold, and the number of bootstrap
# draws used to build it. 1.15 rather than a larger margin because the null is
# derived from this specimen's own dentin and so already carries its real
# histogram shape; a synthetic-Gaussian null needed 1.3 and still clipped true
# positives at z=35-50%.
GATE_SCALE = 1.15
NULL_DRAWS = 120

# Fraction of the z-range at each end examined for orientation detection, and
# the offset above the dentin mode (in dentin sigmas) defining a "dense" voxel.
# 3.0 sigma lands near intensity 180 on the reference scan, where the measured
# contrast between the two ends is 0.48 against exactly 0.000.
ORIENT_END_FRACTION = 0.15
ORIENT_DENSE_SIGMA = 3.0

# Mid-z band used to derive the specimen's global dentin anchor and the
# dentin-only bootstrap pool, where the cross-section is widest.
ANCHOR_BAND = (0.35, 0.65)

# Molar-anchored dentin baseline (see `molar_baseline`).
#
# The per-slice trace (`trace_dentin`) reads dentin off the deep core of each
# cross-section. That is circular near the incisal tip: the dentin horn tapers
# out, the deepest remaining tissue is enamel, and the reference climbs onto the
# enamel peak. MAX_CORE_DEVIATION_SIGMA was meant to catch that, but it is a
# step-detector and the tip contamination on P82 is a gradual ramp that creeps
# in underneath it. Measured on all three P82 controls, tip end:
#
#   z%        80    84    88    90    92    94    96
#   dentin   128.7 136.6 142.3 149.0 150.2 150.2 150.2   (CTR-1; +2.3 sigma)
#   sigma     13.5  14.0  14.3  15.9  17.7  22.4  19.3
#   e_frac    0.24  0.31  0.37  0.44  0.57  0.73  0.94
#   sep       6.84  5.96  5.52  4.82  4.31  2.60  2.71
#
# The reference climbs 22 units while its own width inflates 65% and the two
# components close from 6.8 to 2.6 sigma apart: that is one population being
# read as two, not dentin mineralizing. The ceiling (158.2 on CTR-1) rejected
# only 11 of 177 sampled points; the trace topped out at 150.2, just under it.
#
# The molar is the honest reference. It is fully mature, geometrically thick,
# and -- the point -- not part of the structure whose composition is being
# measured, so it cannot be contaminated by the enamel we are trying to resolve.
# Measured molar dentin against the mid-shaft incisor anchor, same scans:
#
#   scan        molar dentin   incisor anchor   delta
#   P82-CTR-1      119.9           124.5        +4.6
#   P82-CTR-2      119.6           122.9        +3.3
#   P82-CTR-3      118.2           121.7        +3.5
#
# They agree to within half a dentin sigma, which is what licenses using one for
# the other. Dentin does not meaningfully change along the tooth, so a single
# baseline held flat is a better model than a trace free to wander.
BASELINE_SOURCE_DEFAULT = "molar"

# Maximum tolerated disagreement, in dentin sigmas, between the molar baseline
# and the mid-shaft incisor anchor. Beyond this the two tissues are not on a
# common intensity scale -- differing beam hardening between the molar and
# incisor regions, or a mis-segmented molar -- and the baseline is not
# transferable. Measured deltas above are 0.30-0.42 sigma; 1.5 leaves room for
# ordinary between-region variation while still catching a real calibration
# split, which would show up as several sigma.
MAX_BASELINE_ANCHOR_SIGMA = 1.5

# Onset trimming: discard resolved slices cervical to the start of enamel.
#
# Amelogenesis has a fixed order. The first enamel a slice can contain is
# secretory MATRIX, which is protein-rich and barely mineralized -- it is the
# least dense enamel anywhere on the tooth, and always less dense than the
# dentin beside it. Density then rises monotonically toward the tip as the
# matrix is replaced by mineral. So the density profile's global MINIMUM is the
# onset of enamel, and nothing cervical to it can be enamel at all.
#
# Cervical to that minimum the cross-section is the enamel organ and dental
# follicle -- soft tissue at roughly the same attenuation as early matrix, with
# no mineral in it. The two-component fit has no way to tell them apart from the
# histogram alone: both read as "a less-dense population outside the dentin
# core". Measured on P52-Hifla-KO-2 at z=1164, the low component sits at median
# depth 4.5 voxels wrapping the whole perimeter (angular concentration 0.02) and
# carries 80% of the cross-section, versus 26% for the true matrix layer 14
# slices tipward. It is reported as enamel at 83.7 -- ABOVE the true matrix
# trough of 65.6 -- so the profile opens on a spike, falls into the real trough,
# and only then begins its true rise. The spike is the artifact; the trough is
# the measurement.
#
# Trimming to the global minimum is what the developmental order licenses, and
# it is self-calibrating: no fixed z% cutoff can work, because the onset moves
# between specimens (measured 10.6-17.7% across the P52/P82 cohort).
TRIM_BEFORE_ONSET = True

# A trough is only a credible enamel onset if it is an actual minimum rather
# than the first point of a profile that rises throughout. Require the run of
# slices cervical to it to be long enough to be a real feature, and require the
# density to climb by a real margin afterwards, so a profile whose minimum is
# its first slice (nothing to trim) is left alone.
#
# Measured slices-before-trough across the cohort: 2, 4, 6, 9, 13, 49, 52.
# Measured spike height above trough: 1.3-20.9 intensity units. A 2-unit floor
# clears the +/-1.1 unit accuracy of a recovered centre.
MIN_ONSET_RISE = 2.0

# The trough fixes where enamel starts, but not what the fit reports just
# tipward of it. There the secretory matrix is a thin, patchy layer, and on a
# slice where its fragments fall short of the solidity test the fit rejects
# them as rim and falls back to the dentin alone, reporting `coincident`:
# enamel at the dentin's density. Enamel does not get that dense until it
# approaches the crossover, so the profile opens on a plateau at the dentin's
# level, drops to the matrix, and only then rises -- the shape the trim exists
# to prevent, one step further along. Measured on P22_TGM2_WT_F_2: 123 such
# slices at ratio 0.77-0.83 across z%=15.7-24.5, interleaved with matrix at
# 0.50-0.57; the rejected fragments sit at solidity 0.05-0.10 and the accepted
# ones at 0.10-0.14, one population either side of the cut.
#
# Until the matrix has risen OPENING_RISE of the way from its onset density to
# the dentin's, taken as a median over the next OPENING_WINDOW resolved slices,
# a `coincident` reading is withdrawn. Across the 12 P22 TGM2 specimens the
# withdrawn readings are all at 0.77-0.83 against matrix at 0.47-0.57, and the
# stretch ends at 34.6-41.8%, before the first crossover `coincident` slice in
# every one: tipward of it enamel does approach dentin, and `coincident` is the
# honest report there.
OPENING_RISE = 0.5
OPENING_WINDOW = 21

IS_DEBUG = False


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class DentinAnchor:
    """The specimen's global dentin reference and the fit bounds derived from it.

    Attributes:
        mode: Dentin peak intensity, native units.
        sigma: Dentin peak width, native units.
        low: Lower bound for fitted component centres.
        high: Upper bound for fitted component centres.
        pool_size: Number of dentin-only voxels available for the bootstrap null.
    """

    mode: float
    sigma: float
    low: float
    high: float
    pool_size: int


@dataclass
class SliceResult:
    """Per-slice outcome of the density decomposition.

    `status` is one of:
        resolved_hi   a distinct component ABOVE dentin was found
        resolved_lo   a distinct component BELOW dentin was found
        coincident    the fit is sound but no distinct component separates;
                      enamel and dentin are at the same density
        saturated     a component was found but its centre is pinned at the
                      volume's intensity ceiling, so the value is censored
        insufficient  too few voxels to decide
        unresolvable  the cross-section is mostly partial-volume rim; there is
                      no interior population to measure. Also a slice in the
                      enamel's opening stretch where the matrix was too thin to
                      measure and the fit fell back to the dentin (see
                      OPENING_RISE); its dentin fit is kept
        pre_onset     the slice lies cervical to the start of enamel, so there
                      is no enamel in it to measure: the cross-section is dentin
                      plus enamel-organ soft tissue. Supersedes whatever the
                      per-slice fit concluded, including `coincident`, and the
                      enamel density and ratio are 0 (see TRIM_BEFORE_ONSET)
    """

    z: int
    z_percent: float = float("nan")
    z_mm: float = float("nan")
    n_voxels: int = 0
    n_voxels_raw: int = 0
    z_window: int = 0
    rim_fraction: float = float("nan")
    max_depth: float = float("nan")
    dentin_mode: float = float("nan")
    dentin_sigma: float = float("nan")
    dentin_reference: float = float("nan")
    enamel_density: float = float("nan")
    enamel_sigma: float = float("nan")
    enamel_ratio: float = float("nan")
    enamel_fraction: float = float("nan")
    separation_sigma: float = float("nan")
    d_statistic: float = float("nan")
    gate: float = float("nan")
    n_components: int = 0
    angular_concentration: float = float("nan")
    median_depth: float = float("nan")
    solidity: float = float("nan")
    status: str = "insufficient"
    reason: str = ""


# ------------------------------------------------------------------
# Histogram and bounded mixture fitting
# ------------------------------------------------------------------

def build_histogram(values: np.ndarray,
                    value_range: tuple[float, float],
                    nbins: int = HISTOGRAM_BINS,
                    smooth_sigma: float = SMOOTH_SIGMA
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed intensity histogram.

    Parameters:
        values: 1-D array of foreground intensities.
        value_range: (low, high) intensity range spanned by the bins.
        nbins: Number of bins.
        smooth_sigma: Gaussian smoothing width, in bins.

    Returns:
        (bin_centres, smoothed_counts)
    """
    counts, edges = np.histogram(values, bins=nbins, range=value_range)
    centres = (edges[:-1] + edges[1:]) / 2.0
    smoothed = gaussian_filter1d(counts.astype(np.float64), smooth_sigma)
    return centres, smoothed


def _gaussian(x: np.ndarray, amp: float, mean: float, sigma: float) -> np.ndarray:
    """Unnormalized Gaussian, guarded against a zero width."""
    return amp * np.exp(-((x - mean) / np.maximum(sigma, 1e-9)) ** 2 / 2.0)


def _poisson_residual(model: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Residual weighted by the model's own Poisson noise.

    Counting noise in a histogram bin scales as sqrt(counts), so weighting by
    the model rather than the observation avoids the bias that arises when an
    empty bin claims zero uncertainty.
    """
    return (model - observed) / np.sqrt(np.maximum(model, 1.0))


def fit_one(centres: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
    """Fit a single Gaussian, seeded from the dominant peak.

    Returns:
        Array [amplitude, mean, sigma].
    """
    peak = int(np.argmax(smoothed))
    mean0 = centres[peak]

    # Seed the width from the left half-width at half maximum. The left flank is
    # the side least likely to be contaminated by a denser second population.
    half = smoothed[peak] / 2.0
    idx = peak
    while idx > 0 and smoothed[idx] > half:
        idx -= 1
    sigma0 = max((mean0 - centres[idx]) * 2.0 / 2.3548, centres[1] - centres[0])

    def residual(params: np.ndarray) -> np.ndarray:
        return _poisson_residual(_gaussian(centres, *params), smoothed)

    try:
        params = least_squares(
            residual, [smoothed[peak], mean0, sigma0], method="lm", max_nfev=3000
        ).x
    except Exception:  # pragma: no cover - degenerate histogram
        logger.debug("Single-Gaussian fit failed; falling back to seed values")
        params = np.array([smoothed[peak], mean0, sigma0])

    params[2] = abs(params[2])
    return params


def fit_two(centres: np.ndarray,
            smoothed: np.ndarray,
            single: np.ndarray,
            low: float,
            high: float) -> np.ndarray | None:
    """Fit two Gaussians with both centres bounded to a physical tissue range.

    The bounds are what make this usable. An unbounded fit lets the second
    component slide onto a histogram tail, which on the reference volume
    produced enamel centres of 281.8 and 17.9 for a volume whose maximum
    intensity is 248.

    The second component is seeded on BOTH sides of the dentin peak and at two
    separations; the best-fitting result wins. That symmetry is what allows
    enamel less dense than dentin to be found -- nothing in the search prefers
    the dense side.

    Parameters:
        centres: Histogram bin centres.
        smoothed: Smoothed counts.
        single: Parameters of the one-component fit, used to seed this one.
        low: Lower bound for both component centres.
        high: Upper bound for both component centres.

    Returns:
        Array [amp1, mean1, sigma1, amp2, mean2, sigma2], or None if every
        seed failed to converge.
    """
    sigma1 = max(single[2], 1e-6)
    amp_cap = max(smoothed.max() * 5.0, 1.0)
    lower = [0.0, low, sigma1 / FIT_SIGMA_RATIO, 0.0, low, sigma1 / FIT_SIGMA_RATIO]
    upper = [amp_cap, high, sigma1 * FIT_SIGMA_RATIO,
             amp_cap, high, sigma1 * FIT_SIGMA_RATIO]

    def residual(params: np.ndarray) -> np.ndarray:
        model = _gaussian(centres, *params[:3]) + _gaussian(centres, *params[3:])
        return _poisson_residual(model, smoothed)

    best: tuple[np.ndarray, float] | None = None
    for sign in (+1, -1):
        for offset in (1.5, 2.5):
            seed = [single[0], single[1], sigma1,
                    single[0] * 0.3, single[1] + sign * offset * sigma1, sigma1 * 0.8]
            seed = [float(np.clip(s, lo + 1e-9, hi - 1e-9))
                    for s, lo, hi in zip(seed, lower, upper)]
            try:
                params = least_squares(
                    residual, seed, bounds=(lower, upper),
                    method="trf", max_nfev=8000,
                ).x
            except Exception:  # pragma: no cover - non-convergent seed
                continue
            if params[0] <= 0 or params[3] <= 0:
                continue
            cost = float(np.sum(residual(params) ** 2))
            if best is None or cost < best[1]:
                best = (params, cost)

    return best[0] if best is not None else None


def _polish_refit(centres: np.ndarray,
                  smoothed: np.ndarray,
                  params: np.ndarray,
                  lower: list[float],
                  upper: list[float],
                  slice_max: float,
                  d_sigma: float) -> np.ndarray | None:
    """Re-run a constrained refit with the width free, keeping the centre local.

    The width cap in `refit_narrow` is what finds the peak; leaving it clamped is
    what biases the centre. This releases the width while confining the centre to
    a narrow window around the value already found, so the component can take its
    true width without being free to wander back onto the shoulder.

    Returns:
        Polished parameters, or None if the free-width fit left the peak.
    """
    centre = params[4]
    free_lower = list(lower)
    free_upper = list(upper)
    free_lower[4] = max(lower[4], centre - 1.5 * d_sigma)
    free_upper[4] = min(upper[4], centre + 1.5 * d_sigma)
    free_upper[5] = max(upper[5] * 4.0, params[5] * 2.0)
    if free_lower[4] >= free_upper[4]:
        return None

    def residual(p: np.ndarray) -> np.ndarray:
        model = _gaussian(centres, *p[:3]) + _gaussian(centres, *p[3:])
        return _poisson_residual(model, smoothed)

    seed = [float(np.clip(s, lo + 1e-9, hi - 1e-9))
            for s, lo, hi in zip(params, free_lower, free_upper)]
    try:
        out = least_squares(residual, seed, bounds=(free_lower, free_upper),
                            method="trf", max_nfev=8000).x
    except Exception:  # pragma: no cover - non-convergent polish
        return None
    if out[0] <= 0 or out[3] <= 0 or out[4] >= slice_max - 1e-6:
        return None
    return out


def refit_narrow(values: np.ndarray,
                 anchor: "DentinAnchor",
                 fit: dict,
                 reference_sigma: float,
                 slice_max: float,
                 nbins: int = HISTOGRAM_BINS,
                 smooth_sigma: float = SMOOTH_SIGMA) -> dict | None:
    """Refit a runaway second component with its width and centre constrained.

    A free second Gaussian on a thin distal slice widens until it can absorb the
    enamel peak's skewed upper shoulder, which walks its centre off the peak and
    onto the fit bound. Tying the width to the dentin width removes that freedom,
    and capping the centre at the slice's own maximum keeps it inside the data.

    On Odam-CRE-A this recovers the visible enamel mode (~229) on the slices that
    previously reported 247.7 against a brightest voxel of 235-239.

    Returns:
        A fit dict in the same form as `decompose`, or None if the constrained
        fit could not find a distinct second component.
    """
    value_range = (0.0, max(slice_max, 1.0) + 1.0)
    centres, smoothed = build_histogram(values, value_range, nbins, smooth_sigma)
    if smoothed.max() <= 0:
        return None

    d_mean, d_sigma = fit["dentin"], max(fit["dentin_sigma"], 1e-9)
    # Tighter than MAX_ENAMEL_SIGMA_RATIO on purpose. That ratio is the
    # threshold for calling a fit runaway; the refit has to actually pull the
    # component back onto the peak, and at 3.0 dentin widths it is still broad
    # enough to ride the shoulder (recovered 234 where the visible mode is 229).
    # 2.0 tracks the mode.
    width_cap = REFIT_SIGMA_RATIO * reference_sigma

    # Centre bounded by the data; width bounded by the dentin width.
    lower = [0.0, d_mean - 0.5 * d_sigma, d_sigma / FIT_SIGMA_RATIO,
             0.0, d_mean + MIN_SEPARATION_SIGMA * d_sigma, d_sigma / FIT_SIGMA_RATIO]
    upper = [smoothed.max() * 5.0, d_mean + 0.5 * d_sigma, d_sigma * 1.5,
             smoothed.max() * 5.0, slice_max, width_cap]
    if lower[4] >= upper[4]:
        return None

    def residual(params: np.ndarray) -> np.ndarray:
        model = _gaussian(centres, *params[:3]) + _gaussian(centres, *params[3:])
        return _poisson_residual(model, smoothed)

    # Seed from the dense population's own mode: the tallest histogram point
    # clear of dentin. That is where the component should end up, so starting it
    # there is what rescues the slices a bound-side seed cannot converge on.
    dense_from = d_mean + MIN_SEPARATION_SIGMA * d_sigma
    dense_bins = centres >= dense_from
    empirical = (float(centres[dense_bins][np.argmax(smoothed[dense_bins])])
                 if dense_bins.any() else d_mean + 6.0 * d_sigma)

    best = None
    for centre0 in (empirical, slice_max - 2.0 * d_sigma, d_mean + 6.0 * d_sigma,
                    d_mean + 9.0 * d_sigma):
        seed = [smoothed.max(), d_mean, d_sigma,
                smoothed.max() * 0.25, centre0, min(width_cap, 2.0 * d_sigma)]
        seed = [float(np.clip(s, lo + 1e-9, hi - 1e-9))
                for s, lo, hi in zip(seed, lower, upper)]
        try:
            params = least_squares(residual, seed, bounds=(lower, upper),
                                   method="trf", max_nfev=8000).x
        except Exception:  # pragma: no cover - non-convergent seed
            continue
        if params[0] <= 0 or params[3] <= 0:
            continue
        cost = float(np.sum(residual(params) ** 2))
        if best is None or cost < best[1]:
            best = (params, cost)

    if best is None:
        return None

    # The width cap is a search aid, not a claim about the tissue: it exists to
    # stop the component sliding onto the shoulder. Once it has found the peak,
    # holding the width down biases the centre low -- measured as a 9-11 unit
    # sag on slices whose fitted sigma sat exactly on the cap. So release the
    # width, with the centre now seeded at the peak, and keep the result only if
    # it stays on that peak.
    polished = _polish_refit(centres, smoothed, best[0], lower, upper,
                             slice_max, d_sigma)
    if polished is not None:
        best = (polished, float(np.sum(residual(polished) ** 2)))

    amp1, mean1, sig1, amp2, mean2, sig2 = best[0]
    # Still on the bound after constraining: there is no locatable second peak.
    if mean2 >= slice_max - 1e-6:
        return None
    mass1, mass2 = amp1 * sig1, amp2 * sig2
    if mass1 <= 0 or mass2 <= 0:
        return None

    chi1 = _chi_square(centres, smoothed, fit_one(centres, smoothed))
    chi2 = _chi_square(centres, smoothed, best[0])
    return dict(
        d_statistic=float(chi1 - chi2),
        dentin=float(mean1), dentin_sigma=float(sig1),
        enamel=float(mean2), enamel_sigma=float(sig2),
        fraction=float(mass2 / (mass1 + mass2)),
        separation=float(abs(mean2 - mean1) / max(sig1, 1e-9)),
    )


def _chi_square(centres: np.ndarray, smoothed: np.ndarray, params: np.ndarray) -> float:
    """Poisson-weighted chi-square of a one- or two-component model."""
    model = _gaussian(centres, *params[:3])
    if len(params) > 3:
        model = model + _gaussian(centres, *params[3:])
    return float(np.sum(_poisson_residual(model, smoothed) ** 2))


def decompose(values: np.ndarray,
              low: float,
              high: float,
              nbins: int = HISTOGRAM_BINS,
              smooth_sigma: float = SMOOTH_SIGMA,
              dentin_reference: float | None = None) -> dict | None:
    """Decompose a set of intensities into dentin plus an optional second component.

    Role assignment does NOT use position (which would presuppose the answer)
    and cannot use mass alone. Mass fails at the mature tip, where enamel is the
    *larger* population: on the reference volume that inverted the pair and
    reported dentin=230 with "enamel" at 136, a spurious less-dense call.

    Dentin is instead identified by continuity. It is the one tissue present
    along the entire tooth at a slowly-varying density, so when a specimen-level
    `dentin_reference` is supplied the component closer to it is dentin and the
    other is enamel. Which SIDE enamel falls on remains a pure output of the
    fit. Without a reference (as when first deriving the anchor) the heavier
    component is taken as dentin.

    Returns:
        A dict with keys `d_statistic`, `dentin`, `dentin_sigma`, `enamel`,
        `enamel_sigma`, `fraction`, `separation`, or None if the histogram is
        empty. `enamel` is None when no second component could be fitted.
    """
    value_range = (0.0, max(high, float(np.max(values)) if values.size else 1.0) + 1.0)
    centres, smoothed = build_histogram(values, value_range, nbins, smooth_sigma)
    if smoothed.max() <= 0:
        return None

    single = fit_one(centres, smoothed)
    base = dict(d_statistic=0.0, dentin=float(single[1]), dentin_sigma=float(single[2]),
                enamel=None, enamel_sigma=float(single[2]), fraction=0.0, separation=0.0)

    double = fit_two(centres, smoothed, single, low, high)
    if double is None:
        return base

    chi1 = _chi_square(centres, smoothed, single)
    chi2 = _chi_square(centres, smoothed, double)
    # D is the RAW drop in Poisson chi-square from admitting a second component.
    #
    # It was previously divided by the two-component reduced chi-square, the
    # textbook F-like normalization. That is only valid when the model can
    # actually reach the data, and a sum of Gaussians cannot: real tissue
    # histograms carry skew and shoulders no two-Gaussian fit removes. So the
    # denominator measures model inadequacy, and it is precisely the slices with
    # the strongest enamel that are least Gaussian. Measured on Odam-CRE-A,
    # dentin-only bootstrap draws reach chi2/dof = 0.04-0.5 while real mid-tooth
    # slices sit at 6-28, so dividing rescaled the two populations onto
    # incomparable footings and INVERTED the comparison: a real slice with 80x
    # the raw evidence for two peaks (15709 vs 199) scored LOWER than the null
    # (D=666 vs 931). That is what flattened 898 of 916 slices onto the dentin
    # line, discarding fitted enamel of 218/210/203 at z=35/45/55%.
    #
    # Undivided, both sides are counted in the same units and separate cleanly:
    # null p99 = 263 against real slices of 750-16500.
    d_stat = chi1 - chi2

    amp1, mean1, sig1, amp2, mean2, sig2 = double
    # Gaussian mass is proportional to amplitude * width.
    mass1, mass2 = amp1 * sig1, amp2 * sig2
    if mass1 <= 0 or mass2 <= 0:
        return base

    if dentin_reference is None:
        first_is_dentin = mass1 >= mass2
    else:
        # Continuity with the specimen's dentin, which varies slowly along z.
        first_is_dentin = (abs(mean1 - dentin_reference)
                           <= abs(mean2 - dentin_reference))

    if first_is_dentin:
        d_mean, d_sig, e_mean, e_sig, e_mass = mean1, sig1, mean2, sig2, mass2
    else:
        d_mean, d_sig, e_mean, e_sig, e_mass = mean2, sig2, mean1, sig1, mass1

    return dict(
        d_statistic=float(d_stat),
        dentin=float(d_mean),
        dentin_sigma=float(d_sig),
        enamel=float(e_mean),
        enamel_sigma=float(e_sig),
        fraction=float(e_mass / (mass1 + mass2)),
        separation=float(abs(e_mean - d_mean) / max(d_sig, 1e-9)),
    )


# ------------------------------------------------------------------
# Slice extraction and validity
# ------------------------------------------------------------------

def _erosion_structure() -> np.ndarray:
    """3x3 structuring element used for rim erosion."""
    return np.ones((3, 3), dtype=bool)


def eroded_mask(volume: np.ndarray, z: int,
                iterations: int = EROSION_ITERATIONS) -> np.ndarray:
    """Foreground of slice z with the partial-volume rim stripped.

    `border_value=0` so a cross-section touching the array edge is eroded from
    that edge too, rather than being treated as if it continued.
    """
    mask = volume[z] > 0
    if iterations <= 0:
        return mask
    return binary_erosion(mask, _erosion_structure(),
                          iterations=iterations, border_value=0)


def slice_values(volume: np.ndarray, z: int, window: int = 0,
                 iterations: int = EROSION_ITERATIONS) -> np.ndarray:
    """Rim-eroded foreground intensities for slice z, pooled over +/- window."""
    depth = volume.shape[0]
    lo, hi = max(0, z - window), min(depth, z + window + 1)
    chunks = []
    for zz in range(lo, hi):
        mask = eroded_mask(volume, zz, iterations)
        if mask.any():
            chunks.append(volume[zz][mask])
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def slice_validity(volume: np.ndarray, z: int) -> tuple[bool, float, float]:
    """Does this cross-section have enough interior to support a distinct population?

    A slice that is mostly partial-volume rim has no interior tissue whose
    intensity could form a second peak, so any second component the fit finds
    there is an artifact of the surface blur. Measured rim fraction along the
    reference tooth climbs monotonically -- 0.10 mid-tooth, 0.21 at z=78%,
    0.37 at z=90%, 0.43 at z=95% -- so this cleanly retires the extreme
    cervical end without touching the measurable range.

    Returns:
        (is_valid, rim_fraction, max_interior_depth)
    """
    mask = volume[z] > 0
    if mask.sum() < 50:
        return False, 1.0, 0.0
    distance = distance_transform_edt(mask)
    rim_fraction = float(((distance <= RIM_DEPTH_VOXELS) & mask).sum() / mask.sum())
    max_depth = float(distance.max())
    valid = rim_fraction <= MAX_RIM_FRACTION and max_depth >= MIN_INTERIOR_DEPTH
    return valid, rim_fraction, max_depth


def spatial_coherence(volume: np.ndarray, z: int,
                      centre: float, width: float,
                      iterations: int = EROSION_ITERATIONS
                      ) -> tuple[int, float, float, float]:
    """Shape descriptors for the voxels making up a candidate component.

    Real incisor enamel is a labial cap: a single connected region covering one
    angular sector at depth. On the reference volume a known-good enamel slice
    measures 1 component, angular concentration 0.90 and median depth 6.0, while
    a cervical partial-volume artifact measures 58 components, angular
    concentration 0.05 and median depth 1.0.

    `solidity` is the fraction of the component surviving LOW_SIDE_EROSIONS
    erosions -- a direct measure of whether it is thick enough to have an
    interior. Partial-volume rim is a 1-2 voxel blur and cannot have one; a real
    matrix layer does. This is the measurement that distinguishes a wrap-around
    secretory matrix (real, and common in B and C) from a wrap-around rim
    artifact, which angular extent alone cannot do.

    These are recorded as confidence columns rather than used as a hard gate:
    as a gate they rejected genuine mid-tooth enamel at z=30% and z=40%.

    Returns:
        (n_components, angular_concentration, median_depth, solidity)
    """
    mask = eroded_mask(volume, z, iterations)
    if not mask.any():
        return 0, float("nan"), float("nan"), float("nan")

    plane = volume[z]
    selected = mask & (plane >= centre - 1.5 * width) & (plane <= centre + 1.5 * width)
    if selected.sum() < 20:
        return 0, float("nan"), float("nan"), float("nan")

    distance = distance_transform_edt(mask)
    _, n_components = label(selected)

    rows, cols = np.nonzero(mask)
    cy, cx = rows.mean(), cols.mean()
    sy, sx = np.nonzero(selected)
    angles = np.arctan2(sy - cy, sx - cx)
    # Mean resultant length: 1.0 is a tight angular sector, 0.0 a full ring.
    concentration = float(np.hypot(np.cos(angles).mean(), np.sin(angles).mean()))
    median_depth = float(np.median(distance[selected]))

    eroded = binary_erosion(selected, _erosion_structure(),
                            iterations=LOW_SIDE_EROSIONS, border_value=0)
    solidity = float(eroded.sum() / selected.sum())
    return int(n_components), concentration, median_depth, solidity


# ------------------------------------------------------------------
# Specimen-level calibration
# ------------------------------------------------------------------

def build_anchor(volume: np.ndarray,
                 band: tuple[float, float] = ANCHOR_BAND,
                 iterations: int = EROSION_ITERATIONS) -> tuple[DentinAnchor, np.ndarray]:
    """Derive the specimen's global dentin reference and its dentin-only voxel pool.

    Sampled from the mid-z band, where the cross-section is widest and the
    dentin peak is most reliable. The returned pool is the set of voxels close
    to the dentin mode, used to bootstrap the null; enamel in this band sits far
    away (measured ~220 against a dentin mode of ~130), so a +/- few sigma
    window around the mode is effectively enamel-free.

    Returns:
        (anchor, dentin_only_pool)
    """
    depth = volume.shape[0]
    start, stop = int(band[0] * depth), int(band[1] * depth)
    step = max((stop - start) // 12, 1)
    sampled = [slice_values(volume, z, 0, iterations) for z in range(start, stop, step)]
    sampled = [s for s in sampled if s.size]
    if not sampled:
        raise SystemExit("Could not sample the mid-z band; is the volume empty?")
    pooled = np.concatenate(sampled)

    # A coarse sweep of the whole volume, used only to size the upper fit bound.
    wide = [slice_values(volume, z, 0, iterations)
            for z in range(0, depth, max(depth // 40, 1))]
    wide = [w for w in wide if w.size]
    pooled_all = np.concatenate(wide) if wide else pooled

    vmax = float(volume.max())
    result = decompose(pooled, 20.0, vmax)
    if result is None:
        raise SystemExit("Could not fit a dentin peak to the mid-z band.")

    mode, sigma = result["dentin"], result["dentin_sigma"]
    low = max(mode - FIT_BOUND_LOW_SIGMA * sigma, 20.0)
    # The upper bound must clear the densest enamel the specimen actually
    # contains, not merely a fixed multiple of the dentin width. Mature tip
    # enamel on the reference volume reaches ~1.8x the dentin mode, and a bound
    # derived from sigma alone clamped the fit at 230.4 across a long stretch of
    # tip slices -- a censored value, not a measured one. Take the wider of the
    # sigma-based bound and the volume's own bright tail.
    bright = float(np.percentile(pooled_all[pooled_all > mode], 99.9)) \
        if np.any(pooled_all > mode) else mode
    high = min(max(mode + FIT_BOUND_HIGH_SIGMA * sigma, bright * 1.05), vmax)

    pool = pooled[(pooled > mode - 2.5 * sigma) & (pooled < mode + 2.0 * sigma)]
    del pooled_all
    anchor = DentinAnchor(mode=mode, sigma=sigma, low=low, high=high,
                          pool_size=int(pool.size))
    logger.info("Global dentin anchor: mode=%.1f sigma=%.1f, fit bounds=[%.0f, %.0f]",
                mode, sigma, low, high)
    return anchor, pool


def _core_intensity(volume: np.ndarray, z: int,
                    iterations: int = EROSION_ITERATIONS,
                    core_depth: float = DENTIN_CORE_DEPTH,
                    min_core: int = MIN_CORE_VOXELS) -> tuple[float, int]:
    """Median intensity of the deep core of slice z, and how many voxels it holds.

    The core is every foreground voxel at least `core_depth` from the tissue
    surface. Enamel is an outer cap, so the core is dentin wherever dentin
    exists; this is therefore a direct read of the local dentin density that
    does not depend on histogram shape, on which population is larger, or on a
    fit whose roles are the very thing in question.

    Distance is measured on the UNERODED foreground, so `core_depth` means depth
    below the real tissue surface rather than below the eroded mask's edge.

    Returns:
        (median intensity, core voxel count); (nan, n) when the core is too small.
    """
    surface = volume[z] > 0
    if surface.sum() < 50:
        return float("nan"), 0
    distance = distance_transform_edt(surface)
    core = eroded_mask(volume, z, iterations) & (distance >= core_depth)
    count = int(core.sum())
    if count < min_core:
        return float("nan"), count
    return float(np.median(volume[z][core])), count


def trace_dentin(volume: np.ndarray,
                 anchor: DentinAnchor,
                 iterations: int = EROSION_ITERATIONS,
                 stride: int = DENTIN_TRACE_STRIDE,
                 smooth: int = DENTIN_TRACE_SMOOTH,
                 core_depth: float = DENTIN_CORE_DEPTH,
                 min_core: int = MIN_CORE_VOXELS) -> np.ndarray:
    """Per-slice dentin reference, measured along z rather than assumed constant.

    Returns one reference intensity per slice, for `decompose` to assign roles
    against. The global anchor remains the specimen's calibration -- fit bounds,
    the null, orientation -- and this supplies only the local "where is dentin in
    THIS slice?" that role assignment needs.

    Dentin is read off the deep core of each cross-section (see
    DENTIN_CORE_DEPTH), median-smoothed along z. Slices whose core has shrunk
    below `min_core` -- the far cervical taper and the extreme tip -- are skipped
    and interpolated across, then held flat beyond the last measured point rather
    than extrapolated.

    Why this is needed at all: role assignment previously compared both fitted
    components against the specimen's GLOBAL dentin mode, which assumes dentin
    holds one density along the whole tooth. It does not. Measured drift of the
    deep core away from the global anchor is mild on Odam-CRE-A but reaches -20
    on Odam-CRE-B and -C, and the comparison is a nearest-neighbour test with no
    margin, so once drift passes roughly half the component separation the two
    roles swap. At Odam-CRE-C z=880 the fit returns 116.4 (82% of the mass, one
    region covering 74% of the cross-section -- the tooth body) and 142.1 (a thin
    bright band); against a global anchor of 129.5 the body loses by 0.5 units,
    is relabelled "enamel", and is then discarded by the low-side rim gate. That
    inversion produced 38 coincident slices on each of B and C against 4 on A.

    Why the core and not a histogram peak: a peak-based trace has to decide which
    peak is dentin, and cervically that is genuinely ambiguous -- B keeps a
    prominent core peak at ~129 while the tissue around it falls to a median of
    74-91, and C's core fades until the low population dominates. Following the
    tallest peak jumps to enamel at the tip; following the nearest peak sticks to
    whichever population the trace happened to start on. The core sidesteps the
    question: it is dentin by geometry.
    """
    depth = volume.shape[0]
    sampled = np.arange(0, depth, stride)
    measured = np.array([_core_intensity(volume, int(z), iterations,
                                         core_depth, min_core)[0]
                         for z in sampled])

    # Drop points where the core has run up onto enamel: past the dentin horn
    # the deepest tissue is no longer dentin (see MAX_CORE_DEVIATION_SIGMA).
    ceiling = anchor.mode + MAX_CORE_DEVIATION_SIGMA * anchor.sigma
    enamel_core = np.isfinite(measured) & (measured > ceiling)
    if enamel_core.any():
        logger.info("Dentin trace: %d sampled slices sit above %.0f and are read "
                    "as enamel core, not dentin; the trace holds its last dentin "
                    "value there", int(enamel_core.sum()), ceiling)
    measured = np.where(enamel_core, np.nan, measured)

    usable = np.isfinite(measured)
    if usable.sum() < 3:
        logger.warning("Dentin trace: %d slices have a measurable core; "
                       "falling back to the global anchor mode", int(usable.sum()))
        return np.full(depth, anchor.mode, dtype=float)

    points, values = sampled[usable], measured[usable]

    # Median-smooth along z so per-slice noise is not mistaken for drift.
    half = max(int(smooth / max(stride, 1)), 1)
    values = np.array([
        np.median(values[max(i - half, 0): i + half + 1])
        for i in range(values.size)
    ])

    # np.interp holds the end values flat outside the measured span, which is
    # what we want: beyond the last slice with a core there is no evidence of
    # further drift, and extrapolating a trend into the taper would invent one.
    trace = np.interp(np.arange(depth), points, values)
    logger.info("Dentin trace: %d/%d sampled slices had a core, range %.1f-%.1f "
                "(global anchor %.1f, max drift %+.1f)",
                int(usable.sum()), sampled.size, trace.min(), trace.max(),
                anchor.mode,
                max(trace.max() - anchor.mode, trace.min() - anchor.mode, key=abs))
    return trace


def calibrate_null(pool: np.ndarray,
                   anchor: DentinAnchor,
                   sizes: Sequence[int] = (300, 600, 1500, 3000, 6000, 12000, 17000),
                   width_ratios: Sequence[float] = (1.0, 1.3, 1.6, 2.0),
                   draws: int = NULL_DRAWS,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap the distribution of D under "this slice is dentin only".

    The null is built from the specimen's OWN dentin voxels rather than from an
    idealized Gaussian. That distinction is not cosmetic: real dentin carries
    more histogram shape than a Gaussian, and a two-Gaussian fit cannot absorb
    that shape, so the null must inherit it from the specimen instead of
    assuming it away.

    Since D is now the raw chi-square drop (see `decompose`), the thresholds this
    produces are in raw chi-square units and scale with sample size, which is why
    the gate interpolates over size rather than assuming a single cutoff.

    The null depends on the dentin peak's WIDTH as well as the sample size,
    because a broad peak is intrinsically harder for a second component to
    improve upon. Calibrating on the mid-tooth width alone therefore produced a
    threshold too strict for near-tip slices, where dentin broadens to sigma
    15-17. The null is thus a 2-D surface over (sample size, dentin width), and
    slices are judged against their own local width.

    Returns:
        (sizes, width_ratios, thresholds) where thresholds has shape
        (len(sizes), len(width_ratios)), for interpolation by `gate_for`.
    """
    if pool.size < 100:
        raise SystemExit("Dentin pool too small to calibrate the null.")

    rng = np.random.default_rng(seed)
    centred = pool - anchor.mode
    grid = np.zeros((len(sizes), len(width_ratios)), dtype=float)

    for i, n in enumerate(tqdm(sizes, desc="Calibrating null", unit="size")):
        for j, ratio in enumerate(width_ratios):
            observed = []
            for _ in range(draws):
                sample = rng.choice(centred, size=min(n, pool.size), replace=True)
                # Stretch the specimen's own dentin about its mode to emulate a
                # broader peak, preserving its real (non-Gaussian) shape.
                widened = anchor.mode + sample * ratio
                result = decompose(widened, anchor.low, anchor.high,
                                   dentin_reference=anchor.mode)
                if result is not None:
                    observed.append(result["d_statistic"])
            grid[i, j] = float(np.percentile(observed, 99)) if observed else 0.0

    logger.info("Null calibrated over %d sizes x %d widths (p99 range %.0f-%.0f)",
                len(sizes), len(width_ratios), grid.min(), grid.max())
    return (np.asarray(sizes, dtype=float),
            np.asarray(width_ratios, dtype=float), grid)


def gate_for(n: int, sizes: np.ndarray, width_ratios: np.ndarray,
             thresholds: np.ndarray, dentin_sigma: float, anchor_sigma: float,
             scale: float = GATE_SCALE) -> float:
    """Significance threshold for a slice, given its size and dentin width.

    Bilinear in (log sample size, dentin width ratio). Judging a broad-dentin
    slice against a narrow-dentin null is what suppressed genuine near-tip
    enamel, so the local width is used rather than the specimen average.
    """
    ratio = float(np.clip(dentin_sigma / max(anchor_sigma, 1e-9),
                          width_ratios[0], width_ratios[-1]))
    per_size = np.array([
        np.interp(ratio, width_ratios, thresholds[i, :]) for i in range(len(sizes))
    ])
    clipped = float(np.clip(n, sizes[0], sizes[-1]))
    return float(np.interp(np.log(clipped), np.log(sizes), per_size)) * scale


# ------------------------------------------------------------------
# Orientation
# ------------------------------------------------------------------

def detect_orientation(volume: np.ndarray, anchor: DentinAnchor,
                       end_fraction: float = ORIENT_END_FRACTION,
                       iterations: int = EROSION_ITERATIONS) -> tuple[str, dict]:
    """Decide which end of the z-axis is the incisal tip.

    Nothing in the CMPR output fixes which array end is which -- the reference
    test file has the tip at LOW z, the reverse of the plotting convention --
    so this must be measured, never assumed.

    The tip is the mature end: it carries the highest fraction of dense
    (enamel) voxels and the highest dentin mode. Cross-sectional taper, which
    segment_enamel/enamel_seg.py uses, does NOT discriminate here: both ends
    taper (measured 110 voxels at z=0 and 41 at z=100), so it is recorded for
    the log but not voted on.

    Returns:
        (orientation, evidence) where orientation is "tip-low-z" or "tip-high-z".
    """
    depth = volume.shape[0]
    span = max(int(end_fraction * depth), 1)
    dense_cut = anchor.mode + ORIENT_DENSE_SIGMA * anchor.sigma

    def summarize(indices: Iterable[int]) -> tuple[float, float, float]:
        chunks = [slice_values(volume, z, 0, iterations) for z in indices]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            return 0.0, 0.0, 0.0
        pooled = np.concatenate(chunks)
        dense = float((pooled > dense_cut).mean())
        median = float(np.median(pooled))
        area = float(np.mean([c.size for c in chunks]))
        return dense, median, area

    step = max(span // 10, 1)
    low_dense, low_median, low_area = summarize(range(0, span, step))
    high_dense, high_median, high_area = summarize(range(depth - span, depth, step))

    votes = {
        "dense_fraction": "tip-low-z" if low_dense > high_dense else "tip-high-z",
        "median_intensity": "tip-low-z" if low_median > high_median else "tip-high-z",
    }
    orientation = votes["dense_fraction"]
    confident = votes["dense_fraction"] == votes["median_intensity"]
    if not confident:
        logger.warning(
            "Orientation votes disagree (dense=%s, median=%s); using the dense-voxel "
            "vote, which is by far the stronger signal", *votes.values()
        )

    evidence = dict(
        orientation=orientation, confident=confident, votes=votes,
        dense_fraction_low=low_dense, dense_fraction_high=high_dense,
        median_low=low_median, median_high=high_median,
        area_low=low_area, area_high=high_area, dense_cut=dense_cut,
    )
    logger.info("Orientation: %s (dense fraction low-z=%.3f, high-z=%.3f)",
                orientation, low_dense, high_dense)
    return orientation, evidence


def z_to_percent(z: int, depth: int, orientation: str) -> float:
    """Map a slice index to 0% (cervical loop) .. 100% (tip).

    The single place the orientation flip is applied.
    """
    if depth <= 1:
        return 0.0
    if orientation == "tip-low-z":
        return 100.0 * (depth - 1 - z) / (depth - 1)
    return 100.0 * z / (depth - 1)


# ------------------------------------------------------------------
# Per-slice analysis
# ------------------------------------------------------------------

def analyze_slice(volume: np.ndarray, z: int, anchor: DentinAnchor,
                  sizes: np.ndarray, width_ratios: np.ndarray,
                  thresholds: np.ndarray,
                  iterations: int = EROSION_ITERATIONS,
                  target_voxels: int = TARGET_VOXELS,
                  max_window: int = MAX_Z_WINDOW,
                  min_voxels: int = MIN_SLICE_VOXELS,
                  gate_scale: float = GATE_SCALE,
                  volume_max: float | None = None,
                  dentin_trace: np.ndarray | None = None) -> SliceResult:
    """Decompose one slice into dentin plus an optional distinct enamel population.

    `enamel_ratio` is enamel over the dentin REFERENCE, not over this slice's
    own fitted dentin component. The two differ exactly where it matters. As the
    dentin horn tapers out near the tip the fit has less and less dentin to work
    with, so the fitted component drifts upward into the enamel peak -- measured
    137.7, 142.3, 149.3 at z = 89, 91, 94% on P82-Odam-CTR-1 while enamel itself
    barely moves (217.1, 218.5, 228.5). Dividing by that moving denominator
    reports a ratio that FALLS toward the tip (1.577, 1.535, 1.530) purely
    because the denominator is climbing, which reads as the tip being less
    mineralized when it is the most mineralized part of the tooth.

    Dividing by the fixed reference instead makes the denominator mean the same
    thing on every slice, which is what makes ratios comparable along z and
    between specimens -- the entire point of the measurement. It also keeps the
    degenerate tip slices usable: the enamel numerator is still a real read of
    real tissue there even once the fit can no longer separate a dentin
    population to divide by.
    """
    if volume_max is None:
        volume_max = float(volume.max())
    # Role assignment is judged against the LOCAL dentin density, not the
    # specimen-wide mode: dentin drifts along the tooth, and a global reference
    # inverts the two roles wherever it has drifted far enough (see trace_dentin).
    reference = (float(dentin_trace[z]) if dentin_trace is not None
                 else anchor.mode)
    result = SliceResult(z=z)
    result.n_voxels_raw = int((volume[z] > 0).sum())

    valid, rim_fraction, max_depth = slice_validity(volume, z)
    result.rim_fraction = rim_fraction
    result.max_depth = max_depth
    if not valid:
        result.status = "unresolvable"
        result.reason = (f"cross-section is mostly rim "
                         f"(rim={rim_fraction:.2f}, depth={max_depth:.0f})")
        return result

    # Grow the pooling window until the histogram has enough voxels to be
    # informative. Recorded so a wide window is visible as lost z-resolution.
    window = 0
    values = slice_values(volume, z, 0, iterations)
    while values.size < target_voxels and window < max_window:
        window += 1
        values = slice_values(volume, z, window, iterations)
    result.z_window = window
    result.n_voxels = int(values.size)

    if values.size < min_voxels:
        result.status = "insufficient"
        result.reason = f"only {values.size} voxels after erosion"
        return result

    fit = decompose(values, anchor.low, anchor.high,
                    dentin_reference=reference)
    if fit is None:
        result.status = "insufficient"
        result.reason = "empty histogram"
        return result
    result.dentin_reference = reference

    result.dentin_mode = fit["dentin"]
    result.dentin_sigma = fit["dentin_sigma"]
    result.d_statistic = fit["d_statistic"]
    result.gate = gate_for(values.size, sizes, width_ratios, thresholds,
                           fit["dentin_sigma"], anchor.sigma, gate_scale)

    # Genuine saturation: the tissue itself is clipped at the scan ceiling, so
    # every density in this slice is a lower bound. Requires an actual pile-up of
    # voxels at the ceiling, not merely a fitted centre near it.
    #
    # Tested BEFORE the significance and shape gates because saturation is a
    # property of the DATA, and a clipped histogram cannot support the gates in
    # the first place: clipping collapses the bright population into a single
    # spike, the two-component fit puts both components on it (separation 0.00),
    # and D goes sharply NEGATIVE -- measured -6037 and -7956 on Odam-CRE-B's two
    # tip-most slices, where 26% of voxels sit at 254.0. Gating first reported
    # those as `coincident`, i.e. "enamel is the same density as dentin", which
    # is not what a censored measurement means. Odam-CRE-A is unclipped and
    # reaches this branch nowhere, so the ordering only affects genuinely
    # clipped scans.
    if values.size:
        at_ceiling = float((values >= volume_max - SATURATION_MARGIN).sum()
                           / values.size)
        if at_ceiling > SATURATION_VOXEL_FRACTION:
            result.status = "saturated"
            if fit["enamel"] is not None:
                result.enamel_density = fit["enamel"]
                result.enamel_sigma = fit["enamel_sigma"]
                result.enamel_ratio = (fit["enamel"] / reference
                                       if reference else float("nan"))
                result.enamel_fraction = fit["fraction"]
                result.separation_sigma = fit["separation"]
            result.reason = (f"{100 * at_ceiling:.1f}% of voxels sit at the scan "
                             f"ceiling ({volume_max:.0f}); value is a lower bound")
            return result

    if fit["enamel"] is None:
        result.status = "coincident"
        result.enamel_density = fit["dentin"]
        result.enamel_ratio = (fit["dentin"] / reference
                               if reference else float("nan"))
        result.reason = "no second component could be fitted"
        return result

    result.enamel_fraction = fit["fraction"]
    result.separation_sigma = fit["separation"]

    if fit["d_statistic"] < result.gate:
        reason = (f"D={fit['d_statistic']:.0f} below gate {result.gate:.0f}")
    elif fit["separation"] < MIN_SEPARATION_SIGMA:
        reason = f"separation {fit['separation']:.2f} sigma below minimum"
    elif fit["fraction"] < MIN_COMPONENT_FRACTION:
        reason = f"component mass fraction {fit['fraction']:.3f} below minimum"
    else:
        reason = ""

    if reason:
        # Requirement: no separable peak means enamel is at the same density as
        # dentin, not that enamel is absent.
        result.status = "coincident"
        result.enamel_density = fit["dentin"]
        result.enamel_ratio = (fit["dentin"] / reference
                               if reference else float("nan"))
        result.reason = reason
        return result

    # A fitted centre above everything the slice contains is a runaway
    # component, not a measurement -- and NOT the same thing as saturation.
    #
    # These were previously conflated, and the conflation censored 61 good
    # slices. Saturation is a property of the DATA: voxels pile up against the
    # intensity ceiling. Measured on Odam-CRE-A, that pile-up does not exist --
    # the incisor histogram falls smoothly from 13887 counts at 235 to 14 at
    # 248, and the enamel peak (223) closes its upper half-max at 237, eleven
    # units clear of the ceiling. The scan is not clipped anywhere.
    #
    # What actually happened is a fit pathology: on thin distal slices the
    # second Gaussian widens (sigma 62-77 against a dentin sigma of ~12) and
    # slides onto the upper FIT BOUND, absorbing the enamel peak's skewed
    # shoulder instead of locating its centre. It reported 247.7 on slices whose
    # brightest voxel is 235-239 and whose real enamel mode is plainly at ~229.
    #
    # So the test is against the slice's own contents, and the remedy is to
    # refit the component with its width tied to the dentin width, which is what
    # keeps it on the peak rather than on the shoulder.
    slice_max = float(values.max()) if values.size else 0.0
    reference_sigma = min(max(fit["dentin_sigma"], 1e-9), max(anchor.sigma, 1e-9))
    runaway = (fit["enamel"] > slice_max
               or fit["enamel_sigma"] > MAX_ENAMEL_SIGMA_RATIO * reference_sigma)
    if runaway:
        refit = refit_narrow(values, anchor, fit, reference_sigma, slice_max)
        if refit is not None:
            fit = refit
            result.dentin_mode = fit["dentin"]
            result.dentin_sigma = fit["dentin_sigma"]
            result.enamel_fraction = fit["fraction"]
            result.separation_sigma = fit["separation"]
        else:
            result.status = "unresolvable"
            result.reason = (f"enamel component ran away to {fit['enamel']:.0f} "
                             f"(slice max {slice_max:.0f}, width "
                             f"{fit['enamel_sigma']:.0f}) and could not be refitted")
            return result

    n_comp, concentration, median_depth, solidity = spatial_coherence(
        volume, z, fit["enamel"], fit["enamel_sigma"], iterations
    )

    result.n_components = n_comp
    result.angular_concentration = concentration
    result.median_depth = median_depth
    result.solidity = solidity

    is_low_side = fit["enamel"] < fit["dentin"]
    if is_low_side and n_comp:
        # Only the low side can be counterfeited by partial-volume rim, so only
        # the low side has to be shown to be real tissue.
        #
        # THICKNESS IS THE TEST, not shape. Rim is the 1-2 voxel blur across the
        # tissue boundary, so it has no interior and no erosion of it survives;
        # a real layer does. Shape is checked only for components too thin to
        # settle on thickness alone, because a thin component that also wraps the
        # perimeter or shatters into fragments is rim on every available measure.
        thick = np.isfinite(solidity) and solidity >= MIN_LOW_SIDE_SOLIDITY
        if not thick:
            if n_comp > MAX_LOW_SIDE_COMPONENTS:
                result.status = "coincident"
                result.enamel_density = fit["dentin"]
                result.enamel_ratio = (fit["dentin"] / reference
                                       if reference else float("nan"))
                result.reason = (f"low-density component is {n_comp} fragments "
                                 f"with no interior (solidity {solidity:.2f}); "
                                 f"consistent with partial-volume rim")
                return result
            if (np.isfinite(concentration)
                    and concentration < MIN_LOW_SIDE_CONCENTRATION):
                result.status = "coincident"
                result.enamel_density = fit["dentin"]
                result.enamel_ratio = (fit["dentin"] / reference
                                       if reference else float("nan"))
                result.reason = (f"low-density component wraps the perimeter "
                                 f"(angular concentration {concentration:.2f}) "
                                 f"and has no interior (solidity "
                                 f"{solidity:.2f}); consistent with "
                                 f"partial-volume rim")
                return result

    result.enamel_density = fit["enamel"]
    result.enamel_sigma = fit["enamel_sigma"]
    # Denominator is the specimen's dentin REFERENCE, not this slice's fitted
    # dentin component (see `enamel_ratio` note above analyze_slice).
    result.enamel_ratio = fit["enamel"] / reference if reference else float("nan")
    result.status = "resolved_lo" if is_low_side else "resolved_hi"
    return result


# Statuses that carry a real enamel measurement.
RESOLVED = ("resolved_hi", "resolved_lo")


def find_enamel_onset(results: Sequence[SliceResult],
                      min_rise: float = MIN_ONSET_RISE) -> int | None:
    """Index of the first slice that can contain enamel, or None to trim nothing.

    Enamel begins as secretory matrix, the least mineralized enamel on the
    tooth, and mineralizes monotonically toward the tip. The density profile's
    global minimum is therefore the onset, and any resolved slice cervical to it
    is measuring something that is not enamel (see TRIM_BEFORE_ONSET).

    `results` must already be ordered cervical loop -> tip.

    The trough is located over every slice that carries a density, `coincident`
    as well as `resolved_*`. Searching only the resolved ones leaves the onset
    blind to whatever sits cervical to the first resolved slice, and a specimen
    can carry a long pre-enamel run that never resolves: measured on P52-WT-3,
    127 `coincident` slices at ratio 0.79-0.99 span z%=5.7-15.8, all of them
    rim fragments with no interior, and every one of them was reported as a
    density because the first `resolved_lo` slice did not appear until 15.0%.
    Including them changes the trough's location in none of the eleven cohort
    specimens -- it is the same slice either way -- and changes only which
    slices are seen to precede it.

    Returns:
        The index into `results` of the trough slice, or None when nothing
        carries a density at all.
    """
    valued = [(i, r) for i, r in enumerate(results)
              if r.status in RESOLVED or r.status == "coincident"]
    if not valued:
        return None

    densities = np.array([r.enamel_density for _, r in valued], dtype=float)
    if not np.isfinite(densities).any():
        return None

    trough = int(np.nanargmin(densities))
    return valued[trough][0]


def _sustained_onset(results: Sequence[SliceResult],
                     resolved_before: Sequence[tuple[int, SliceResult]]) -> int:
    """First index of the run of resolved slices that is no longer interrupted.

    Used only when the slices before the trough are a flat matrix plateau rather
    than a spike. Walking back from the trough, a `coincident` slice among them
    is the pre-enamel population still showing through, so the onset sits after
    the last such interruption. Falls back to the first resolved slice when
    there is no interruption at all.
    """
    indices = [i for i, _ in resolved_before]

    # Look a little past the trough, not merely up to it: the interleaving runs
    # on. On P52-WT-3 the trough sits at z%=16.2 while the last interrupting rim
    # slice is at 15.83, so a scan that stopped at the trough would still call
    # 15.60 the onset and spare it. The window is the span of the plateau
    # itself, which keeps the test local -- coincident slices elsewhere on the
    # tooth are a different phenomenon and must not reach back to here.
    span = indices[-1] - indices[0]
    last = min(indices[-1] + span + 1, len(results) - 1)

    for i in indices:
        if not any(r.status == "coincident" for r in results[i:last + 1]):
            return i
    return indices[-1]


def trim_pre_onset(results: Sequence[SliceResult],
                   min_rise: float = MIN_ONSET_RISE) -> int:
    """Re-label every slice cervical to the enamel onset as `pre_onset`.

    Mutates `results` in place. Both `resolved_*` and `coincident` are
    withdrawn, because both assert that enamel is present -- one that it
    separated from dentin, the other that it sits at the same density -- and
    cervical to the onset neither is true. The dentin fit and the shape
    descriptors stay on the row so the trim is auditable in the CSV; the enamel
    density and ratio are set to ZERO, because the region contains no enamel
    and that is a quantity, not a missing measurement.

    Then withdraw_opening_fallbacks() clears the stretch just tipward of the
    onset of the dentin readings the fit fell back to where the matrix was too
    thin to measure.

    Returns:
        How many slices were re-labelled `pre_onset`.
    """
    onset = find_enamel_onset(results, min_rise)
    if onset is None:
        logger.info("Enamel onset: no slice carries a density; nothing trimmed")
        return 0

    onset_percent = results[onset].z_percent
    onset_density = results[onset].enamel_density

    # `min_rise` decides only whether the RESOLVED slices before the trough are
    # a spike worth removing or a flat opening plateau worth keeping. It must
    # not gate the coincident ones. Measured on P52-WT-3, the resolved slices
    # open flat -- they wander 0.64 units, inside the +/-1.1 unit accuracy of a
    # recovered centre, so trimming them would delete real matrix -- but 127
    # `coincident` slices at ratio 0.79-0.99 still sat cervical to the trough.
    # Returning early on the rise test left every one of them on the plot, which
    # is the failure this split repairs: a coincident slice before the onset is
    # pre-enamel on the developmental order alone, whatever the resolved slices
    # around it are doing.
    resolved_before = [(i, r) for i, r in enumerate(results[:onset])
                       if r.status in RESOLVED]
    if resolved_before:
        heights = np.array([r.enamel_density for _, r in resolved_before],
                           dtype=float)
        rise = float(np.nanmax(heights) - onset_density)
        if not (np.isfinite(rise) and rise >= min_rise):
            # The resolved slices before the trough are a flat opening plateau,
            # not a spike, so they are real matrix and the onset belongs among
            # them rather than at the trough.
            #
            # It belongs at the first SUSTAINED one, not simply the first. Where
            # the pre-enamel run overlaps the start of matrix, the fit alternates
            # between the two populations slice by slice: measured on P52-WT-3
            # at z%=15.0-15.9, matrix at 62.5 (ratio 0.52) interleaves with the
            # same rim artifact at 101-103 (ratio 0.84-0.86) that runs
            # continuously from 5.7%. Anchoring on the first matrix slice leaves
            # every interleaved rim slice tipward of the boundary and so spares
            # it. The onset is therefore the first slice after which matrix
            # holds without further interruption.
            onset = _sustained_onset(results, resolved_before)
            onset_percent = results[onset].z_percent
            onset_density = results[onset].enamel_density
            logger.info("Enamel onset moved to z%%=%.1f: the %d resolved "
                        "slice(s) before the trough rise only %.2f (< %.2f), "
                        "so they are a flat opening plateau of real matrix "
                        "rather than a spike", onset_percent,
                        len(resolved_before), rise, min_rise)

    # Everything cervical to the onset is pre-enamel, whatever the per-slice
    # fit concluded. `coincident` is re-labelled along with `resolved_*`
    # because it is not a null result: it asserts that enamel is present at the
    # same density as dentin, and reports a number and a ratio (~1.03) saying
    # so. Before enamel exists that claim is simply false -- the value is the
    # dentin fit, and the tissue is dentin plus enamel organ. `unresolvable`
    # and `insufficient` are left alone: they assert nothing about enamel, so
    # there is nothing to withdraw.
    trimmed = 0
    for res in results[:onset]:
        if res.status not in RESOLVED and res.status != "coincident":
            continue
        res.status = "pre_onset"
        res.reason = (f"cervical to the enamel onset at z%={onset_percent:.1f}; "
                      f"enamel begins as the least dense matrix, so no enamel "
                      f"is present here and the fitted value is dentin")
        # Zero, not blank: there is no enamel here, and "no enamel" is a
        # quantity (none of it) rather than a failure to measure. Blank would
        # put these slices in the same bucket as `unresolvable`, where the
        # cross-section could not be read at all, and would drop them from any
        # downstream mean or plot instead of contributing the zero they should.
        # The fitted dentin components stay on the row for audit.
        res.enamel_density = 0.0
        res.enamel_ratio = 0.0
        res.enamel_sigma = 0.0
        trimmed += 1

    if trimmed:
        logger.info("Enamel onset at z%%=%.1f (density %.1f); %d slice(s) "
                    "cervical to it hold no enamel and were re-labelled "
                    "pre_onset", onset_percent, onset_density, trimmed)

    # The onset fixes where enamel starts; the stretch just tipward of it can
    # still open on the dentin's level (see OPENING_RISE).
    withdraw_opening_fallbacks(results, onset)
    return trimmed


def withdraw_opening_fallbacks(results: Sequence[SliceResult], onset: int,
                               rise: float = OPENING_RISE,
                               window: int = OPENING_WINDOW) -> int:
    """Withdraw the dentin readings the fit fell back to in the enamel's opening stretch.

    Mutates `results` in place; they must be ordered cervical loop -> tip, with
    `onset` the index of the enamel onset.

    A `coincident` slice asserts that enamel is present at the dentin's
    density. From the onset until the matrix has risen `rise` of the way from
    its onset density toward the dentin -- each resolved slice measured against
    its own dentin fit, taken as a median over the next `window` resolved
    slices -- that cannot be true: enamel only gets denser toward the tip, and
    the matrix around such a slice reads far below dentin. What the slice holds
    is matrix too thin to pass the solidity test, rejected as rim before the
    fit fell back to the dentin alone. Its enamel could not be read, so it
    becomes `unresolvable`; the dentin fit stays on the row for audit.

    Tipward of the opening stretch enamel does approach dentin, and a
    `coincident` slice there is left as the fit reported it. A profile whose
    resolved slices carry no dentin fit is left alone: there is nothing to
    measure the matrix's progress against.

    Returns:
        How many slices were withdrawn.
    """
    base = results[onset].enamel_density
    progress = []
    for i in range(onset, len(results)):
        res = results[i]
        span = res.dentin_mode - base
        if (res.status in RESOLVED and np.isfinite(res.enamel_density)
                and np.isfinite(span) and span > 0):
            progress.append((i, (res.enamel_density - base) / span))
    if not progress:
        return 0

    values = np.array([p for _, p in progress])
    end = len(results)
    for k, (i, _) in enumerate(progress):
        if float(np.median(values[k:k + window])) >= rise:
            end = i
            break

    withdrawn = 0
    for res in results[onset:end]:
        if res.status != "coincident":
            continue
        res.status = "unresolvable"
        res.reason = (f"in the enamel's opening stretch (onset at "
                      f"z%={results[onset].z_percent:.1f}), where the matrix is "
                      f"still far below dentin: the fit fell back to the dentin, "
                      f"so the enamel here could not be read")
        # Blank, as for any slice that could not be read; the dentin fit stays.
        res.enamel_density = float("nan")
        res.enamel_ratio = float("nan")
        res.enamel_sigma = float("nan")
        withdrawn += 1

    if withdrawn:
        stop = (f"z%={results[end].z_percent:.1f}" if end < len(results)
                else "the tip")
        logger.info("Opening stretch z%%=%.1f to %s: %d `coincident` slice(s) "
                    "read at the dentin's level while the matrix is still far "
                    "below it were withdrawn as unresolvable",
                    results[onset].z_percent, stop, withdrawn)
    return withdrawn


def analyze_volume(volume: np.ndarray, anchor: DentinAnchor,
                   sizes: np.ndarray, width_ratios: np.ndarray,
                   thresholds: np.ndarray,
                   orientation: str, spacing_z: float | None = None,
                   stride: int = 1, dentin_trace: np.ndarray | None = None,
                   trim_onset: bool = TRIM_BEFORE_ONSET,
                   **kwargs) -> list[SliceResult]:
    """Run the per-slice decomposition over the whole volume.

    The per-slice pass is followed by a profile-level trim: enamel onset can
    only be recognised from the shape of the density profile along z, never from
    a single cross-section, so it is applied here rather than in `analyze_slice`
    (see TRIM_BEFORE_ONSET).
    """
    depth = volume.shape[0]
    volume_max = float(volume.max())
    if dentin_trace is None:
        dentin_trace = trace_dentin(
            volume, anchor, iterations=kwargs.get("iterations", EROSION_ITERATIONS))
    results = []
    for z in tqdm(range(0, depth, stride), desc="Analyzing slices", unit="slice"):
        res = analyze_slice(volume, z, anchor, sizes, width_ratios, thresholds,
                            volume_max=volume_max, dentin_trace=dentin_trace,
                            **kwargs)
        res.z_percent = z_to_percent(z, depth, orientation)
        if spacing_z is not None:
            res.z_mm = z * spacing_z
        results.append(res)

    results.sort(key=lambda r: r.z_percent)

    # Must run on the sorted list: "cervical to the onset" is a statement about
    # position along the tooth, not about acquisition order.
    if trim_onset:
        trim_pre_onset(results)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    logger.info("Slice outcomes: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return results


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

CSV_COLUMNS = [
    "z", "z_percent", "z_mm", "n_voxels", "n_voxels_raw", "z_window",
    "rim_fraction", "max_depth", "dentin_mode", "dentin_sigma", "dentin_reference",
    "enamel_density", "enamel_sigma", "enamel_ratio", "enamel_fraction",
    "separation_sigma", "d_statistic", "gate", "n_components",
    "angular_concentration", "median_depth", "solidity", "status", "reason",
]


def write_csv(results: Sequence[SliceResult], path: str) -> str:
    """Write the per-slice profile, ordered cervical loop -> tip."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for res in results:
            row = asdict(res)
            writer.writerow({
                key: ("" if isinstance(row[key], float) and not np.isfinite(row[key])
                      else row[key])
                for key in CSV_COLUMNS
            })
    logger.info("Wrote %s", path)
    return path


def _shaded_spans(results: Sequence[SliceResult], statuses: Sequence[str]
                  ) -> list[tuple[float, float]]:
    """Contiguous z_percent ranges whose slices all carry one of `statuses`."""
    spans, start, prev = [], None, None
    for res in results:
        if res.status in statuses:
            if start is None:
                start = res.z_percent
            prev = res.z_percent
        elif start is not None:
            spans.append((start, prev if prev is not None else start))
            start, prev = None, None
    if start is not None:
        spans.append((start, prev if prev is not None else start))
    return spans


def make_plot(results: Sequence[SliceResult], path: str, title: str,
              ratio: bool = False) -> str:
    """Plot enamel density against normalized position along the tooth.

    x runs 0% (cervical loop) to 100% (tip). Resolved slices are drawn as a
    connected line; slices where enamel is LESS dense than dentin get their own
    marker so that regime is immediately visible; coincident slices are drawn at
    the dentin level with open markers AND a shaded band, so an inferred
    "same as dentin" never looks like a measured value.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    for lo, hi in _shaded_spans(results, ("unresolvable", "insufficient")):
        ax.axvspan(lo, hi, color="0.85", alpha=0.55, zorder=0)
    # Shaded, not omitted: a trimmed span means "enamel has not started here",
    # which is a finding about the tooth, not missing data.
    for lo, hi in _shaded_spans(results, ("pre_onset",)):
        ax.axvspan(lo, hi, color="tab:red", alpha=0.10, zorder=0)
    for lo, hi in _shaded_spans(results, ("coincident",)):
        ax.axvspan(lo, hi, color="tab:orange", alpha=0.10, zorder=0)
    for lo, hi in _shaded_spans(results, ("saturated",)):
        ax.axvspan(lo, hi, color="tab:purple", alpha=0.10, zorder=0)

    def series(statuses):
        pts = [(r.z_percent,
                r.enamel_ratio if ratio else r.enamel_density)
               for r in results if r.status in statuses]
        pts = [(x, y) for x, y in pts if np.isfinite(y)]
        return [p[0] for p in pts], [p[1] for p in pts]

    if not ratio:
        dx = [r.z_percent for r in results if np.isfinite(r.dentin_mode)]
        dy = [r.dentin_mode for r in results if np.isfinite(r.dentin_mode)]
        ds = [r.dentin_sigma for r in results if np.isfinite(r.dentin_mode)]
        if dx:
            ax.plot(dx, dy, "--", color="0.45", lw=1.2, label="dentin reference", zorder=2)
            ax.fill_between(dx, np.array(dy) - np.array(ds), np.array(dy) + np.array(ds),
                            color="0.6", alpha=0.15, zorder=1)
    else:
        ax.axhline(1.0, ls="--", color="0.45", lw=1.2,
                   label="dentin reference (ratio = 1)", zorder=2)

    hx, hy = series(("resolved_hi",))
    if hx:
        ax.plot(hx, hy, "-o", color="tab:blue", ms=3.4, lw=1.5,
                label="enamel denser than dentin", zorder=4)
    lx, ly = series(("resolved_lo",))
    if lx:
        ax.plot(lx, ly, "-s", color="tab:red", ms=4.0, lw=1.5,
                label="enamel less dense than dentin (secretory matrix)",
                zorder=4)
    sx, sy = series(("saturated",))
    if sx:
        # Caret-up: the true value is at least this high, but the scan's
        # intensity ceiling prevents measuring how much higher.
        ax.plot(sx, sy, "^", mfc="none", mec="tab:purple", ms=5.0, lw=0,
                label="saturated (censored at scan ceiling)", zorder=4)

    cx, cy = series(("coincident",))
    if cx:
        ax.plot(cx, cy, "o", mfc="none", mec="tab:orange", ms=4.5, lw=0,
                label="coincident (enamel = dentin)", zorder=3)

    if any(r.status in ("unresolvable", "insufficient") for r in results):
        ax.plot([], [], "s", color="0.85", ms=8, lw=0, label="not measurable")

    # The pre-onset band carries no markers -- its slices are zero, and drawing
    # them would pull the y-axis down to 0 and flatten the real profile -- so
    # the legend is the only thing that says what the shading means.
    if any(r.status == "pre_onset" for r in results):
        ax.plot([], [], "s", color="tab:red", alpha=0.25, ms=8, lw=0,
                label="before enamel onset (no enamel)")

    ax.set_xlim(0, 100)
    ax.set_xlabel("position along incisor  (%  cervical loop $\\rightarrow$ tip)")
    ax.set_ylabel("enamel / dentin density ratio" if ratio
                  else "enamel density (scanner intensity)")
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="best", fontsize=8, framealpha=0.92)

    ax.annotate("cervical loop", xy=(0, 0), xytext=(2, -0.115),
                textcoords=("data", "axes fraction"), fontsize=8, color="0.35")
    ax.annotate("tip", xy=(100, 0), xytext=(98, -0.115), ha="right",
                textcoords=("data", "axes fraction"), fontsize=8, color="0.35")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def write_run_json(path: str, payload: dict) -> str:
    """Record everything needed to reproduce the run from its own outputs."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Wrote %s", path)
    return path


def log_summary(results: Sequence[SliceResult]) -> None:
    """Report the profile's shape and flag anything that looks unstable."""
    resolved = [r for r in results if r.status in RESOLVED]
    n_saturated = sum(1 for r in results if r.status == "saturated")
    if n_saturated:
        logger.warning("%d slices are saturated at the scan's intensity ceiling; "
                       "their density is a lower bound, not a measurement",
                       n_saturated)
    if not resolved:
        logger.warning("No slice yielded a separable enamel population.")
        return

    ratios = np.array([r.enamel_ratio for r in resolved])
    percents = np.array([r.z_percent for r in resolved])
    logger.info("Resolved %d slices, z=%.0f%%-%.0f%%, ratio %.3f-%.3f",
                len(resolved), percents.min(), percents.max(),
                np.nanmin(ratios), np.nanmax(ratios))

    # Low-contrast warning. Once a specimen stops resolving most of its
    # measurable length, the mean of the slices that DID resolve is no longer a
    # valid density estimate: only favourable fluctuations clear the gate, and
    # their fitted centres sit high. Measured on synthetic data, a specimen whose
    # true ratio is anywhere from 1.09 down to 1.04 reports a mean of ~1.15.
    # The resolvable FRACTION still tracks severity monotonically, so say so here
    # rather than let the mean be quoted.
    # `pre_onset` is excluded by construction: those slices hold no enamel, so
    # counting them would dilute the coincident fraction with pre-enamel tissue
    # and understate a genuinely low-contrast specimen.
    measurable = [r for r in results
                  if r.status in RESOLVED or r.status == "coincident"]
    n_coincident = sum(1 for r in measurable if r.status == "coincident")
    if measurable and n_coincident / len(measurable) > 0.25:
        logger.warning(
            "%.0f%% of measurable slices are `coincident` (%d of %d): enamel is "
            "at or near dentin density over much of this tooth. The mean density "
            "of the resolved slices is biased HIGH in this regime -- report the "
            "resolved/coincident fraction as the severity measure instead. See "
            "README, 'Low-contrast and hypomineralized specimens'.",
            100 * n_coincident / len(measurable), n_coincident, len(measurable))

    low = [r for r in resolved if r.status == "resolved_lo"]
    if low:
        lp = np.array([r.z_percent for r in low])
        ld = np.array([r.enamel_density for r in low])
        logger.info("%d slices show enamel LESS dense than dentin "
                    "(z=%.0f%%-%.0f%%, density %.0f-%.0f): unmineralized "
                    "secretory matrix", len(low), lp.min(), lp.max(),
                    np.nanmin(ld), np.nanmax(ld))

    # Spearman correlation without scipy.stats: rank then Pearson.
    #
    # Computed over the MINERALIZED slices only. The secretory matrix is a
    # different tissue state, not an earlier point on the same mineralization
    # curve, so pooling the two regimes measures the gap between them rather
    # than the maturation trend this is meant to check.
    dense = [r for r in resolved if r.status == "resolved_hi"]
    if len(dense) > 2:
        dp = np.array([r.z_percent for r in dense])
        dr = np.array([r.enamel_ratio for r in dense])
        rx = np.argsort(np.argsort(dp)).astype(float)
        ry = np.argsort(np.argsort(dr)).astype(float)
        if rx.std() > 0 and ry.std() > 0:
            rho = float(np.corrcoef(rx, ry)[0, 1])
            logger.info("Spearman rho(ratio vs position) = %+.3f over %d "
                        "mineralized slices (expect positive: enamel matures "
                        "toward the tip)", rho, len(dense))
            if rho < 0:
                logger.warning("Negative correlation -- check the detected orientation.")

    # An isolated slice disagreeing with both neighbours usually means the fit
    # latched onto a different local optimum there. Checked within each regime:
    # the step across the secretory/mineralized boundary is a real tissue
    # transition, and comparing across it flags that boundary every run.
    for group in (dense, low):
        for i in range(1, len(group) - 1):
            prev, cur, nxt = group[i - 1], group[i], group[i + 1]
            if (abs(cur.enamel_ratio - prev.enamel_ratio) > 0.15
                    and abs(cur.enamel_ratio - nxt.enamel_ratio) > 0.15):
                logger.warning("Slice z=%d (%.1f%%) deviates from both neighbours "
                               "(ratio %.3f vs %.3f / %.3f)", cur.z, cur.z_percent,
                               cur.enamel_ratio, prev.enamel_ratio, nxt.enamel_ratio)


# ------------------------------------------------------------------
# Molar cross-check
# ------------------------------------------------------------------

def molar_cross_check(molar: np.ndarray,
                      iterations: int = EROSION_ITERATIONS) -> dict:
    """Independent validation that the machinery finds a known answer.

    Molar dentin and enamel are well separated, so the same decomposition run
    over the whole molar must recover both peaks. On the reference specimen the
    expected values are a dentin peak at 122.5 and an enamel peak at 188.5; the
    prototype recovered 188.7-189.2. A large deviation means the fit or its
    bounds are wrong and nothing downstream should be trusted.
    """
    mask = molar > 0
    if iterations > 0:
        structure = np.ones((3, 3, 3), dtype=bool)
        mask = binary_erosion(mask, structure, iterations=iterations, border_value=0)
    values = molar[mask].astype(np.float64)
    if values.size < 1000:
        logger.warning("Molar has too few voxels for a cross-check")
        return {}

    vmax = float(molar.max())
    coarse = decompose(values, 20.0, vmax)
    if coarse is None:
        logger.warning("Molar cross-check: could not fit the histogram")
        return {}

    result = dict(dentin=coarse["dentin"], dentin_sigma=coarse["dentin_sigma"],
                  enamel=coarse["enamel"], separation=coarse["separation"],
                  fraction=coarse["fraction"], d_statistic=coarse["d_statistic"],
                  n_voxels=int(values.size))
    if coarse["enamel"] is None:
        logger.warning("Molar cross-check: no enamel component found")
    else:
        logger.info("Molar cross-check: dentin=%.1f (sigma %.1f), enamel=%.1f, "
                    "separation=%.2f sigma", coarse["dentin"], coarse["dentin_sigma"],
                    coarse["enamel"], coarse["separation"])
    return result


def molar_baseline(molar_check: dict, anchor: DentinAnchor,
                   depth: int,
                   max_delta_sigma: float = MAX_BASELINE_ANCHOR_SIGMA
                   ) -> tuple[np.ndarray | None, dict]:
    """A flat, specimen-wide dentin reference taken from the molar.

    Returns one reference intensity per slice -- the same interface as
    `trace_dentin`, so role assignment is unchanged -- but constant along z.
    Dentin does not meaningfully change along the tooth, and holding the
    reference flat denies the tip contamination any way in (see
    BASELINE_SOURCE_DEFAULT).

    The molar value is validated against the mid-shaft incisor anchor before it
    is used. Both are reads of the same tissue on the same scan, so they must
    agree; if they do not, the two regions are not on a common intensity scale
    and transferring one to the other would substitute a calibration error for
    the contamination we are removing. On that failure this returns None and the
    caller falls back to the per-slice trace, which is wrong only at the tip
    rather than wrong everywhere.

    Returns:
        (baseline array or None, diagnostics dict)
    """
    dentin = molar_check.get("dentin") if molar_check else None
    if dentin is None or not np.isfinite(dentin):
        logger.warning("Molar baseline: no usable molar dentin peak; "
                       "falling back to the per-slice trace")
        return None, {"source": "unavailable"}

    dentin = float(dentin)
    sigma = max(anchor.sigma, 1e-9)
    delta = dentin - anchor.mode
    delta_sigma = delta / sigma
    info = {"source": "molar", "value": dentin,
            "molar_sigma": molar_check.get("dentin_sigma"),
            "incisor_anchor": anchor.mode, "delta": delta,
            "delta_sigma": delta_sigma, "max_delta_sigma": max_delta_sigma}

    if abs(delta_sigma) > max_delta_sigma:
        logger.warning(
            "Molar baseline: molar dentin %.1f disagrees with the mid-shaft "
            "incisor anchor %.1f by %+.2f sigma (limit %.2f). The molar and "
            "incisor are not on a common intensity scale on this scan, so the "
            "baseline is not transferable; falling back to the per-slice trace",
            dentin, anchor.mode, delta_sigma, max_delta_sigma)
        info["source"] = "rejected"
        return None, info

    logger.info("Molar baseline: dentin=%.1f held flat along z "
                "(mid-shaft incisor anchor %.1f, agreement %+.2f sigma)",
                dentin, anchor.mode, delta_sigma)
    return np.full(depth, dentin, dtype=float), info


# ------------------------------------------------------------------
# I/O and CLI
# ------------------------------------------------------------------

def load_volume(path: str) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """Load a .tif/.tiff or .nrrd volume as (Z, Y, X), with spacing if recorded."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        volume, spacing = load_tiff(path, return_spacing=True)
    elif ext == ".nrrd":
        volume, spacing = load_nrrd(path, return_spacing=True)
    else:
        raise SystemExit(f"Unsupported input format: {ext!r} (expected .tif or .nrrd)")
    if volume is None:
        raise SystemExit(f"Could not read {path}")
    return volume, spacing


def setup_logging(out_root: str) -> None:
    """Coloured console logging plus a plain-text .log file under out_root."""

    class ColorFormatter(logging.Formatter):
        COLORS = {
            "DEBUG": "\033[94m", "INFO": "\033[92m", "WARNING": "\033[93m",
            "ERROR": "\033[91m", "CRITICAL": "\033[95m",
        }
        RESET = "\033[0m"

        def format(self, record):
            # Colorize a copy so the file handler still sees a plain levelname.
            color = self.COLORS.get(record.levelname, self.RESET)
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
            return super().format(record)

    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter("[%(levelname)s] - %(message)s"))

    os.makedirs(out_root, exist_ok=True)
    log_path = os.path.join(out_root, f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[console, file_handler])
    logging.info("Logging to %s", log_path)


def parse_arguments() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Measure enamel density along the length of a mouse incisor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="CMPR resliced incisor volume (.tif or .nrrd)")
    parser.add_argument("--molar", "-m", default=None,
                        help="Optional molar volume, used only as an independent "
                             "cross-check that the fit recovers a known answer")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: <input dir>/enamel_density)")
    parser.add_argument("--log", "-l", action="store_true", help="Enable logging")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Write per-slice diagnostic figures")
    parser.add_argument("--debug-stride", type=int, default=50,
                        help="z-stride between debug figures")
    parser.add_argument("--orientation", default="auto",
                        choices=("auto", "tip-low-z", "tip-high-z"),
                        help="Which end of z is the incisal tip")
    parser.add_argument("--stride", type=int, default=1,
                        help="Analyze every Nth slice")
    parser.add_argument("--erosion", type=int, default=EROSION_ITERATIONS,
                        help="Rim erosion iterations (0 disables; see README)")
    parser.add_argument("--nbins", type=int, default=HISTOGRAM_BINS,
                        help="Histogram bins")
    parser.add_argument("--smooth-sigma", type=float, default=SMOOTH_SIGMA,
                        help="Histogram smoothing sigma, in bins")
    parser.add_argument("--target-voxels", type=int, default=TARGET_VOXELS,
                        help="Pooled-voxel target before the z-window stops growing")
    parser.add_argument("--max-z-window", type=int, default=MAX_Z_WINDOW,
                        help="Maximum z-pooling half-width")
    parser.add_argument("--min-voxels", type=int, default=MIN_SLICE_VOXELS,
                        help="Voxels below which a slice is 'insufficient'")
    parser.add_argument("--max-rim-fraction", type=float, default=MAX_RIM_FRACTION,
                        help="Rim fraction above which a slice is 'unresolvable'")
    parser.add_argument("--min-depth", type=float, default=MIN_INTERIOR_DEPTH,
                        help="Minimum interior depth (voxels) for a usable slice")
    parser.add_argument("--gate-scale", type=float, default=GATE_SCALE,
                        help="Safety factor on the bootstrapped null threshold")
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS,
                        help="Bootstrap draws per size when calibrating the null")
    parser.add_argument("--baseline", default=BASELINE_SOURCE_DEFAULT,
                        choices=("molar", "trace"),
                        help="Dentin reference: a flat baseline from the molar "
                             "(requires --molar; falls back to the trace if the "
                             "molar disagrees with the incisor anchor), or the "
                             "per-slice core trace")
    parser.add_argument("--max-baseline-delta", type=float,
                        default=MAX_BASELINE_ANCHOR_SIGMA,
                        help="Largest tolerated molar-vs-incisor-anchor "
                             "disagreement, in dentin sigmas")
    parser.add_argument("--no-trim-onset", action="store_true",
                        help="Keep resolved slices cervical to the enamel onset "
                             "(the density trough) instead of labelling them "
                             "'pre_onset', and the dentin readings the fit falls "
                             "back to just tipward of it. Diagnostic only: those "
                             "slices measure enamel-organ soft tissue or dentin, "
                             "not enamel")
    parser.add_argument("--no-plot", action="store_true", help="Skip the figures")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    global EROSION_ITERATIONS, HISTOGRAM_BINS, SMOOTH_SIGMA, TARGET_VOXELS
    global MAX_Z_WINDOW, MIN_SLICE_VOXELS, MAX_RIM_FRACTION, MIN_INTERIOR_DEPTH
    global GATE_SCALE, NULL_DRAWS, IS_DEBUG

    args = parse_arguments()

    out_root = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)), "enamel_density")
    os.makedirs(out_root, exist_ok=True)
    if args.log or args.debug:
        setup_logging(out_root)
    else:
        logging.basicConfig(level=logging.WARNING)

    EROSION_ITERATIONS = args.erosion
    HISTOGRAM_BINS = args.nbins
    SMOOTH_SIGMA = args.smooth_sigma
    TARGET_VOXELS = args.target_voxels
    MAX_Z_WINDOW = args.max_z_window
    MIN_SLICE_VOXELS = args.min_voxels
    MAX_RIM_FRACTION = args.max_rim_fraction
    MIN_INTERIOR_DEPTH = args.min_depth
    GATE_SCALE = args.gate_scale
    NULL_DRAWS = args.null_draws
    IS_DEBUG = args.debug

    volume, spacing = load_volume(args.input)
    logger.info("Loaded %s: shape=%s dtype=%s range=[%g, %g]", args.input,
                volume.shape, volume.dtype, float(volume.min()), float(volume.max()))
    spacing_z = float(spacing[0]) if spacing else None

    anchor, pool = build_anchor(volume, iterations=EROSION_ITERATIONS)
    sizes, width_ratios, thresholds = calibrate_null(pool, anchor, draws=NULL_DRAWS)

    # The molar is read before the reference is chosen: with --baseline molar it
    # supplies that reference, not merely a cross-check.
    molar_check = {}
    if args.molar:
        molar, _ = load_volume(args.molar)
        molar_check = molar_cross_check(molar, iterations=EROSION_ITERATIONS)
        del molar

    baseline_info = {"source": "trace"}
    dentin_trace = None
    if args.baseline == "molar":
        if not args.molar:
            logger.warning("--baseline molar needs --molar; "
                           "falling back to the per-slice trace")
        else:
            dentin_trace, baseline_info = molar_baseline(
                molar_check, anchor, volume.shape[0],
                max_delta_sigma=args.max_baseline_delta)
    if dentin_trace is None:
        dentin_trace = trace_dentin(volume, anchor, iterations=EROSION_ITERATIONS)
        baseline_info.setdefault("source", "trace")
        if args.baseline == "trace":
            baseline_info["source"] = "trace"

    if args.orientation == "auto":
        orientation, evidence = detect_orientation(volume, anchor,
                                                   iterations=EROSION_ITERATIONS)
    else:
        orientation = args.orientation
        evidence = {"orientation": orientation, "source": "user override"}
        logger.info("Orientation forced to %s", orientation)

    results = analyze_volume(
        volume, anchor, sizes, width_ratios, thresholds, orientation, spacing_z,
        stride=args.stride, dentin_trace=dentin_trace, iterations=EROSION_ITERATIONS,
        target_voxels=TARGET_VOXELS, max_window=MAX_Z_WINDOW,
        min_voxels=MIN_SLICE_VOXELS, gate_scale=GATE_SCALE,
        trim_onset=not args.no_trim_onset,
    )
    log_summary(results)

    sample = derive_sample_name(args.input)
    write_csv(results, os.path.join(out_root, f"{sample}_enamel_density_profile.csv"))

    if not args.no_plot:
        make_plot(results, os.path.join(out_root, f"{sample}_enamel_density_profile.png"),
                  f"Enamel density along the incisor — {sample}", ratio=False)
        make_plot(results, os.path.join(out_root, f"{sample}_enamel_density_ratio.png"),
                  f"Enamel / dentin density ratio — {sample}", ratio=True)

    write_run_json(os.path.join(out_root, f"{sample}_enamel_density_run.json"), dict(
        input=os.path.abspath(args.input),
        molar=os.path.abspath(args.molar) if args.molar else None,
        sample=sample, spacing=spacing, orientation=evidence,
        anchor=asdict(anchor), molar_cross_check=molar_check,
        dentin_baseline=baseline_info,
        null=dict(sizes=sizes.tolist(), width_ratios=width_ratios.tolist(),
                  thresholds=thresholds.tolist(), draws=NULL_DRAWS,
                  scale=GATE_SCALE),
        constants=dict(
            EROSION_ITERATIONS=EROSION_ITERATIONS, HISTOGRAM_BINS=HISTOGRAM_BINS,
            SMOOTH_SIGMA=SMOOTH_SIGMA, TARGET_VOXELS=TARGET_VOXELS,
            MAX_Z_WINDOW=MAX_Z_WINDOW, MIN_SLICE_VOXELS=MIN_SLICE_VOXELS,
            MAX_RIM_FRACTION=MAX_RIM_FRACTION, MIN_INTERIOR_DEPTH=MIN_INTERIOR_DEPTH,
            MIN_SEPARATION_SIGMA=MIN_SEPARATION_SIGMA,
            MIN_COMPONENT_FRACTION=MIN_COMPONENT_FRACTION,
            FIT_BOUND_LOW_SIGMA=FIT_BOUND_LOW_SIGMA,
            FIT_BOUND_HIGH_SIGMA=FIT_BOUND_HIGH_SIGMA,
            SATURATION_MARGIN=SATURATION_MARGIN,
            SATURATION_VOXEL_FRACTION=SATURATION_VOXEL_FRACTION,
            MAX_ENAMEL_SIGMA_RATIO=MAX_ENAMEL_SIGMA_RATIO,
            MAX_LOW_SIDE_COMPONENTS=MAX_LOW_SIDE_COMPONENTS,
            MIN_LOW_SIDE_CONCENTRATION=MIN_LOW_SIDE_CONCENTRATION,
            GATE_SCALE=GATE_SCALE,
            TRIM_BEFORE_ONSET=not args.no_trim_onset,
            MIN_ONSET_RISE=MIN_ONSET_RISE,
            OPENING_RISE=OPENING_RISE, OPENING_WINDOW=OPENING_WINDOW,
            BASELINE_SOURCE=args.baseline,
            MAX_BASELINE_ANCHOR_SIGMA=args.max_baseline_delta,
        ),
        status_counts={s: sum(1 for r in results if r.status == s)
                       for s in sorted({r.status for r in results})},
    ))


if __name__ == "__main__":
    main()
