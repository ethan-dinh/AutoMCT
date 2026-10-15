"""
Validation suite for enamel_density.py.

Author: Ethan Dinh
Date:   2026-09-09

These tests encode the measurements that drove the design. The synthetic cases
are the ones that matter most: they are the only place where the true answer is
known, so they are what distinguishes a working detector from one that merely
produces plausible-looking curves.

Run:  python longitudinal_incisor_analysis/tests/test_enamel_density.py [path/to/resliced_volume.tif]
The real-data tests are skipped when the volume is not reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import enamel_density as ed  # noqa: E402


PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    _results.append(ok)
    return ok


def _mixture(rng, n, dentin_mu, dentin_sigma, offset_sigma, fraction,
             enamel_sigma=10.0):
    """Dentin plus an enamel component `offset_sigma` dentin-sigmas away."""
    n_enamel = int(n * fraction)
    return np.concatenate([
        rng.normal(dentin_mu, dentin_sigma, n - n_enamel),
        rng.normal(dentin_mu + offset_sigma * dentin_sigma, enamel_sigma, n_enamel),
    ])


def _classify(values, low=60.0, high=200.0, gate=None):
    """Apply the module's decision rule to a bare array of intensities."""
    fit = ed.decompose(values, low, high)
    if fit is None:
        return "insufficient", None, None
    if gate is None:
        gate = 0.0
    if (fit["enamel"] is None
            or fit["d_statistic"] < gate
            or fit["separation"] < ed.MIN_SEPARATION_SIGMA
            or fit["fraction"] < ed.MIN_COMPONENT_FRACTION):
        return "coincident", fit["dentin"], fit["dentin"]
    status = "resolved_hi" if fit["enamel"] > fit["dentin"] else "resolved_lo"
    return status, fit["enamel"], fit["dentin"]


def _synthetic_gate(size, draws=80, seed=1234):
    """Null threshold for pure dentin at a given sample size."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        fit = ed.decompose(rng.normal(120, 12, size), 60.0, 200.0)
        if fit:
            stats.append(fit["d_statistic"])
    return float(np.percentile(stats, 99)) * ed.GATE_SCALE


# ------------------------------------------------------------------

def test_false_positives(gate):
    """A pure-dentin histogram must not yield an enamel component.

    The single most important test: a detector that invents enamel where there
    is none makes every downstream comparison meaningless.
    """
    print("\nFalse positives on pure dentin (target < 2%)")
    for n in (500, 2000, 6000, 20000):
        rng = np.random.default_rng(9000)
        size_gate = _synthetic_gate(n, draws=60, seed=4321)
        resolved = sum(
            _classify(rng.normal(120, 12, n), gate=size_gate)[0].startswith("resolved")
            for _ in range(120)
        )
        rate = 100.0 * resolved / 120
        check(f"n={n:<6d} false-positive rate {rate:.1f}%", rate < 2.0)


def test_two_sided_detection(gate):
    """Detection must be symmetric: enamel below dentin is as findable as above.

    This is the acceptance test for resolving enamel that is LESS dense than
    dentin -- the regime where a one-sided method is blind by construction.
    """
    print("\nTwo-sided detection at 3.0 sigma separation (target 100%)")
    for label, sign, want in (("above", +1, "resolved_hi"),
                              ("below", -1, "resolved_lo")):
        hits = 0
        for seed in range(20):
            rng = np.random.default_rng(2000 + seed)
            values = _mixture(rng, 6000, 120, 12, sign * 3.0, 0.20)
            hits += _classify(values, gate=gate)[0] == want
        rate = 100.0 * hits / 20
        check(f"enamel {label} dentin: {rate:.0f}% detected", rate >= 90.0)

    print("\nTwo-sided detection at 2.0 sigma separation (target > 40%)")
    for label, sign, want in (("above", +1, "resolved_hi"),
                              ("below", -1, "resolved_lo")):
        hits = 0
        for seed in range(20):
            rng = np.random.default_rng(2000 + seed)
            values = _mixture(rng, 6000, 120, 12, sign * 2.0, 0.20)
            hits += _classify(values, gate=gate)[0] == want
        rate = 100.0 * hits / 20
        check(f"enamel {label} dentin: {rate:.0f}% detected", rate >= 40.0)


def test_recovery_accuracy(gate):
    """The recovered centre is the reported density, so it must be accurate."""
    print("\nRecovered enamel centre (target within +/- 3 intensity units)")
    for sign in (+1, -1):
        truth = 120 + sign * 2.5 * 12
        estimates = []
        for seed in range(15):
            rng = np.random.default_rng(2000 + seed)
            values = _mixture(rng, 6000, 120, 12, sign * 2.5, 0.20)
            status, enamel, _ = _classify(values, gate=gate)
            if status.startswith("resolved"):
                estimates.append(enamel)
        if not estimates:
            check(f"offset {sign * 2.5:+.1f} sigma: recovered", False, "none detected")
            continue
        mean = float(np.mean(estimates))
        check(f"offset {sign * 2.5:+.1f} sigma: true={truth:.0f} est={mean:.1f}",
              abs(mean - truth) <= 3.0, f"n={len(estimates)}")


def test_bounds_respected():
    """A fitted component must never leave the physical tissue range.

    An unbounded fit reported centres of 281.8 and 17.9 on a volume whose
    maximum intensity is 248; the bounds are what prevent that.
    """
    print("\nFit bounds")
    rng = np.random.default_rng(7)
    # A heavy one-sided tail is what tempts the fit off the end of the range.
    values = np.concatenate([rng.normal(120, 12, 5000),
                             rng.exponential(30, 1500) + 120])
    fit = ed.decompose(values, 60.0, 200.0)
    ok = fit is not None and (fit["enamel"] is None or 60.0 <= fit["enamel"] <= 200.0)
    check("fitted centre stays inside [60, 200]", ok,
          f"enamel={fit['enamel'] if fit else None}")


def test_saturation_flagged():
    """A component pinned at the intensity ceiling must be flagged, not reported.

    Mature tip enamel drives voxels into the top of the scan's range; the fit
    then piles a very wide component against that ceiling and returns a centre
    bounded by the data rather than the tissue.
    """
    print("\nSaturation detection")
    rng = np.random.default_rng(11)
    ceiling = 248.0
    # Dentin plus a broad enamel population running into the ceiling. The
    # dentin peak stays clean, as it does on a real scan -- only the enamel is
    # clipped -- so the fitted enamel width blows up against the bound.
    enamel = np.clip(rng.normal(300, 70, 3000), 0, ceiling)
    values = np.concatenate([rng.normal(128, 12, 6000), enamel])

    fit = ed.decompose(values, 60.0, ceiling, dentin_reference=128.0)
    if fit is None or fit["enamel"] is None:
        check("a clipped component trips the saturation rule", False, "no fit")
        return
    reference_sigma = min(fit["dentin_sigma"], 12.0)
    saturated = (fit["enamel"] >= ceiling - ed.SATURATION_MARGIN
                 and fit["enamel_sigma"] > ed.MAX_ENAMEL_SIGMA_RATIO * reference_sigma)
    check("a clipped component trips the saturation rule", saturated,
          f"enamel={fit['enamel']:.1f} sigma={fit['enamel_sigma']:.1f}")


def test_orientation_mapping():
    """The flip must be applied in exactly one place, and be a true mirror."""
    print("\nOrientation mapping")
    depth = 1000
    check("tip-high-z: z=0 is the cervical loop",
          ed.z_to_percent(0, depth, "tip-high-z") == 0.0)
    check("tip-low-z: z=0 is the tip",
          ed.z_to_percent(0, depth, "tip-low-z") == 100.0)
    mirrored = all(
        abs(ed.z_to_percent(z, depth, "tip-low-z")
            + ed.z_to_percent(z, depth, "tip-high-z") - 100.0) < 1e-9
        for z in range(0, depth, 97)
    )
    check("the two conventions are exact mirrors", mirrored)


def test_molar_baseline():
    """The flat molar baseline, and the guard that refuses a mismatched one."""
    print("\nMolar dentin baseline")
    depth = 1000
    anchor = ed.DentinAnchor(mode=124.5, sigma=11.2, low=57.0, high=239.0,
                             pool_size=56068)

    # Measured P82-CTR-1 values: molar 119.9 against an incisor anchor of 124.5,
    # a -0.41 sigma agreement, well inside the limit.
    baseline, info = ed.molar_baseline({"dentin": 119.9, "dentin_sigma": 14.3},
                                       anchor, depth)
    check("an agreeing molar supplies the baseline",
          baseline is not None and info["source"] == "molar")
    check("the baseline is flat along z",
          baseline is not None and float(baseline.min()) == float(baseline.max())
          == 119.9, f"range=[{baseline.min():.1f}, {baseline.max():.1f}]")
    check("one reference per slice", baseline is not None and baseline.size == depth)

    # A molar several sigma off the incisor anchor means the two regions are not
    # on a common intensity scale; transferring it would be a calibration error.
    baseline, info = ed.molar_baseline({"dentin": 160.0, "dentin_sigma": 14.3},
                                       anchor, depth)
    check("a disagreeing molar is rejected",
          baseline is None and info["source"] == "rejected",
          f"delta={info.get('delta_sigma', float('nan')):+.2f} sigma")

    # Without a molar peak there is nothing to transfer.
    baseline, info = ed.molar_baseline({}, anchor, depth)
    check("a missing molar falls back to the trace",
          baseline is None and info["source"] == "unavailable")

    # The guard is two-sided: a molar far BELOW the anchor is equally untrusted.
    baseline, _ = ed.molar_baseline({"dentin": 90.0}, anchor, depth)
    check("the agreement limit is two-sided", baseline is None)


def test_ratio_uses_the_reference():
    """enamel_ratio must divide by the reference, not the fitted dentin peak.

    Near the tip the fitted dentin component drifts up into enamel as the dentin
    horn runs out, so a ratio built on it falls toward the tip purely because its
    denominator is climbing. A fixed reference denominator is what makes ratios
    comparable along z.
    """
    print("\nRatio denominator")
    rng = np.random.default_rng(31)
    reference = 120.0

    # A tip-like slice: the fitted dentin component sits well above the true
    # dentin reference, as it does once the horn has tapered out.
    values = np.concatenate([rng.normal(150.0, 13.0, 3000),
                             rng.normal(220.0, 12.0, 4000)])
    depth = 8
    volume = np.zeros((depth, 40, 40), dtype=np.float32)
    for z in range(depth):
        volume[z, 4:36, 4:36] = rng.choice(values, (32, 32)).astype(np.float32)

    anchor = ed.DentinAnchor(mode=reference, sigma=12.0, low=60.0, high=260.0,
                             pool_size=10000)
    trace = np.full(depth, reference, dtype=float)
    sizes, width_ratios, thresholds = ed.calibrate_null(
        rng.normal(reference, 12.0, 20000), anchor, draws=8)

    res = ed.analyze_slice(volume, depth // 2, anchor, sizes, width_ratios,
                           thresholds, dentin_trace=trace, iterations=0,
                           min_voxels=100)
    if not np.isfinite(res.enamel_ratio) or not np.isfinite(res.enamel_density):
        check("a tip-like slice still reports a ratio", False, f"status={res.status}")
        return
    check("a tip-like slice still reports a ratio (not rejected)",
          np.isfinite(res.enamel_ratio), f"status={res.status}")
    check("ratio == enamel / reference",
          abs(res.enamel_ratio - res.enamel_density / reference) < 1e-9,
          f"{res.enamel_ratio:.4f} vs {res.enamel_density / reference:.4f}")
    # The whole point: dividing by the drifted fitted peak would understate it.
    if np.isfinite(res.dentin_mode) and res.dentin_mode > reference:
        check("the reference denominator exceeds the fitted-peak ratio",
              res.enamel_ratio > res.enamel_density / res.dentin_mode,
              f"reference={res.enamel_ratio:.3f} "
              f"fitted={res.enamel_density / res.dentin_mode:.3f}")


# ------------------------------------------------------------------
# Real-data tests
# ------------------------------------------------------------------

def test_real_volume(path):
    """Regressions pinned to the reference CMPR volume."""
    print(f"\nReal volume: {path}")
    volume, _ = ed.load_volume(path)
    check("loaded as (Z, Y, X)", volume.ndim == 3, f"shape={volume.shape}")

    anchor, pool = ed.build_anchor(volume)
    check("dentin anchor is plausible", 100.0 < anchor.mode < 160.0,
          f"mode={anchor.mode:.1f} sigma={anchor.sigma:.1f}")

    orientation, evidence = ed.detect_orientation(volume, anchor)
    check("tip detected at low z (reference file's known orientation)",
          orientation == "tip-low-z",
          f"dense low={evidence['dense_fraction_low']:.3f} "
          f"high={evidence['dense_fraction_high']:.3f}")

    sizes, width_ratios, thresholds = ed.calibrate_null(pool, anchor, draws=25)
    # Role assignment is judged against the local dentin density, so every
    # analyze_slice call below gets the same trace the pipeline would build.
    trace = ed.trace_dentin(volume, anchor)

    # The phantom regression. Without rim erosion these slices' modes collapse
    # onto partial-volume voxels and a spurious component appears.
    #
    # These raw-z fractions are the CERVICAL end (the reference volume is
    # tip-low-z, so raw z=78%/86% are anatomical z=22%/14%). They were once
    # asserted to be unresolvable outright, which was wrong: that is the
    # secretory zone, where enamel matrix is real tissue and legitimately less
    # dense than dentin. Blanket-rejecting them discarded the entire secretory
    # stage.
    #
    # What must hold is that the phantom itself stays caught. Rim artifact and
    # real matrix are told apart by ANGULAR EXTENT, so the test now pins that:
    # eroded, these slices give a coherent cap; un-eroded, the fit lands on the
    # perimeter ring and concentration collapses below the gate.
    print("\n  Phantom regression (rim artifact must fail the cap test)")
    depth = volume.shape[0]
    for pct in (78, 86, 90):
        z = int(pct / 100.0 * (depth - 1))
        values = ed.slice_values(volume, z, 0, 0)          # no erosion
        fit = ed.decompose(values, anchor.low, anchor.high,
                           dentin_reference=anchor.mode)
        if fit is None or fit["enamel"] is None:
            check(f"raw z={pct}% un-eroded finds no component", True)
            continue
        _, conc, _, _ = ed.spatial_coherence(volume, z, fit["enamel"],
                                             fit["enamel_sigma"], 0)
        check(f"raw z={pct}% un-eroded is rejected as a ring",
              conc < ed.MIN_LOW_SIDE_CONCENTRATION,
              f"concentration={conc:.2f} < {ed.MIN_LOW_SIDE_CONCENTRATION}")

    # The flip side: with erosion, the same cervical slices are genuine
    # secretory matrix -- a single low-density cap, not a ring.
    # Secretory matrix sits at a developmental STAGE, not at a fixed array index:
    # the three reference scans are cropped differently, so the same raw z
    # fraction lands at 34% of maximum cross-section on Odam-CRE-A but 49-51% on
    # -B and -C. Addressing the zone by cross-sectional area rather than by z
    # index compares the same part of the tooth across specimens.
    print("\n  Secretory matrix must be recovered (cervical end)")
    areas = (volume > 0).reshape(depth, -1).sum(1)
    widest = int(areas.max())
    # 0.70 and 0.60 of maximum area are both still inside the secretory zone on
    # all three reference scans. Reaching further down (0.45) lands at anatomical
    # 17% on Odam-CRE-A but 11% on -B and -C, which is past their cervical loop
    # and into true partial-volume rim -- measured solidity 0.01-0.03 there,
    # against 0.18-0.38 for real matrix. Those slices SHOULD read coincident, so
    # asserting resolved_lo at a fixed depth would encode A's cropping as truth.
    for target in (0.70, 0.60):
        band = [z for z in range(depth // 2, depth)
                if abs(areas[z] - target * widest) <= 0.03 * widest]
        if not band:
            check(f"secretory zone at {target:.0%} of max area is present",
                  False, "no slice in that area band")
            continue
        z = band[len(band) // 2]
        res = ed.analyze_slice(volume, z, anchor, sizes, width_ratios, thresholds,
                               dentin_trace=trace)
        check(f"{target:.0%} of max area (raw z={z}, anatomical "
              f"{ed.z_to_percent(z, depth, 'tip-low-z'):.0f}%) -> {res.status}",
              res.status == "resolved_lo",
              f"enamel={res.enamel_density:.0f} vs dentin={res.dentin_mode:.0f}, "
              f"concentration={res.angular_concentration:.2f}, "
              f"solidity={res.solidity:.2f}")

    # Mid-tooth enamel is unambiguous and must be found.
    print("\n  Mid-tooth detection")
    for pct in (25, 50):
        z = int(pct / 100.0 * (depth - 1))
        res = ed.analyze_slice(volume, z, anchor, sizes, width_ratios, thresholds,
                               dentin_trace=trace)
        check(f"z={pct}% resolves enamel", res.status == "resolved_hi",
              f"ratio={res.enamel_ratio:.3f}" if np.isfinite(res.enamel_ratio) else res.reason)

    # No resolved slice may sit at the intensity ceiling; those are `saturated`.
    print("\n  Saturation is separated from measurement")
    vmax = float(volume.max())
    pinned = []
    for pct in (68, 72, 76, 80, 84):
        z = int(pct / 100.0 * (depth - 1))
        res = ed.analyze_slice(volume, z, anchor, sizes, width_ratios, thresholds,
                               volume_max=vmax, dentin_trace=trace)
        if res.status in ed.RESOLVED and np.isfinite(res.enamel_density):
            if res.enamel_density >= vmax - ed.SATURATION_MARGIN:
                pinned.append(pct)
    check("no 'resolved' slice reports a value at the scan ceiling", not pinned,
          f"pinned at z%={pinned}" if pinned else f"ceiling={vmax:.0f}")

    # Near-tip slices are visibly bimodal (dentin ~138, enamel ~217, with 25% of
    # voxels above 180). They must resolve; reporting them coincident means the
    # significance gate is mis-calibrated for the local dentin width.
    print("\n  Near-tip enamel must resolve")
    for pct in (88, 92):
        z = int((1.0 - pct / 100.0) * (depth - 1))
        res = ed.analyze_slice(volume, z, anchor, sizes, width_ratios, thresholds,
                               dentin_trace=trace)
        check(f"z={pct}% resolves enamel", res.status.startswith("resolved"),
              f"{res.status}: {res.reason}" if not res.status.startswith("resolved")
              else f"dentin={res.dentin_mode:.0f} enamel={res.enamel_density:.0f}")

    # The dentin reference must follow the tooth. Judging every slice against one
    # global dentin mode inverts the two roles wherever dentin has drifted far
    # enough, which is what made Odam-CRE-B and -C report ~38 coincident slices
    # against A's 4: at C z=880 the tooth body (82% of the mass) lost the "which
    # component is dentin" test to a thin bright band by 0.5 intensity units and
    # was relabelled enamel.
    print("\n  Dentin reference tracks the tooth")
    core = np.array([ed._core_intensity(volume, z)[0]
                     for z in range(0, depth, 40)])
    zs = np.array([z for z in range(0, depth, 40)])
    ok_core = np.isfinite(core) & (core <= anchor.mode
                                   + ed.MAX_CORE_DEVIATION_SIGMA * anchor.sigma)
    if ok_core.sum() >= 5:
        error = np.abs(trace[zs[ok_core]] - core[ok_core])
        check("traced reference matches the measured dentin core",
              float(np.median(error)) <= 3.0,
              f"median |trace - core| = {np.median(error):.1f} "
              f"over {int(ok_core.sum())} slices")
    check("trace stays within the physical dentin range",
          float(trace.max()) <= anchor.mode + ed.MAX_CORE_DEVIATION_SIGMA * anchor.sigma
          + 1e-6,
          f"trace range {trace.min():.0f}-{trace.max():.0f}, "
          f"ceiling {anchor.mode + ed.MAX_CORE_DEVIATION_SIGMA * anchor.sigma:.0f}")

    # Role assignment must put the BULK of the tooth on the dentin side. A slice
    # whose "enamel" covers most of the cross-section is an inverted fit.
    inverted = []
    for z in range(depth // 2, depth, 40):
        res = ed.analyze_slice(volume, z, anchor, sizes, width_ratios, thresholds,
                               dentin_trace=trace)
        if res.status == "resolved_lo" and res.enamel_fraction > 0.85:
            inverted.append((z, res.enamel_fraction))
    check("no slice reports most of the cross-section as low-density enamel",
          not inverted,
          f"{len(inverted)} inverted: {inverted[:3]}" if inverted else "none")

    # The validity gate should retire the extreme cervical end on physical
    # grounds rather than reporting numbers from an all-rim cross-section.
    print("\n  Validity gate")
    valid_mid, rim_mid, _ = ed.slice_validity(volume, int(0.50 * (depth - 1)))
    valid_end, rim_end, _ = ed.slice_validity(volume, int(0.95 * (depth - 1)))
    check("mid-tooth slice is analyzable", valid_mid, f"rim={rim_mid:.2f}")
    check("z=95% slice is rejected as mostly rim", not valid_end, f"rim={rim_end:.2f}")


def test_erosion_matters(path):
    """Erosion must be what suppresses the phantom, not an unrelated guard."""
    print("\n  Erosion is load-bearing")
    volume, _ = ed.load_volume(path)
    depth = volume.shape[0]
    z = int(0.90 * (depth - 1))

    _, rim_eroded, _ = ed.slice_validity(volume, z)
    raw = volume[z][volume[z] > 0].astype(float)
    eroded = ed.slice_values(volume, z, 0, ed.EROSION_ITERATIONS)
    if raw.size and eroded.size:
        centres_r, smooth_r = ed.build_histogram(raw, (0.0, 256.0))
        centres_e, smooth_e = ed.build_histogram(eroded, (0.0, 256.0))
        mode_raw = centres_r[int(np.argmax(smooth_r))]
        mode_eroded = centres_e[int(np.argmax(smooth_e))]
        check("erosion raises the collapsed cervical mode",
              mode_eroded > mode_raw,
              f"raw={mode_raw:.0f} -> eroded={mode_eroded:.0f}")


def _profile(densities, statuses=None):
    """A minimal cervical-loop -> tip profile with the given resolved densities."""
    out = []
    for i, d in enumerate(densities):
        res = ed.SliceResult(z=i)
        res.z_percent = 100.0 * i / max(len(densities) - 1, 1)
        res.enamel_density = float(d)
        res.status = (statuses[i] if statuses is not None
                      else ("resolved_lo" if np.isfinite(d) else "unresolvable"))
        out.append(res)
    return out


def test_onset_trim():
    """Everything cervical to the density trough is pre-enamel, not enamel."""
    print("\n  Enamel onset trimming")

    # The measured P52 failure shape: a spike that decays into the real matrix
    # trough, then the true monotone rise toward the tip.
    spike = [84.0, 84.1, 84.0, 83.9]
    rise = [65.6, 66.1, 67.3, 70.0, 80.0, 120.0, 180.0, 220.0]
    results = _profile(spike + rise)
    trimmed = ed.trim_pre_onset(results)

    check("the spike is trimmed", trimmed == len(spike), f"{trimmed} slices")
    check("the trough itself is kept",
          results[len(spike)].status == "resolved_lo",
          f"{results[len(spike)].enamel_density:.1f}")
    check("trimmed slices are labelled pre_onset",
          all(r.status == "pre_onset" for r in results[:len(spike)]))
    check("the rise is untouched",
          all(r.status == "resolved_lo" for r in results[len(spike):]))
    check("the trimmed spike reports zero enamel density",
          results[0].enamel_density == 0.0)
    check("the trim records why, for audit",
          "onset" in results[0].reason)

    # A profile that rises throughout has no spike to remove: its minimum IS its
    # first slice, so trimming it would silently delete the matrix layer.
    clean = _profile([65.0, 70.0, 90.0, 140.0, 200.0])
    check("a monotone profile is left alone", ed.trim_pre_onset(clean) == 0)

    # A trough that is barely below its neighbours is noise, not an onset.
    flat = _profile([66.0, 65.5, 66.2, 67.0, 68.0])
    check("a sub-threshold dip is not treated as onset",
          ed.trim_pre_onset(flat) == 0,
          f"rise {66.0 - 65.5:.1f} < {ed.MIN_ONSET_RISE}")

    # Cervical to the onset there is NO enamel, so `coincident` -- which claims
    # enamel is present at the dentin density -- must be withdrawn too, while
    # `unresolvable` (which claims nothing) is left alone.
    mixed = _profile(
        [float("nan"), 84.0, 118.3, 65.6, 70.0, 150.0],
        statuses=["unresolvable", "resolved_lo", "coincident",
                  "resolved_lo", "resolved_lo", "resolved_hi"])
    n = ed.trim_pre_onset(mixed)
    check("coincident before the onset is withdrawn too", n == 2,
          f"{n} re-labelled")
    check("pre-onset coincident becomes pre_onset",
          mixed[2].status == "pre_onset")
    check("unresolvable is left alone", mixed[0].status == "unresolvable")
    check("coincident AFTER the onset is untouched",
          ed.trim_pre_onset(_profile(
              [65.0, 70.0, 118.0, 150.0],
              statuses=["resolved_lo", "resolved_lo",
                        "coincident", "resolved_hi"])) == 0)

    # No enamel is a QUANTITY, so a pre-onset slice reports zero rather than a
    # blank: blank would bucket it with `unresolvable`, where the slice could
    # not be read at all, and drop it from downstream means instead of
    # contributing the zero it should.
    check("enamel density is zero", mixed[2].enamel_density == 0.0)
    check("enamel ratio is zero", mixed[2].enamel_ratio == 0.0)

    check("no resolved slices means nothing to do",
          ed.trim_pre_onset(_profile([float("nan")] * 4)) == 0)


def test_opening_fallbacks():
    """Just tipward of the onset, a dentin reading is not enamel at the dentin's density."""
    print("\n  The enamel's opening stretch")

    # The measured P22_TGM2_WT_F_2 shape, on a dentin of 100: a pre-enamel
    # spike, then matrix at ~65 interleaved with slices where the thin matrix
    # failed the solidity test and the fit fell back to the dentin, then the
    # rise, a crossover where enamel genuinely meets dentin, and the tip.
    spike = [("resolved_lo", 84.0)] * 3
    opening = [("resolved_lo", 64.0)] + [("coincident", 100.0) if k % 3 else ("resolved_lo", 65.0 + k / 20)
                                         for k in range(24)]
    rise = [("resolved_lo", 66.0 + 24.0 * k / 29) for k in range(30)]
    crossover = [("coincident", 100.0) if k % 2 else ("resolved_lo", 96.0) for k in range(10)]
    tip = [("resolved_hi", 110.0 + 4.0 * k) for k in range(10)]
    slices = spike + opening + rise + crossover + tip
    results = _profile([d for _, d in slices], statuses=[s for s, _ in slices])
    for r in results:
        r.dentin_mode = 100.0
    trimmed = ed.trim_pre_onset(results)
    begin, end = len(spike), len(spike) + len(opening)
    fallbacks = [r for r in results[begin:end] if r.enamel_density != r.enamel_density or r.status == "coincident"]

    check("the pre-enamel spike is still what the trim counts", trimmed == len(spike), f"{trimmed}")
    check("every dentin reading in the opening is withdrawn",
          len(fallbacks) == 16 and all(r.status == "unresolvable" for r in fallbacks),
          f"{sum(r.status == 'unresolvable' for r in fallbacks)} of {len(fallbacks)}")
    check("a withdrawn slice reads no enamel but keeps its dentin fit",
          all(np.isnan(r.enamel_density) and r.dentin_mode == 100.0 for r in fallbacks))
    check("and says why", "opening stretch" in fallbacks[0].reason)
    check("the matrix between them is kept",
          all(r.status == "resolved_lo" for r in results[begin:end] if r not in fallbacks))
    check("where enamel meets dentin, `coincident` stands",
          sum(r.status == "coincident" for r in results[end + len(rise):]) == 5)
    check("trimming again changes nothing",
          ed.trim_pre_onset(results) == 0 and sum(r.status == "unresolvable" for r in results) == 16)

    # Without a dentin fit there is no measuring how far the matrix has risen.
    blind = _profile([64.0, 100.0, 65.0, 70.0], statuses=["resolved_lo", "coincident", "resolved_lo", "resolved_lo"])
    ed.trim_pre_onset(blind)
    check("a profile without dentin fits is left alone", blind[1].status == "coincident")


def main():
    print("=" * 66)
    print("enamel_density.py validation suite")
    print("=" * 66)

    gate = _synthetic_gate(6000)
    print(f"\nSynthetic null gate at n=6000: D >= {gate:.0f}")

    test_false_positives(gate)
    test_two_sided_detection(gate)
    test_recovery_accuracy(gate)
    test_bounds_respected()
    test_saturation_flagged()
    test_orientation_mapping()
    test_molar_baseline()
    test_ratio_uses_the_reference()
    test_onset_trim()
    test_opening_fallbacks()

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/Volumes/Z Drive/Incisor Enamel Analysis/CMPR/volumes/resliced_volume.tif")
    import os
    if os.path.exists(path):
        test_real_volume(path)
        test_erosion_matters(path)
    else:
        print(f"\n[skip] real-data tests: {path} not reachable")

    print("\n" + "=" * 66)
    passed, total = sum(_results), len(_results)
    print(f"{passed}/{total} checks passed")
    print("=" * 66)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
