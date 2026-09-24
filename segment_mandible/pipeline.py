"""
Mandible segmentation pipeline: loads raw data and returns segmented volumes.
"""

import gc
import glob
import hashlib
import logging
import os

import numpy as np
from tqdm import tqdm

from data_io import (
    is_dicom_dir,
    load_bmp_stack,
    load_dicom_series,
    load_nrrd,
    load_tiff,
    permute_spacing_for_record,
)
from logging_setup import progress_disabled
from preprocessing import (
    clahe_volume,
    find_min_intensity_of_bone,
    find_threshold,
    non_local_means_filter,
    normalize_volume,
    reorient_mandible,
    tv_denoise_volume,
)
from segmentation import (
    polish_incisor_mask,
    recover_cervical_loop,
    recover_low_density_margin,
    segment_incisor,
    segment_incisor_by_threshold_descent,
    segment_molar_bone,
    shrink_incisor_margin,
    track_cervical_loop,
)
from segmentation.utils import dilate_mask, remove_small_islands
from validation import (
    SampleValidation,
    check_incisor_mask,
    check_incisor_seed,
    check_isolation_volume,
    check_loaded_volume,
    check_preprocessed_volume,
    check_reorientation,
    check_shrink,
)
from visualization import create_3d_visualization

logger = logging.getLogger(__name__)


def _volume_fingerprint(input_path: str, volume: np.ndarray) -> str:
    """
    A short key identifying this input volume, for naming its filter cache.

    Hashing several gigabytes would cost a noticeable fraction of what the
    cache saves, so the volume is sampled rather than read whole: shape, dtype
    and a strided subset of the data. The stride is chosen to touch every axis,
    so a re-export that changes the data without changing its size still
    produces a different key.

    The path is deliberately *not* part of the key. Scans get moved between
    drives, and keying on the absolute path turned every move into a silent
    cache miss and a full re-denoise of a volume whose cache was still valid.
    A million sampled voxels already distinguish two different scans; the
    shape is also re-checked when the cache is loaded, so a collision degrades
    to a recompute rather than to a wrong answer.
    """
    del input_path  # kept in the signature for callers; see above
    digest = hashlib.sha256()
    digest.update(str(volume.shape).encode())
    digest.update(str(volume.dtype).encode())

    # Roughly a million sampled voxels regardless of volume size: enough that
    # an edit anywhere is very likely to land on one, cheap enough to be
    # negligible next to the denoisers this cache exists to skip.
    stride = max(1, int(round((volume.size / 1_000_000) ** (1 / volume.ndim))))
    digest.update(np.ascontiguousarray(volume[::stride, ::stride, ::stride]).tobytes())
    return digest.hexdigest()[:16]


class _StrictCheckFailure(Exception):
    """
    Raised when a stage's checks fail under ``strict``.

    Signalling this by exception rather than by a return value keeps the
    pipeline body linear: the seven check points stay one-liners instead of
    each growing an ``if ... return None`` arm, and the abort is handled once
    at the top of the function.
    """


def segment_mandible(
    input_path: str,
    *,
    debug: bool = False,
    cache_dir: str | None = None,
    cache_molar_stage: bool = False,
    spacing_out: dict | None = None,
    incisor_margin_shrink_mm: float = 0.0,
    voxel_size_override: tuple[float, float, float] | None = None,
    workers: int | None = None,
    apply_dicom_rescale: bool = False,
    validation: "SampleValidation | None" = None,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Full mandible segmentation pipeline. See :func:`_run_segmentation` for the
    parameters and the stage-by-stage behaviour.

    This wrapper exists to turn a ``strict`` check failure back into the
    ``None`` return the callers already handle, so aborting early looks the
    same to them as any other failure.
    """
    try:
        return _run_segmentation(
            input_path,
            debug=debug,
            cache_dir=cache_dir,
            cache_molar_stage=cache_molar_stage,
            spacing_out=spacing_out,
            incisor_margin_shrink_mm=incisor_margin_shrink_mm,
            voxel_size_override=voxel_size_override,
            workers=workers,
            apply_dicom_rescale=apply_dicom_rescale,
            validation=validation,
            strict=strict,
        )
    except _StrictCheckFailure as failure:
        logger.error(
            "Aborting %s: %s checks failed and strict mode is on",
            os.path.basename(input_path.rstrip(os.sep)), failure,
        )
        return None


# The stage-per-try-block structure is what produces the branch and return
# counts here: each step catches its own exception and returns None with a
# message naming that step. Splitting the body would trade a genuinely linear
# pipeline for a chain of functions threading a dozen intermediates between
# them, and would lose the per-stage error messages that make a failed run
# diagnosable.
# pylint: disable=too-many-branches,too-many-return-statements
def _run_segmentation(
    input_path: str,
    *,
    debug: bool = False,
    cache_dir: str | None = None,
    cache_molar_stage: bool = False,
    spacing_out: dict | None = None,
    incisor_margin_shrink_mm: float = 0.0,
    voxel_size_override: tuple[float, float, float] | None = None,
    workers: int | None = None,
    apply_dicom_rescale: bool = False,
    validation: "SampleValidation | None" = None,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Full mandible segmentation pipeline.

    Accepts a .nrrd/.tif file, or a directory holding a DICOM series or a
    stack of slices (auto-detects .nrrd > .tif > DICOM > .bmp).

    Parameters:
        input_path: Path to a volume file or directory of slices.
        debug: Open a napari viewer after each major pipeline step.
        cache_dir: Directory for two independent caches, both safe to delete.

            1. A *filter* cache holding the normalized, NLM- and TV-denoised
               volume, keyed by the input's path and a sample of its contents.
               This is the one that matters while iterating on segmentation:
               denoising dominates a run's wall time and does not depend on
               anything downstream, so reusing it makes a re-run start at the
               thresholding step. Editing the input file changes the key and
               the cache is rebuilt.

            2. A *post-incisor* cache holding bone_molar_volume, bmp_data and
               preprocessed_volume, which skips loading, denoising, CLAHE and
               incisor segmentation outright. Off unless ``cache_molar_stage``
               is set, because it also freezes the incisor result: sharing one
               switch with the filter cache would mean that asking to skip
               denoising silently stopped incisor changes from taking effect.
        cache_molar_stage: Enable cache (2) above. Only useful when iterating
            on segment_molar_bone() with everything upstream of it settled.
        incisor_margin_shrink_mm: Trim this much off the incisor surface, in
            mm, after it is segmented. Applies to the incisor only -- the
            conservative pass that isolates surrounding bone, and the mask used
            to remove the incisor from the bone/molar volume, both stay
            unshrunk, so the trimmed rim is dropped rather than reassigned to
            bone. 0 (default) disables it.
        workers: Threads for the incisor margin shrink's distance transform.
            None (default) uses every available core.
        apply_dicom_rescale: For DICOM inputs, apply the header's
            RescaleSlope/RescaleIntercept to convert stored values to physical
            units. Default False, which keeps raw stored values -- the volume
            is min-max normalized before any threshold is taken, so an affine
            rescale does not move a segmentation boundary.
        voxel_size_override: Voxel size (dz, dy, dx) in mm in the *input*
            frame, used for the margin shrink when the file carries no spacing
            or its header is wrong. Permuted through reorientation like a
            header-derived spacing.
        validation: If given, each stage's plausibility checks are recorded
            here (see ``validation.checks``). The checks run regardless -- this
            only collects them for an end-of-run summary.
        strict: Abort the sample as soon as a stage's checks FAIL, rather than
            carrying on and producing an output that is known to be wrong.
            Failures are always logged either way.
        spacing_out: If given, ``spacing_out["record"]`` is set to the
            reorientation record and ``spacing_out["spacing"]`` to the voxel
            size (dz, dy, dx) in mm of the *returned* volumes -- read from the
            input file and permuted through reorientation's axis shuffle, so
            it describes the output frame rather than the input one. None when
            the input carried no spacing (e.g. a BMP stack).

    Returns:
        (reoriented_volume, preprocessed_volume, incisor_volume, bone_volume, molar_volume),
        or None on failure.
    """
    def _check(report) -> None:
        """
        Log a stage's report and record it for the run summary.

        Raises ``_StrictCheckFailure`` only under ``strict``: otherwise a
        failed check is a loud warning on an output that still gets produced,
        which is what you want when you are looking at the result to decide
        whether the check itself is calibrated correctly.
        """
        report.log(logger)
        if validation is not None:
            validation.add(report)
        if report.failed and strict:
            raise _StrictCheckFailure(report.stage)

    cache_path = None
    if cache_dir is not None and cache_molar_stage:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = hashlib.sha256(os.path.abspath(input_path).encode()).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"molar_bone_cache_{cache_key}.npz")

        if os.path.exists(cache_path):
            logger.info("Loading cached pre-molar-segmentation state from %s", cache_path)
            cached = np.load(cache_path, allow_pickle=True)
            bmp_data = cached["bmp_data"]
            preprocessed_volume = cached["preprocessed_volume"]
            incisor_volume = cached["incisor_volume"]
            incisor_mask = cached["incisor_mask"]
            bone_molar_volume = cached["bone_molar_volume"]
            if spacing_out is not None:
                # The cache skips reorient_mandible(), so replay the spacing it
                # produced rather than leaving the caller without one.
                cached_spacing = cached["output_spacing"] if "output_spacing" in cached else None
                if cached_spacing is not None and np.asarray(cached_spacing).size == 3:
                    spacing_out["spacing"] = tuple(float(v) for v in np.asarray(cached_spacing))
                else:
                    spacing_out["spacing"] = None
                spacing_out["record"] = (
                    cached["reorient_record"].item()
                    if "reorient_record" in cached else {}
                )
            return _segment_bone_molar_and_finish(
                bmp_data, preprocessed_volume, incisor_volume, incisor_mask,
                bone_molar_volume, debug=debug,
            )

    input_spacing: tuple[float, float, float] | None = None

    try:
        if input_path.endswith(".nrrd"):
            bmp_data, input_spacing = load_nrrd(input_path, return_spacing=True)
            if bmp_data is None:
                logger.error("Failed to load NRRD file: %s", input_path)
                return None
        elif input_path.endswith((".tif", ".tiff")):
            bmp_data, input_spacing = load_tiff(input_path, return_spacing=True)
            if bmp_data is None:
                logger.error("Failed to load TIFF file: %s", input_path)
                return None
        elif os.path.isdir(input_path):
            nrrd_files = glob.glob(os.path.join(input_path, "*.nrrd"))
            tif_files = glob.glob(os.path.join(input_path, "*.tif")) + glob.glob(os.path.join(input_path, "*.tiff"))
            bmp_files = glob.glob(os.path.join(input_path, "*.bmp"))

            if nrrd_files:
                bmp_data, input_spacing = load_nrrd(nrrd_files[0], return_spacing=True)
                if bmp_data is None:
                    logger.error("Failed to load NRRD file: %s", nrrd_files[0])
                    return None
            elif tif_files:
                bmp_data, input_spacing = load_tiff(tif_files[0], return_spacing=True)
                if bmp_data is None:
                    logger.error("Failed to load TIFF file: %s", tif_files[0])
                    return None
            # DICOM before BMP: a directory holding both is an export whose
            # BMPs are preview renderings, while the DICOMs carry the header
            # geometry and the full stored bit depth.
            elif is_dicom_dir(input_path):
                bmp_data, input_spacing = load_dicom_series(
                    input_path, return_spacing=True,
                    apply_rescale=apply_dicom_rescale,
                )
                if bmp_data is None:
                    logger.error("Failed to load DICOM series: %s", input_path)
                    return None
            elif bmp_files:
                bmp_data = load_bmp_stack(
                    input_path, file_pattern="*.bmp", exclude_pattern="*spr.bmp"
                )
            else:
                logger.error(
                    "No supported files (.nrrd, .tif, .tiff, DICOM, .bmp) found in %s",
                    input_path,
                )
                return None
        else:
            logger.error("Unsupported input path: %s", input_path)
            return None
    except Exception as e:
        logger.error("Error loading input data: %s", e)
        return None

    logger.info(
        "Loaded volume: shape %s, dtype %s, intensity range [%g, %g]",
        bmp_data.shape, bmp_data.dtype, float(bmp_data.min()), float(bmp_data.max()),
    )
    _check(check_loaded_volume(bmp_data))

    # Normalization plus the two denoisers is by far the slowest part of a run
    # and depends only on the loaded volume, so its result is cached separately
    # from the post-incisor cache below. That split is the point: the other
    # cache is written after incisor segmentation and so is useless while the
    # incisor code is what you are changing, whereas this one survives those
    # edits and turns a re-run into seconds.
    filter_cache_path = None
    if cache_dir is not None:
        filter_cache_path = os.path.join(
            cache_dir, f"filtered_{_volume_fingerprint(input_path, bmp_data)}.npy"
        )

    normalized_volume = None
    if filter_cache_path is not None and os.path.exists(filter_cache_path):
        try:
            normalized_volume = np.load(filter_cache_path)
            if normalized_volume.shape != bmp_data.shape:
                # A fingerprint collision, or a cache written by a different
                # build. Cheap to detect and cheaper to ignore than to debug
                # later as a mysteriously wrong segmentation.
                logger.warning(
                    "Ignoring filter cache %s: shape %s does not match the "
                    "loaded volume %s",
                    filter_cache_path, normalized_volume.shape, bmp_data.shape,
                )
                normalized_volume = None
            else:
                logger.info(
                    "Loaded normalized + denoised volume from %s "
                    "(skipping NLM and TV denoising)", filter_cache_path,
                )
        except Exception as e:
            # A truncated or corrupt cache must never be fatal: the inputs to
            # recompute it are all still in hand.
            logger.warning(
                "Could not read filter cache %s (%s); recomputing",
                filter_cache_path, e,
            )
            normalized_volume = None

    if normalized_volume is None:
        try:
            # bmp_data is kept; we need it at the end for the returned volumes
            normalized_volume = normalize_volume(bmp_data)

            # Denoising filters can be very slow, so we apply them after
            # normalization to speed up processing and reduce memory usage. We
            # found that applying denoising before normalization can actually
            # amplify noise in some cases, so this order seems to work best for
            # our data.
            normalized_volume = non_local_means_filter(normalized_volume)
            normalized_volume = tv_denoise_volume(normalized_volume)

        except Exception as e:
            logger.error("Error normalizing volume: %s", e)
            return None

        if filter_cache_path is not None:
            try:
                # Write via a temporary file in the same directory, then
                # rename. A run interrupted mid-write would otherwise leave a
                # truncated cache that the next run has to discover by failing
                # to read it; rename is atomic, so the cache is either absent
                # or complete.
                os.makedirs(os.path.dirname(filter_cache_path), exist_ok=True)
                # The suffix ends in .npy because np.save appends that
                # extension to any path lacking it, which would otherwise
                # write one filename and rename another.
                tmp = f"{filter_cache_path}.{os.getpid()}.tmp.npy"
                np.save(tmp, normalized_volume)
                os.replace(tmp, filter_cache_path)
                logger.info(
                    "Cached normalized + denoised volume to %s (%.1f GB)",
                    filter_cache_path, normalized_volume.nbytes / 1e9,
                )
            except Exception as e:
                # Caching is an optimisation; failing to write one must not
                # fail the run that already did the work.
                logger.warning(
                    "Could not write filter cache %s (%s); continuing",
                    filter_cache_path, e,
                )

    # Remove background by keeping voxels above 25% of minimum bone intensity. This is a
    # conservative threshold that should retain all bone and tooth structures while removing
    # most of the background. We use a percentage of the minimum bone intensity rather than
    # a fixed threshold to account for variability in scan intensity across samples.
    try:
        logger.info("Finding minimum bone intensity from middle slices")
        min_bone_intensity = find_min_intensity_of_bone(normalized_volume)

        logger.info("Finding threshold between background and bone peaks")
        bg_bone_valley = find_threshold(
            normalized_volume,
            strategy="knee",
            debug=debug
        )

        preprocessed_volume = np.where(
            normalized_volume > bg_bone_valley,
            normalized_volume,
            0,
        )

        # Clean up any remaining noise by removing small islands
        logger.info("Removing small disconnected islands")
        foreground_mask = preprocessed_volume > 0
        foreground_mask = remove_small_islands(foreground_mask, min_voxels=3500, connectivity=3)

        preprocessed_volume = np.where(foreground_mask, preprocessed_volume, 0)

        # Both numbers drive every threshold downstream, so record them
        # together: when a sample segments badly, the first question is whether
        # its thresholds landed anywhere near the others in the batch.
        logger.info(
            "Thresholds — bone floor %.4g, background/bone knee %.4g",
            min_bone_intensity, bg_bone_valley,
        )

    except Exception as e:
        logger.error("Error estimating bone intensity: %s", e)
        return None

    _check(check_preprocessed_volume(preprocessed_volume))

    if debug:
        create_3d_visualization(
            preprocessed_volume,
            additional_volumes={"Original": (normalized_volume, "gray")},
            title="[debug] Step 2 - Background Removed",
        )

    try:
        logger.info("Reorienting mandible volume")
        reorient_record: dict = {}
        # normalized_volume rides along so the incisor descent can run on it.
        # It differs from preprocessed_volume in one way that matters: it still
        # has its background. Zeroing background removes ~93% of the voxels and
        # so lifts the median tissue intensity (0.077 -> 0.455 on
        # P82-OdamcKo-CTR-2), which raises the percentile-derived floor of the
        # descent's sweep and stops it before it reaches the tooth's
        # low-density apical end. Measured on that sample, running the descent
        # on the background-removed volume halves the mask -- 278k voxels over
        # 73% of the slices becomes 135k over 27%.
        preprocessed_volume, (bmp_data, normalized_volume) = reorient_mandible(
            preprocessed_volume,
            companion_volumes=[bmp_data, normalized_volume],
            debug=debug,
            record=reorient_record,
        )
        # Reorientation transposes/swaps axes, so the per-axis spacing has to
        # follow the data -- otherwise an anisotropic scan ends up with the
        # wrong physical size attached to each axis.
        output_spacing = permute_spacing_for_record(input_spacing, reorient_record) \
            if input_spacing is not None else None
        if output_spacing is not None:
            logger.info("Output voxel spacing (Z, Y, X) mm: %s", output_spacing)
        if spacing_out is not None:
            spacing_out["spacing"] = output_spacing
            # The record lets the caller map a --voxel-size override, stated
            # in the input frame, onto the reoriented output axes.
            spacing_out["record"] = dict(reorient_record)

        # The margin shrink is in mm, so it needs the spacing of the frame the
        # mask lives in -- the reoriented one. An explicit override wins over
        # the header, matching how the CLI resolves spacing for the outputs.
        shrink_spacing = (
            permute_spacing_for_record(voxel_size_override, reorient_record)
            if voxel_size_override is not None
            else output_spacing
        )
    except Exception as e:
        logger.error("Error reorienting mandible: %s", e)
        return None

    _check(check_reorientation(preprocessed_volume, reorient_record))

    if debug:
        create_3d_visualization(
            preprocessed_volume,
            title="[debug] Step 3 - Reoriented",
        )

    try:
        logger.info("Applying CLAHE to enhance local contrast before incisor segmentation")
        clahe_preprocessed = clahe_volume(preprocessed_volume)
    except Exception as e:
        logger.error("Error applying CLAHE: %s", e)
        return None

    # Conservative Incisor Segmentation to remove the bone surrounding the incisor
    try:
        logger.info(
            "Conservative Incisor Isolation by fusion-aware threshold descent"
        )
        # The conservative incisor comes from the threshold descent rather than
        # from a valley threshold plus the slice tracker. The descent keys on
        # where the incisor stops being its own connected component as a global
        # intensity threshold falls, which is a property of the tooth's own
        # contrast against bone -- it needs no histogram valley to exist, and a
        # valley that lands in the wrong place cannot silently mis-cut it.
        #
        # It runs on preprocessed_volume, NOT the CLAHE'd one: the descent
        # needs true, globally comparable intensities, and CLAHE's per-slice,
        # per-tile remap destroys exactly that (the same attenuation maps to
        # different values in different tiles).
        conserve_incisor_mask, conservative_incisor_threshold = (
            segment_incisor_by_threshold_descent(normalized_volume)
        )
        # Recover the tooth's low-density apical end, which no single global
        # threshold reaches: relaxing the threshold inside a thin corridor
        # around the confident mask recovers it without letting bone in.
        conserve_incisor_mask = recover_low_density_margin(
            normalized_volume, conserve_incisor_mask,
            conservative_incisor_threshold,
        )
        # Then the cervical loop at the apical end, which the margin recovery
        # above cannot reach: it is the tooth's forming region and so its least
        # mineralised part, sitting well below any threshold that keeps the
        # rest of the tooth separate from bone. Without this the mask tapers to
        # a few dozen voxels per slice exactly where incisor growth studies
        # need it most.
        conserve_incisor_mask = recover_cervical_loop(
            normalized_volume, conserve_incisor_mask,
            conservative_incisor_threshold,
        )
        # Finally follow the tooth's own axis further apically. The corridor
        # recovery above thickens the mask where it already reaches, but it
        # cannot extend it: the loop's far end is dimmer than the bone beside
        # it, so no global level reaches it without also taking the jaw (at
        # 0.9x the descent's threshold the component stops at z=139; at 0.7x it
        # spans the whole volume). Tracking replaces the intensity criterion
        # with a geometric one and extends the mask 21-59 slices further.
        # Polish first, then track. polish_incisor_mask keeps only the largest
        # connected component, and the tracked slices are faint enough that
        # they sometimes touch the body of the mask only diagonally -- running
        # it afterwards discarded the whole extension on three of four P82
        # scans, putting the apical end back where it started. Tracking last
        # also means it extends a mask that has already been cleaned up.
        conserve_incisor_mask = polish_incisor_mask(conserve_incisor_mask)
        conserve_incisor_mask = track_cervical_loop(
            normalized_volume, conserve_incisor_mask,
            conservative_incisor_threshold,
        )
        logger.info(
            "Conservative incisor: %d voxels at threshold %.4g",
            int(conserve_incisor_mask.sum()), conservative_incisor_threshold,
        )

        # The descent's threshold is what separates the tooth from bone, so it
        # also defines the conservative volume the surrounding bone is read
        # from -- replacing the role the valley threshold used to play. Cut on
        # the pre-CLAHE volume (where that threshold was measured) but keep the
        # CLAHE'd values inside, which is what segment_incisor's Otsu step
        # benefits from downstream.
        #
        # Cutting preprocessed_volume with a threshold measured on
        # normalized_volume is sound because the two share an intensity scale:
        # preprocessing only zeroes voxels below the background knee, and this
        # threshold sits well above it, so every voxel it selects holds the
        # same value in both.
        #
        # The descent reports NaN when it found no seed at all. Comparing
        # against NaN is False everywhere, which would silently empty the
        # conservative volume and hand segment_molar_bone nothing; fall back to
        # the background/bone knee, which is the threshold that already
        # defined this volume's foreground.
        if not np.isfinite(conservative_incisor_threshold):
            logger.warning(
                "Threshold descent found no incisor; using the "
                "background/bone knee %.4g for the conservative volume.",
                bg_bone_valley,
            )
            conservative_incisor_threshold = bg_bone_valley

        conservative_volume = np.where(
            preprocessed_volume > conservative_incisor_threshold,
            clahe_preprocessed,
            0,
        )

        # The bone-plus-incisor foreground, captured before the incisor is
        # zeroed out below. This is the region isolation actually acts on, and
        # the only region its retained fraction is meaningful over.
        isolation_region = preprocessed_volume > conservative_incisor_threshold

        # remove the incisor from the conservative volume to isolate the surrounding bone
        logger.info("Isolating surrounding bone by removing incisor from conservative volume")
        conservative_volume = np.where(conserve_incisor_mask, 0, conservative_volume)

        # Convert to a mask
        logger.info("Creating bone mask from conservative volume")
        bone_mask = conservative_volume > 0

        # Dilate the bone mask and remove small islands to ensure we capture all the bone around the incisor without leaving gaps
        logger.info("Dilating bone mask")
        bone_mask = dilate_mask(bone_mask, radius=2)

        logger.info("Removing small islands from bone mask")
        
        # Iterate slice by slice to remove all of the small islands without removing the bones
        for i in tqdm(
            range(bone_mask.shape[0]),
            desc="Cleaning bone mask slices",
            unit="slice",
            leave=False,
            disable=progress_disabled(),
        ):
            bone_mask[i] = remove_small_islands(bone_mask[i], min_voxels=2500, connectivity=1)

        # Create a new volume that removes the bone via the conservative mask
        incisor_segmentation_volume = np.where(bone_mask, 0, clahe_preprocessed)
        del bone_mask

        # Visualize the conservative incisor mask to verify it captures the incisor without surrounding bone
        if debug:
            create_3d_visualization(
                incisor_segmentation_volume,
                additional_volumes={
                    "Conservative Incisor": (conserve_incisor_mask, "red"),
                    "Conservative Volume": (conservative_volume, "green"),
                },
                title="[debug] Conservative Incisor Isolation",
            )
        # conserve_incisor_mask is deliberately kept: it is the fallback the
        # final incisor falls back to if its own checks fail.

    except Exception as e:
        logger.error("Error during conservative incisor isolation: %s", e)
        return None

    # Measured within the thresholded bone+incisor foreground. Not against the
    # CLAHE volume: CLAHE lifts air off zero everywhere, so a `> 0` count over
    # it is ~the whole array and the ratio never drops far enough to mean
    # anything.
    _check(check_isolation_volume(incisor_segmentation_volume, isolation_region))
    del isolation_region

    try:
        logger.info("Segmenting incisor")
        # full_incisor_mask is the untrimmed segmentation; incisor_mask is what
        # gets written out. They differ only when a margin shrink is asked for.
        seed_info: dict = {}
        full_incisor_mask = segment_incisor(
            incisor_segmentation_volume, min_bone_intensity * 1.25,
            seed_out=seed_info,
        )
    except Exception as e:
        logger.error("Error segmenting incisor: %s", e)
        return None

    # The seed is checked before the mask: when both are wrong, the seed is the
    # cause and the mask is the symptom, so naming the seed first points at the
    # thing that actually needs adjusting.
    _check(
        check_incisor_seed(
            seed_info.get("slice_index"),
            int(seed_info.get("area", 0)),
            depth=incisor_segmentation_volume.shape[0],
            slice_area=int(np.prod(incisor_segmentation_volume.shape[1:])),
        )
    )

    # The final mask is scored before it is recorded, so a failure can be acted
    # on rather than only reported. When the slice tracker's result is not
    # usable -- it leaked into bone, shattered, or came back empty -- the
    # conservative threshold-descent mask replaces it: that mask is derived
    # from the tooth's own separation from bone and fails in different ways, so
    # it is a genuine second opinion rather than the same answer re-derived.
    incisor_report = check_incisor_mask(
        full_incisor_mask, preprocessed_volume, spacing=shrink_spacing
    )

    if incisor_report.failed and conserve_incisor_mask.any():
        for problem in incisor_report.problems:
            logger.warning(
                "Incisor mask rejected (%s): %s", problem.name, problem.message,
            )
        logger.warning(
            "Falling back to the conservative threshold-descent incisor "
            "(%d voxels, vs %d from the slice tracker).",
            int(conserve_incisor_mask.sum()), int(full_incisor_mask.sum()),
        )
        full_incisor_mask = conserve_incisor_mask

        # Re-check the fallback rather than assuming it is sound: it is a
        # different method, not a guaranteed-good answer, and a run where both
        # masks fail must still be reported as failing.
        incisor_report = check_incisor_mask(
            full_incisor_mask, preprocessed_volume, spacing=shrink_spacing
        )
        incisor_report.stage = "Incisor mask (conservative fallback)"

    _check(incisor_report)
    del conserve_incisor_mask

    # Freed before the shrink, not after: these three are finished with by the
    # incisor segmentation above, and the shrink's distance transform is the
    # run's memory high-water mark. Holding several GB of dead volume across it
    # is what pushes the process into swap on a large scan.
    del incisor_segmentation_volume, clahe_preprocessed, normalized_volume

    try:
        incisor_mask = shrink_incisor_margin(
            full_incisor_mask,
            margin_shrink_mm=incisor_margin_shrink_mm,
            spacing=shrink_spacing,
            workers=workers,
        )
        incisor_volume = np.where(incisor_mask, bmp_data, 0)
    except Exception as e:
        logger.error("Error shrinking incisor margin: %s", e)
        return None

    _check(check_shrink(full_incisor_mask, incisor_mask, incisor_margin_shrink_mm))

    if debug:
        create_3d_visualization(
            preprocessed_volume,
            additional_volumes={"Incisor": (incisor_volume, "orange")},
            title="[debug] Step 4 - Incisor Segmented",
        )

    try:
        logger.info("Removing incisor from volume")
        # Dilate the *untrimmed* mask: a margin shrink is meant to drop the
        # ambiguous incisor rim from the output, not hand it to the bone.
        incisor_mask_dilated = dilate_mask(full_incisor_mask, radius=2)
        
        # Use the non-CLAHE preprocessed volume so molar segmentation sees true
        # relative intensities — CLAHE's local contrast boost flattens the
        # global brightness gap between enamel and bone that segment_molar_bone
        # relies on to find enamel seeds.
        #
        # The region is the whole background-removed foreground, NOT
        # conservative_volume. conservative_volume is cut at the incisor
        # descent's threshold, which is chosen to be high enough that the
        # incisor stands alone -- far above most bone and molar dentin. On
        # WT-M-1 (f0004019) that threshold was 0.433 against a bone floor of
        # 0.27, so the molar stage received 4.3M of ~50M foreground voxels:
        # bone truncated to its densest cortex, molars reduced to enamel caps,
        # and both masks came back empty.
        bone_molar_volume = np.where(incisor_mask_dilated, 0, preprocessed_volume)
        del incisor_mask_dilated, conservative_volume, full_incisor_mask
    except Exception as e:
        logger.error("Error removing incisor: %s", e)
        return None

    if cache_path is not None:
        logger.info("Caching pre-molar-segmentation state to %s", cache_path)
        np.savez(
            cache_path,
            bmp_data=bmp_data,
            preprocessed_volume=preprocessed_volume,
            incisor_volume=incisor_volume,
            incisor_mask=incisor_mask,
            bone_molar_volume=bone_molar_volume,
            # Wrapped in a 0-d object array explicitly: savez would do this
            # implicitly for a dict, and the load path already unwraps it with
            # .item(). Saying so here keeps the round-trip visible at both ends.
            reorient_record=np.array(reorient_record, dtype=object),
            output_spacing=(
                np.asarray(output_spacing, dtype=float)
                if output_spacing is not None else np.zeros(0)
            ),
        )

    return _segment_bone_molar_and_finish(
        bmp_data, preprocessed_volume, incisor_volume, incisor_mask,
        bone_molar_volume, debug=debug,
    )


def _segment_bone_molar_and_finish(
    bmp_data: np.ndarray,
    preprocessed_volume: np.ndarray,
    incisor_volume: np.ndarray,
    incisor_mask: np.ndarray,
    bone_molar_volume: np.ndarray,
    *,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Segment bone/molar from bone_molar_volume and assemble the final outputs."""
    try:
        logger.info("Segmenting bone and molar")
        bone_mask, molar_mask = segment_molar_bone(bone_molar_volume, debug=debug)
        del bone_molar_volume

        # Dilating the molar mask to capture its porous structure is
        # deliberately off: it bled the mask into neighbouring bone. Kept here
        # because it is the first thing to try if molars come out fragmented.
        #   molar_mask = dilate_mask(molar_mask, radius=3)

        bone_volume = np.where(bone_mask, bmp_data, 0)
        molar_volume = np.where(molar_mask, bmp_data, 0)
        del bone_mask, molar_mask, incisor_mask
        gc.collect()
    except Exception as e:
        logger.error("Error segmenting bone and molar: %s", e)
        return None

    if debug:
        create_3d_visualization(
            preprocessed_volume,
            additional_volumes={
                "Bone": (bone_volume, "grey"),
                "Molar": (molar_volume, "cyan"),
            },
            title="[debug] Step 6 - Bone & Molar Segmented",
        )

    return bmp_data, preprocessed_volume, incisor_volume, bone_volume, molar_volume
