"""
Mandible segmentation pipeline: loads raw data and returns segmented volumes.
"""

import gc
import glob
import hashlib
import logging
import os

import numpy as np

from data_io import load_bmp_stack, load_nrrd, load_tiff
from preprocessing import (
    clahe_volume,
    find_min_intensity_of_bone,
    find_threshold,
    non_local_means_filter,
    normalize_volume,
    reorient_mandible,
    tv_denoise_volume,
)
from segmentation import segment_incisor, segment_molar_bone
from segmentation.utils import dilate_mask, remove_small_islands
from visualization import create_3d_visualization
from tqdm import tqdm

logger = logging.getLogger(__name__)


def segment_mandible(
    input_path: str,
    debug: bool = False,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Full mandible segmentation pipeline.

    Accepts a .nrrd/.tif file or a directory (auto-detects .nrrd > .tif > .bmp stack).

    Parameters:
        input_path: Path to a volume file or directory of slices.
        debug: Open a napari viewer after each major pipeline step.
        cache_dir: If given, cache bone_molar_volume/bmp_data/preprocessed_volume
            here after incisor removal and reuse them on later calls with the
            same input_path — skips loading, denoising, CLAHE, and incisor
            segmentation entirely. Useful when iterating on
            segment_molar_bone() alone. Delete the cache dir to force a
            full rerun.

    Returns:
        (reoriented_volume, preprocessed_volume, incisor_volume, bone_volume, molar_volume),
        or None on failure.
    """
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = hashlib.sha256(os.path.abspath(input_path).encode()).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"molar_bone_cache_{cache_key}.npz")

        if os.path.exists(cache_path):
            logger.info("Loading cached pre-molar-segmentation state from %s", cache_path)
            cached = np.load(cache_path)
            bmp_data = cached["bmp_data"]
            preprocessed_volume = cached["preprocessed_volume"]
            incisor_volume = cached["incisor_volume"]
            incisor_mask = cached["incisor_mask"]
            bone_molar_volume = cached["bone_molar_volume"]
            return _segment_bone_molar_and_finish(
                bmp_data, preprocessed_volume, incisor_volume, incisor_mask,
                bone_molar_volume, debug,
            )

    try:
        if input_path.endswith(".nrrd"):
            bmp_data = load_nrrd(input_path)
            if bmp_data is None:
                logger.error("Failed to load NRRD file: %s", input_path)
                return None
        elif input_path.endswith((".tif", ".tiff")):
            bmp_data = load_tiff(input_path)
            if bmp_data is None:
                logger.error("Failed to load TIFF file: %s", input_path)
                return None
        elif os.path.isdir(input_path):
            nrrd_files = glob.glob(os.path.join(input_path, "*.nrrd"))
            tif_files = glob.glob(os.path.join(input_path, "*.tif")) + glob.glob(os.path.join(input_path, "*.tiff"))
            bmp_files = glob.glob(os.path.join(input_path, "*.bmp"))

            if nrrd_files:
                bmp_data = load_nrrd(nrrd_files[0])
                if bmp_data is None:
                    logger.error("Failed to load NRRD file: %s", nrrd_files[0])
                    return None
            elif tif_files:
                bmp_data = load_tiff(tif_files[0])
                if bmp_data is None:
                    logger.error("Failed to load TIFF file: %s", tif_files[0])
                    return None
            elif bmp_files:
                bmp_data = load_bmp_stack(
                    input_path, file_pattern="*.bmp", exclude_pattern="*spr.bmp"
                )
            else:
                logger.error("No supported files (.nrrd, .tif, .tiff, .bmp) found in %s", input_path)
                return None
        else:
            logger.error("Unsupported input path: %s", input_path)
            return None
    except Exception as e:
        logger.error("Error loading input data: %s", e)
        return None

    try:
        # bmp_data is kept; we need it at the end for the returned volumes
        normalized_volume = normalize_volume(bmp_data)
        
        # Denoising filters can be very slow, so we apply them after normalization to speed up processing and reduce memory usage. We found that applying denoising before normalization can actually amplify noise in some cases, so this order seems to work best for our data.
        normalized_volume = non_local_means_filter(normalized_volume)
        normalized_volume = tv_denoise_volume(normalized_volume)
    
    except Exception as e:
        logger.error("Error normalizing volume: %s", e)
        return None

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

        logger.info("Determining conservative threshold for incisor isolation")
        conservative_threshold = find_threshold(
            normalized_volume,
            debug=debug,
            strategy="valley",
            fallback_fraction=0.38,
            bias=0.75
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

    except Exception as e:
        logger.error("Error estimating bone intensity: %s", e)
        return None

    if debug:
        create_3d_visualization(
            preprocessed_volume,
            additional_volumes={"Original": (normalized_volume, "gray")},
            title="[debug] Step 2 - Background Removed",
        )

    try:
        logger.info("Reorienting mandible volume")
        preprocessed_volume, (bmp_data,) = reorient_mandible(
            preprocessed_volume,
            companion_volumes=[bmp_data],
            debug=debug,
        )
    except Exception as e:
        logger.error("Error reorienting mandible: %s", e)
        return None

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
        logger.info("Conservative Incisor Isolation")
        conservative_volume = np.where(
            clahe_preprocessed > conservative_threshold,
            clahe_preprocessed,
            0,
        )

        conserve_incisor_mask = segment_incisor(conservative_volume, min_bone_intensity)
        
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
        del conserve_incisor_mask

    except Exception as e:
        logger.error("Error during conservative incisor isolation: %s", e)
        return None

    try:
        logger.info("Segmenting incisor")
        incisor_mask = segment_incisor(incisor_segmentation_volume, min_bone_intensity * 1.25)
        incisor_volume = np.where(incisor_mask, bmp_data, 0)
        del incisor_segmentation_volume, clahe_preprocessed
    except Exception as e:
        logger.error("Error segmenting incisor: %s", e)
        return None

    if debug:
        create_3d_visualization(
            preprocessed_volume,
            additional_volumes={"Incisor": (incisor_volume, "orange")},
            title="[debug] Step 4 - Incisor Segmented",
        )

    try:
        logger.info("Removing incisor from volume")
        incisor_mask_dilated = dilate_mask(incisor_mask, radius=2)
        
        # Use the non-CLAHE preprocessed volume (masked to the same bone/molar
        # region as conservative_volume) so molar segmentation sees true
        # relative intensities — CLAHE's local contrast boost flattens the
        # global brightness gap between enamel and bone that segment_molar_bone
        # relies on to find enamel seeds.
        
        bone_molar_region = conservative_volume > 0
        bone_molar_volume = np.where(
            incisor_mask_dilated, 0, np.where(bone_molar_region, preprocessed_volume, 0)
        )
        del incisor_mask_dilated, conservative_volume, bone_molar_region
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
        )

    return _segment_bone_molar_and_finish(
        bmp_data, preprocessed_volume, incisor_volume, incisor_mask,
        bone_molar_volume, debug,
    )


def _segment_bone_molar_and_finish(
    bmp_data: np.ndarray,
    preprocessed_volume: np.ndarray,
    incisor_volume: np.ndarray,
    incisor_mask: np.ndarray,
    bone_molar_volume: np.ndarray,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Segment bone/molar from bone_molar_volume and assemble the final outputs."""
    try:
        logger.info("Segmenting bone and molar")
        bone_mask, molar_mask = segment_molar_bone(bone_molar_volume, debug=debug)
        del bone_molar_volume

        # Dilate the molar_mask to ensure we capture the entire molar structure, which can be porous and have gaps in the initial segmentation
        """ logger.info("Dilating molar mask to capture porous structure")
        molar_mask = dilate_mask(molar_mask, radius=3) """

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
