"""
Enamel Analysis

This module provides functionality for analyzing enamel in incisor volumes. Using the output from Curved Multi-Planar Reconstruction (CMPR), it quantifies enamel thickness and the intensity ratio between enamel and dentin along the full length of the incisor. The analysis involves segmenting enamel from dentin in the 2D reformatted slices and computing both metrics across the entire reconstructed volume.

Author: Ethan Dinh
Date: August 6, 2025
"""

import numpy as np
import logging
import tifffile as tiff
import argparse
from tqdm import tqdm
import os
import microct_analysis as mca
from skimage.measure import regionprops
from skimage.morphology import binary_opening, disk
from skimage import exposure
import pandas as pd

IS_VISUALIZE = False
IS_LOGGING = False

# ----------------------------------------- #
# Logging Setup
# ----------------------------------------- #

def setup_logging():
    class ColorFormatter(logging.Formatter):
        COLORS = {'DEBUG': '\033[94m', 'INFO': '\033[92m', 'WARNING': '\033[93m', 'ERROR': '\033[91m', 'CRITICAL': '\033[95m'}
        RESET = '\033[0m'
        def format(self, record):
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.RESET)
            record.levelname = f"{color}{levelname}{self.RESET}"
            return super().format(record)
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter('[%(levelname)s] - %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

# ----------------------------------------- #
# Helper Functions
# ----------------------------------------- #

def load_volume(path):
    logging.info(f"Loading volume from {path}")
    return tiff.imread(path)

def save_ct_volume_as_tiff(volume: np.ndarray, output_dir: str, name: str):
    """
    Save a 3D CT volume as a multi-page TIFF.
    
    Parameters:
        volume (np.ndarray): 3D array of CT data (e.g. int16 or float32).
        output_dir (str): Directory to save the TIFF file.
        name (str): Base name for the output file (without extension).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Log the volume details
    logging.info(f"Saving volume with shape: {volume.shape}, dtype: {volume.dtype}, min: {np.min(volume)}, max: {np.max(volume)}")

    # Downgrade float64 to float32
    # This is necessary for the TIFF file to be saved correctly
    if volume.dtype == np.float64:
        logging.info(f"Downgrading float64 to float32")
        volume = volume.astype(np.float32)

    # Ensure proper data type
    if volume.dtype not in [np.int16, np.float32]:
        volume = volume.astype(np.int16)

    output_path = os.path.join(output_dir, f"{name}.tif")
    
    # Save using BigTIFF for large volumes, ImageJ-compatible if needed
    tiff.imwrite(
        output_path,
        volume,
        dtype=volume.dtype,
        compression='zlib',
        bigtiff=True
    )

# ----------------------------------------- #
# Enamel Analysis
# ----------------------------------------- #
def crop_small_end_slices(volume, min_area=200):
    """
    Crop the small end slices from the volume.

    Parameters:
        volume (np.ndarray): 3D array of CT data (e.g. int16 or float32).
        min_area (int): The minimum area of the slice to keep.

    Returns:
        np.ndarray: The cropped volume.
    """
    start_idx = 0
    end_idx = volume.shape[0] - 1

    # Find first valid slice from the start
    for z in range(volume.shape[0]):
        labeled_slice = mca.segmentation.label_slice(volume[z] > 0)
        props = regionprops(labeled_slice)
        if props and props[0].area >= min_area:
            start_idx = z
            break

    # Find first valid slice from the end
    for z in range(volume.shape[0] - 1, -1, -1):
        labeled_slice = mca.segmentation.label_slice(volume[z] > 0)
        props = regionprops(labeled_slice)
        if props and props[0].area >= min_area:
            end_idx = z
            break

    # Zero out the ends
    volume[:start_idx] = 0
    volume[end_idx+1:] = 0

    return volume


def determine_dentin_ring_slice(volume: np.ndarray) -> int:
    """
    Determine the slice where the dentin ring is complete.

    Parameters:
        volume (np.ndarray): 3D array of CT data (e.g. int16 or float32).

    Returns:
        int: The slice index where the dentin ring is complete.
    """

    # Iterate over the slices and determine when there is a complete dentin ring
    prev_area = None
    complete_dentin_ring_slice = None
    for z in tqdm(reversed(range(volume.shape[0])), desc="Analyzing dentin ring", total=volume.shape[0], unit=" slices"):
        slice_2d = volume[z]
        binary_slice = slice_2d > 0
        binary_slice = mca.segmentation.fill_holes(binary_slice)

        # Label the slice
        labeled_slice = mca.segmentation.label_slice(binary_slice)

        # Get the region properties
        props = regionprops(labeled_slice)
        if not props:
            continue 

        # Heuristic: pick region closest to previous centroid (or most anterior)
        if prev_area is not None and props[0].area < (prev_area * 0.5):
            complete_dentin_ring_slice = z + 1
            break
        else:
            prev_area = props[0].area

    if complete_dentin_ring_slice is None:
        logging.error("No complete dentin ring found")
        return None

    logging.info(f"Dentin ring slice: {complete_dentin_ring_slice}")
    return complete_dentin_ring_slice

def find_avg_dentin_intensity(volume: np.ndarray) -> float:
    """
    Find the minimum intensity of dentin in the volume.

    Parameters:
        volume (np.ndarray): 3D array of CT data (e.g. int16 or float32).

    Returns:
        float: The minimum intensity of dentin in the volume.
    """
    num_slices = volume.shape[0]
    start = int(num_slices // 2 - (np.floor(num_slices * 0.1)))
    end = int(num_slices // 2 + (np.floor(num_slices * 0.1)))
    middle_slices = volume[start:end]

    # Segment the middle slices
    thresholds = mca.segmentation.multi_otsu_threshold(middle_slices, num_thresholds=3, nbins=512)

    # Determine the enamel mask
    dentin_mask = np.where(middle_slices <= thresholds[1], middle_slices, 0)
    dentin_mask = mca.segmentation.convert_to_binary(dentin_mask)
    dentin_labeled = mca.segmentation.label_3d_volume(dentin_mask, connectivity=1)

    average_intensity = np.mean(middle_slices[dentin_labeled == 1])
    logging.info(f"Average intensity of dentin: {average_intensity}")
    
    return average_intensity

def determine_unknown_region(volume: np.ndarray, lower_thresh: float, upper_thresh: float) -> tuple[int, int]:
    """
    Determine the region of unknown in the volume.
    """
    # Create a copy of the volume
    volume_copy = volume.copy()

    # Iterate over the slices of the mask
    areas = []
    enamel_mask = np.zeros_like(volume_copy, dtype=bool)
    for z in tqdm(range(volume_copy.shape[0]), desc="Segmenting enamel", total=volume_copy.shape[0], unit="slice"):
        slice_2d = volume_copy[z]
        enamel_mask[z] = np.logical_and(slice_2d > 0, np.logical_or(slice_2d <= lower_thresh, slice_2d >= upper_thresh))

        # Erode the mask
        enamel_mask[z] = mca.segmentation.erode_mask(enamel_mask[z], radius=2, if_2d=True)

        # Label the slice
        labeled_slice = mca.segmentation.label_slice(enamel_mask[z])
        props = regionprops(labeled_slice)
        if not props:
            continue

        # Get the largest region
        largest_region = max(props, key=lambda p: p.area)
        areas.append(largest_region.area)
        if largest_region.area < 200:
            enamel_mask[z] = np.zeros_like(enamel_mask[z], dtype=bool)
        else:
            enamel_mask[z] = labeled_slice == largest_region.label

    if IS_VISUALIZE:
        mca.visualization.create_3d_visualization(volume_copy, None, {'Enamel': (enamel_mask, 'green')})

    # Iterate through the mask to determine the region of unknown 
    # Find first slice index with area < 100
    THRESHOLD = 100
    start_idx = None
    end_idx = None

    prev_area = None
    for i, area in enumerate(areas):
        if prev_area is not None and prev_area > THRESHOLD and area < THRESHOLD:
            start_idx = i
            prev_area = area
        elif prev_area is not None and prev_area < THRESHOLD and area > THRESHOLD:
            end_idx = i
            break
        else:
            prev_area = area

    
    print(start_idx, end_idx)

    lower_z = start_idx
    upper_z = end_idx
    percent_unknown = (upper_z - lower_z) / enamel_mask.shape[0] * 100
    logging.info(f"Percent Unknown: {percent_unknown:.2f}%")
    logging.info(f"Upper Index: {upper_z}, Lower Index: {lower_z}")

    return lower_z, upper_z


def analyze_region(region: np.ndarray, pre_unknown: bool):
    """
    Analyze the region to compute volumetric characteristics and the enamel-to-dentin density ratio per slice along its full length.

    Parameters:
        region (np.ndarray): 2D array of the region to analyze.

    Returns:
        tuple: A tuple containing the volumetric characteristics (list of floats) and the enamel-to-dentin density ratio (list of float).
    """

    # Compute the volumetric characteristics
    region_copy = region.copy()

    # Preprocess the region
    histogram_equalized_region = exposure.equalize_hist(region_copy, nbins=512)

    enamel = np.zeros_like(region_copy, dtype=bool)
    dentin = np.zeros_like(region_copy, dtype=bool)

    intensity_ratios = []
    raw_enamel_intensities = []
    enamel_areas = []

    if pre_unknown:
        for z in tqdm(reversed(range(region_copy.shape[0])), desc="Segmenting enamel and dentin", total=region_copy.shape[0], unit="slice"):
            slice_2d = region_copy[z]
            slice_2d_histogram_equalized = histogram_equalized_region[z]

            # Multi-otsu thresholding
            if pre_unknown:
                thresholds = mca.segmentation.multi_otsu_threshold(slice_2d_histogram_equalized, num_thresholds=3, nbins=512)
            else:
                slice_2d_processed = exposure.adjust_gamma(slice_2d, 1.8)
                slice_2d_processed = exposure.equalize_adapthist(slice_2d_processed, clip_limit=0.03)
                thresholds = mca.segmentation.multi_otsu_threshold(slice_2d_processed, num_thresholds=3, nbins=512)

            # Segment the enamel and dentin
            if pre_unknown:
                dentin_mask = np.logical_and(slice_2d > 0, np.where(slice_2d_histogram_equalized > thresholds[1], slice_2d, 0))
                enamel_mask = np.logical_and(slice_2d > 0, np.logical_not(dentin_mask))
                enamel_mask = binary_opening(enamel_mask, disk(2))
            else:
                enamel_mask = np.logical_and(slice_2d_processed > 0, np.where(slice_2d_processed >= thresholds[1], slice_2d_processed, 0))
                dentin_mask = np.logical_and(slice_2d_processed > 0, np.where(slice_2d_processed < thresholds[1], slice_2d_processed, 0))
                dentin_mask = mca.segmentation.erode_mask(dentin_mask, radius=2, if_2d=True)

            
            # original_slice should be your grayscale image for this z
            labeled_enamel_slice = mca.segmentation.label_slice(enamel_mask)
            labeled_dentin_slice = mca.segmentation.label_slice(dentin_mask)

            # Pass intensity_image to compute mean_intensity
            enamel_props = regionprops(labeled_enamel_slice, intensity_image=slice_2d)
            dentin_props = regionprops(labeled_dentin_slice, intensity_image=slice_2d)

            if not enamel_props or not dentin_props:
                continue

            # Select region based on pre_unknown flag
            if pre_unknown:
                selected_dentin_region = max(dentin_props, key=lambda p: p.area)
                selected_enamel_region = max(enamel_props, key=lambda p: p.centroid[0])
            else:
                selected_enamel_region = max(enamel_props, key=lambda p: p.area)
                selected_dentin_region = max(dentin_props, key=lambda p: p.area)

            enamel[z] = labeled_enamel_slice == selected_enamel_region.label
            dentin[z] = labeled_dentin_slice == selected_dentin_region.label

            # Mean intensity ratio
            intensity_ratios.append(
                selected_enamel_region.mean_intensity / selected_dentin_region.mean_intensity
            )
            raw_enamel_intensities.append(selected_enamel_region.mean_intensity)
            enamel_areas.append(selected_enamel_region.area)
    else:
        thresholds = mca.segmentation.multi_otsu_threshold(region_copy, num_thresholds=3, nbins=512)
        enamel_mask = np.logical_and(region_copy > 0, np.where(region_copy >= thresholds[1], region_copy, 0))
        dentin_mask = np.logical_and(region_copy > 0, np.where(region_copy < thresholds[1], region_copy, 0))

        if IS_VISUALIZE:
            mca.visualization.create_3d_visualization(region_copy, None, {'Enamel': (enamel_mask, 'green'), 'Dentin': (dentin_mask, 'blue')})

        for z in tqdm(range(region_copy.shape[0]), desc="Segmenting enamel and dentin", total=region_copy.shape[0], unit="slice"):
            slice_2d = region_copy[z]
            enamel_slice = enamel_mask[z]
            dentin_slice = dentin_mask[z]

            labeled_enamel_slice = mca.segmentation.label_slice(enamel_slice)
            enamel_props = regionprops(labeled_enamel_slice, intensity_image=slice_2d)

            labeled_dentin_slice = mca.segmentation.label_slice(dentin_slice)
            dentin_props = regionprops(labeled_dentin_slice, intensity_image=slice_2d)
            
            if not enamel_props or not dentin_props:
                intensity_ratios.append(1)
                raw_enamel_intensities.append(0)
                enamel_areas.append(0)
                continue

            selected_enamel_region = max(enamel_props, key=lambda p: p.area)
            selected_dentin_region = max(dentin_props, key=lambda p: p.area)

            
            intensity_ratios.append(
                selected_enamel_region.mean_intensity / selected_dentin_region.mean_intensity
            )
            raw_enamel_intensities.append(selected_enamel_region.mean_intensity)
            enamel_areas.append(selected_enamel_region.area)

    return intensity_ratios, raw_enamel_intensities, enamel, dentin, enamel_areas

def enamel_analysis_pipeline(path, output_dir, file_name):
    # Load in the CMPR results
    CMPR_volume = load_volume(os.path.join(path, "resliced_volume.tif"))

    # Crop the volume from the edges if the area is less than 200
    CMPR_volume = crop_small_end_slices(CMPR_volume, min_area=200)

    # Reverse the volume
    CMPR_volume = CMPR_volume[::-1]

    # Determine the dentin ring slice
    n_slices_to_keep = int(CMPR_volume.shape[0] * 0.2) # Use the first 20% of the slices
    reduced_volume = CMPR_volume[:n_slices_to_keep]
    cropped_volume = CMPR_volume[determine_dentin_ring_slice(reduced_volume):]

    # Determine average minimum intensity of dentin
    avg_dentin_intensity = find_avg_dentin_intensity(cropped_volume)

    # Manually threshold the enamel from the dentin via the average intensity of the dentin
    upper_thresh = avg_dentin_intensity * 1.35
    lower_thresh = avg_dentin_intensity * 0.85

    # Determine the region of unknown
    lower_z, upper_z = determine_unknown_region(cropped_volume, lower_thresh, upper_thresh)

    # Segment each region (lower and upper)
    lower_region = cropped_volume[:lower_z]
    upper_region = cropped_volume[upper_z:]

    # Analyze the regions
    lower_intensity_ratios, lower_raw_enamel_intensities, lower_enamel, lower_dentin, lower_enamel_areas = analyze_region(lower_region, pre_unknown=True)
    upper_intensity_ratios, upper_raw_enamel_intensities, upper_enamel, upper_dentin, upper_enamel_areas = analyze_region(upper_region, pre_unknown=False)

    # Combine the lower and upper regions and fill in the unknown region
    combined_enamel = np.zeros_like(cropped_volume, dtype=bool)
    combined_enamel[:lower_z] = lower_enamel
    combined_enamel[upper_z:] = upper_enamel
    combined_enamel[lower_z:upper_z] = False

    combined_dentin = np.zeros_like(cropped_volume, dtype=bool)
    combined_dentin[:lower_z] = lower_dentin
    combined_dentin[upper_z:] = upper_dentin
    combined_dentin[lower_z:upper_z] = False

    # Create dataframe
    # Assemble a single intensity vector across all selected slices
    N = len(cropped_volume)
    intensity_ratios = [1] * N                  
    intensity_ratios[:lower_z] = lower_intensity_ratios          
    intensity_ratios[upper_z:] = upper_intensity_ratios
    intensity_intensities = [1] * N                  
    intensity_intensities[:lower_z] = lower_raw_enamel_intensities          
    intensity_intensities[upper_z:] = upper_raw_enamel_intensities
    enamel_areas = [1] * N                  
    enamel_areas[:lower_z] = lower_enamel_areas          
    enamel_areas[upper_z:] = upper_enamel_areas

    # Slice percentage = index / total selected slices
    slice_percentage = np.arange(N, dtype=float) / float(N)

    # Build and save DataFrame
    df = pd.DataFrame({
        'slice_percentage': slice_percentage,
        'intensity_ratios': intensity_ratios,
        'intensity_intensities': intensity_intensities,
        'enamel_areas': enamel_areas
    })
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(os.path.join(output_dir, f"{file_name}_results.csv"), index=False)

    # Save the enamel mask
    if IS_VISUALIZE:
        additional_volumes = {
            'Enamel': (combined_enamel, 'green'),
            'Dentin': (combined_dentin, 'blue'),
        }
        mca.visualization.create_3d_visualization(cropped_volume, None, additional_volumes)
    
# ----------------------------------------- #
# Main Functionality
# ----------------------------------------- #

def parse_arguments():
    parser = argparse.ArgumentParser(description="Enamel Analysis")
    parser.add_argument("--log", "-l", action="store_true", help="Enable logging")
    parser.add_argument("--visualize", "-v", action="store_true", help="Enable interactive 3D visualization")
    return parser.parse_args()

def main():
    global IS_VISUALIZE
    global IS_LOGGING
    
    args = parse_arguments()
    if args.log:
        IS_LOGGING = True
        setup_logging()

    if args.visualize:
        IS_VISUALIZE = True

    logging.info("Enamel Analysis Pipeline Starting")

    output_dir = "./results"
    if not os.path.exists(output_dir):
        logging.info("Creating output directory")
        os.makedirs(output_dir)

    # Find all of the folders in the input path (each folder is a stack of files)
    input_path = "../reorient_mandible/results/"
    input_folders = [
        name for name in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, name))
    ]

    # Check which folders have already been analyzed
    for dir in input_folders:
        if os.path.exists(os.path.join(output_dir, dir, f"{dir}_results.csv")):
            input_folders.remove(dir)

    if not input_folders:
        logging.info("All folders have already been analyzed")
        return

    for dir in input_folders:
        logging.info(f"Processing {dir}")
        enamel_analysis_pipeline(
            path=os.path.join(input_path, dir),
            output_dir=os.path.join(output_dir, dir),
            file_name=dir
        )

    logging.info("Enamel Analysis Pipeline Completed Successfully")

if __name__ == "__main__":
    main()
