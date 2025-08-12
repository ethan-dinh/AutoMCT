"""
Visualize the mandible in an orthographic view.
"""

import os
import numpy as np
import logging
import automct as mca
from skimage.measure import regionprops
import argparse
from tqdm import tqdm
import tifffile as tiff

# Setup logging
def setup_logging():
    """Setup basic logging."""
    class ColorFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[94m',    # Blue
            'INFO': '\033[92m',     # Green
            'WARNING': '\033[93m',  # Yellow
            'ERROR': '\033[91m',    # Red
            'CRITICAL': '\033[95m', # Magenta
        }
        RESET = '\033[0m'

        def format(self, record):
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.RESET)
            record.levelname = f"{color}{levelname}{self.RESET}"
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter('[%(levelname)s] - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

# Process command line arguments
def process_arg():
    """Process command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize the mandible in an orthographic view.")
    parser.add_argument("--log", "-l", action="store_true", help="Enable logging.")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the mandible.")
    return parser.parse_args()

# Load the BMP file
def load_bmp_file(input_path):
    """Load the BMP file."""
    bmp_data = mca.load_bmp_stack(input_path, file_pattern="*.bmp", exclude_pattern="*spr.bmp")
    return bmp_data

# Visualize the mandible in an orthographic view.
def visualize_mandible(raw_volume, segmented_volume, additional_volumes=None):
    """Visualize the mandible in an orthographic view."""
    logging.info(f"Visualizing mandible...")

    # Visualize the mandible in an orthographic view.
    mca.visualization.create_3d_visualization(raw_volume, segmented_volume, additional_volumes)

# Normalize the volume
def normalize_volume(volume):
    """Normalize the volume."""
    return mca.normalize_volume(volume, method='histogram')

def save_ct_volume_as_tiff(volume: np.ndarray, output_dir: str, name: str):
    """
    Save a 3D CT volume as a multi-page TIFF.
    
    Parameters:
        volume (np.ndarray): 3D array of CT data (e.g. int16 or float32).
        output_dir (str): Directory to save the TIFF file.
        name (str): Base name for the output file (without extension).
    """
    if not os.path.exists("./segmentation_results"):
        os.makedirs("./segmentation_results")
    if not os.path.exists(os.path.join("./segmentation_results", output_dir)):
        os.makedirs(os.path.join("./segmentation_results", output_dir))

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

    output_path = os.path.join("./segmentation_results", output_dir, f"{name}.tif")
    
    # Save using BigTIFF for large volumes, ImageJ-compatible if needed
    tiff.imwrite(
        output_path,
        volume,
        dtype=volume.dtype,
        compression='zlib',
        bigtiff=True
    )

# ------------------------------------------------------------
# Find mean intensity of bone in middle slices
# ------------------------------------------------------------
def find_min_intensity_of_bone(preprocessed_volume: np.ndarray) -> float:
    """Find the mean intensity of bone in middle slices."""
    # Get middle slices
    num_slices = preprocessed_volume.shape[0]
    middle_slice_index = num_slices // 2
    start = middle_slice_index - 15
    end = middle_slice_index + 15
    middle_slices = preprocessed_volume[start:end]

    # Gaussian filter the middle slices
    middle_slices = mca.volume_loader.gaussian_filter_volume(middle_slices, sigma=10)

    # Segment middle slices
    segmented_middle = mca.segmentation.segment_volume(
        middle_slices, method='otsu', min_area=200
    )

    # Get largest region and compute average intensities
    largest_region_intensities = []
    for i in range(segmented_middle.shape[0]):
        labeled_slice = segmented_middle[i]
        slice_2d = middle_slices[i]
        
        largest_label = mca.segmentation.get_largest_region(labeled_slice)
        
        intensity = mca.segmentation.get_region_intensity(
            slice_2d, labeled_slice, largest_label
        )
        
        largest_region_intensities.append(intensity)

    # Remove anything that is below the min intensity of the bone
    min_intensity = np.min(largest_region_intensities)
    
    return min_intensity

# ------------------------------------------------------------
# Segment incisor from mandible
# ------------------------------------------------------------
def segment_incisor(preprocessed_volume: np.ndarray, mean_intensity: float) -> np.ndarray:
    """Segment the incisor from the mandible."""
    
    # Initialize the incisor mask and labeled volume
    volume_shape = preprocessed_volume.shape
    incisor_mask = np.zeros(volume_shape, dtype=bool)
    labeled_volume = np.zeros(volume_shape, dtype=int)
    prev_centroid = None
    prev_area = None

    # Iterate over the slices
    for z in tqdm(reversed(range(volume_shape[0])), desc="Segmenting incisor", total=volume_shape[0], unit="slice"):

        # Segment the slice
        slice_2d = preprocessed_volume[z]
        segmented_slice = mca.segmentation.segment_slice(slice_2d, method='otsu', min_area=200)

        # Label the slice
        labeled_slice = mca.segmentation.label_slice(segmented_slice)
        labeled_volume[z] = labeled_slice
        
        # Get the region properties
        props = regionprops(labeled_slice)
        if not props:
            continue

        # Heuristic: pick region closest to previous centroid (or most anterior)
        if prev_centroid:
            def dist(p): return np.linalg.norm(np.array(p.centroid) - np.array(prev_centroid))
            incisor_region = min(props, key=dist)
        else:
            # First slice: pick region with smallest x (most anterior)
            incisor_region = min(props, key=lambda p: p.centroid[1])  # column = x-axis

        # Heuristic: if the area is too large, skip
        if prev_area is not None and incisor_region.area > prev_area * 2:
            continue
        else:
            prev_centroid = incisor_region.centroid
            prev_area = incisor_region.area

        # Mark this region in the output mask
        incisor_mask[z] = labeled_slice == incisor_region.label

    # Isolate the incisor
    incisor_volume = np.where(incisor_mask, preprocessed_volume, 0)

    # Remove the specs that were picked up
    incisor_volume = mca.segmentation.convert_to_binary(incisor_volume, fill_holes=True)
    incisor_volume = mca.segmentation.label_3d_volume(incisor_volume, connectivity=1)
    incisor_label = mca.segmentation.get_largest_region(incisor_volume)
    incisor_mask = np.where(incisor_volume == incisor_label, incisor_volume, 0)

    return incisor_mask

# -------------------------------------
# Segment bone and molar
# -------------------------------------

def segment_bone(preprocessed_volume: np.ndarray) -> np.ndarray:
    """Segment the bone from the mandible."""
    
    # Gaussian filter the volume
    preprocessed_volume = mca.volume_loader.gaussian_filter_volume(preprocessed_volume, sigma=1)

    # Segment using otsu method
    segmented_volume = mca.segmentation.segment_volume(preprocessed_volume, method='otsu', min_area=200, nbins=1024)

    # Binary fill the segmented volume
    segmented_volume = mca.segmentation.convert_to_binary(segmented_volume, fill_holes=True)

    # Iterate over the slices
    bone_mask = np.zeros(segmented_volume.shape, dtype=bool)
    prev_centroid = None
    for z in reversed(range(segmented_volume.shape[0])):
        
        # Get the slice
        slice_2d = segmented_volume[z]

        # Label the slice
        labeled_slice = mca.segmentation.label_slice(slice_2d, connectivity=1)
        props = regionprops(labeled_slice)
        if not props:
            continue

        # Heuristic: pick region closest to previous centroid (or most anterior)
        if prev_centroid:
            def dist(p): return np.linalg.norm(np.array(p.centroid) - np.array(prev_centroid))
            bone_region = min(props, key=dist)
        else:
            # First slice: pick region with smallest x (most anterior)
            bone_region = min(props, key=lambda p: p.centroid[1])  # column = x-axis

        # Update the previous centroid
        prev_centroid = bone_region.centroid

        # Mark this region in the output mask
        bone_mask[z] = labeled_slice == bone_region.label

    # Get the bone volume
    bone_volume = np.where(bone_mask, preprocessed_volume, 0)

    # Remove the specs that were picked up
    bone_volume = mca.segmentation.convert_to_binary(bone_volume)
    bone_volume = mca.segmentation.label_3d_volume(bone_volume, connectivity=1)
    bone_label = mca.segmentation.get_largest_region(bone_volume)
    bone_mask = np.where(bone_volume == bone_label, bone_volume, 0)
    bone_volume = np.where(bone_mask, preprocessed_volume, 0)

    return bone_volume

# -------------------------------------
# Segment molar and bone in 3D
# -------------------------------------

def segment_molar_bone(preprocessed_volume: np.ndarray) -> np.ndarray:
    """Segment the molar and bone from the mandible."""

    # Gaussian filter the volume
    preprocessed_volume = mca.volume_loader.gaussian_filter_volume(preprocessed_volume, sigma=1)

    # Segment using otsu method
    segmented_volume = mca.segmentation.segment_volume(preprocessed_volume, method='otsu', min_area=100, nbins=512)

    # Binary close the volume
    segmented_volume = mca.segmentation.morphological_closing(segmented_volume, ball_radius=1)

    # Label the volume
    labeled_volume = mca.segmentation.label_3d_volume(segmented_volume, connectivity=1)

    # Get the bone region (largest region)
    bone_region = mca.segmentation.get_largest_region(labeled_volume)
    bone_mask = np.where(labeled_volume == bone_region, labeled_volume, 0)

    # Get the molar region (any region that is not the bone)
    molar_mask = np.where(labeled_volume != bone_region, labeled_volume, 0)

    # Clean up the molar mask
    # Relabel molar mask to get connected regions
    labeled_molar = mca.segmentation.label_3d_volume(molar_mask, connectivity=1)

    # Get all region properties
    molar_regions = regionprops(labeled_molar)

    if not molar_regions:
        molar_mask_cleaned = np.zeros_like(molar_mask)
    else:
        # Compute the volume of each region
        region_sizes = {r.label: r.area for r in molar_regions}
        max_area = max(region_sizes.values())
        keep_labels = [label for label, area in region_sizes.items() if area >= 0.5 * max_area]

        # Create mask with only regions ≥ 50% of max
        molar_mask_cleaned = np.isin(labeled_molar, keep_labels).astype(np.uint8)

    return bone_mask, molar_mask_cleaned

# ------------------------------------------------------------
# Segment the mandible
#
# This function will segment the mandible, incisor, bone, and molar
# It will also save the volumes as tiff files
# ------------------------------------------------------------
def segment_mandible(input_path: str) -> np.ndarray:
    # Load the BMP files
    try: 
        bmp_data = load_bmp_file(input_path)
    except Exception as e:
        logging.error(f"Error loading BMP file: {e}")
        return

    # Normalize
    try:
        normalized_volume = mca.volume_loader.normalize_volume(bmp_data)
    except Exception as e:
        logging.error(f"Error normalizing volume and initial thresholding: {e}")
        return

    # Find mean intensity of bone in middle slices
    logging.info(f"Finding mean intensity of bone in middle slices")
    try:
        min_intensity = find_min_intensity_of_bone(normalized_volume)
        
        # We want to keep any values that are greater than 30% of the mean bone intensity
        preprocessed_volume = np.where(normalized_volume > max(-500, min_intensity * 0.3), normalized_volume, 0)
    except Exception as e:
        logging.error(f"Error finding mean intensity of bone in middle slices: {e}")
        return

    # Segment the incisor
    logging.info(f"Segmenting incisor")
    try:
        incisor_mask = segment_incisor(preprocessed_volume, min_intensity)
        incisor_volume = np.where(incisor_mask, bmp_data, 0)
    except Exception as e:
        logging.error(f"Error segmenting incisor: {e}")
        return

    # Threshold the volume in preparation for bone and molar segmentation 
    # Here since the enamel is already developed for the molars, we can use a higher threshold
    threshold = min_intensity * 0.80
    threshold_volume = np.where(preprocessed_volume >= threshold, preprocessed_volume, 0)

    # Remove the incisor from the volume
    logging.info(f"Removing incisor from volume")
    try:
        incisor_mask = mca.segmentation.dilate_mask(incisor_mask, radius=2)
        bone_molar_volume = np.where(incisor_mask, 0, threshold_volume)
    except Exception as e:
        logging.error(f"Error removing incisor from volume: {e}")
        return

    # Segment the bone and molar
    logging.info(f"Segmenting bone and molar")
    try:
        bone_mask, molar_mask = segment_molar_bone(bone_molar_volume)
        bone_volume = np.where(bone_mask, bmp_data, 0)
        molar_volume = np.where(molar_mask, bmp_data, 0)
    except Exception as e:
        logging.error(f"Error segmenting bone and molar: {e}")
        return
    
    return preprocessed_volume, incisor_volume, bone_volume, molar_volume


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------

def main():
    input_path = "../../data/"
    args = process_arg()
    if args.log:
        setup_logging()

    # Find all of the folders in the input path (each folder is a stack of files)
    input_folders = [
        name for name in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, name)) and name != "Compressed"
    ]

    # Check if all of the folders have been segmented
    has_segmented = True
    for dir in input_folders:
        if not os.path.exists(os.path.join("output", f"incisor_volume_{dir}.tif")):
            logging.info(f"Folder {dir} has not been segmented")
            has_segmented = False
            break

    # Check if the mandible has already been segmented
    if not has_segmented:
        for dir in input_folders:
            logging.info(f"Segmenting {dir}")
            preprocessed_volume, incisor_volume, bone_volume, molar_volume = segment_mandible(os.path.join(input_path, dir))

            logging.info(f"Attempting to save volumes as tiff files")
            try:
                save_ct_volume_as_tiff(incisor_volume, dir, "incisor_volume")
                save_ct_volume_as_tiff(bone_volume, dir, "bone_volume")
                save_ct_volume_as_tiff(molar_volume, dir, "molar_volume")
            except Exception as e:
                logging.error(f"Error saving volumes as tiff files: {e}")

        if args.visualize:
            additional_volumes = {
                'Incisor': (incisor_volume, 'orange'),
                'Bone': (bone_volume, 'grey'),
                'Molar': (molar_volume, 'cyan')
            }
            visualize_mandible(preprocessed_volume, None, additional_volumes)
    else:
        logging.info(f"Mandible has already been segmented")

if __name__ == "__main__":
    main()