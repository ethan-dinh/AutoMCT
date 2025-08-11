# Curved Multi-Planar Reconstruction
The protocol for this procedure was based on the following [3D slicer plugin](https://www.youtube.com/watch?v=thExIlffL0I). At its core, this script is a way to slice account for the curvature of the mouse incisor. Fundamentally, it works by creating a centerline of the incisor and using that centerline to create a series of slices that the orthonormal to the centerline.

It works by:
1. Skeletonizing the incisor
2. Utilizing Dijkstra's algorithm to find the shortest path between the two ends of the incisor
3. Smoothing the centerline based on the number of points in the centerline
4. Iteratively creating vectors that are tangent to the centerline and slices that are orthonormal to the tangent vector
5. Using the slices, create a 3D volume that is a reconstruction of the incisor based on the new coordinate system

## Usage

```bash
python CMPR.py [-h] [--log] [--slice_size SLICE_SIZE] [--spacing SPACING] [--visualize]
```

## Arguments
- `-h, --help`: Show help message and exit
- `-l, --log`: Enable logging
- `-v, --visualize`: Enable interactive 3D visualization
- `--slice_size SLICE_SIZE`: The size of the slices to create (default: 64)
- `--spacing SPACING`: The spacing between the slices (default: 1.0)

## Output
The output is a 3D volume that is a reconstruction of the incisor based on the new coordinate system.