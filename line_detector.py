"""
Climber Detector for FIRST Robotics

Detects the climber structure from a RealSense point cloud.
Supports two modes:
  - 'rectangle': Detect the flat rectangular face of the climber (side view).
                  Uses RANSAC plane fitting + oriented bounding box to find
                  the rectangle's center, dimensions, and orientation.
  - 'lines':     Detect vertical/horizontal lines/cylinders (front view).

Usage:
    python line_detector.py <pcd_file> [options]

Examples:
    python line_detector.py scan.pcd --mode rectangle
    python line_detector.py scan.pcd --mode lines --max-lines 10
    python line_detector.py scan.pcd --mode rectangle --max-distance 2.0 --downsample 0.01
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys

try:
    import pyransac3d as pyrsc
except ImportError:
    print("Error: pyransac3d is not installed. Install it with: pip install pyransac3d")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is not installed. Install it with: pip install open3d")
    sys.exit(1)

# Optional: pyransac provides a generic RANSAC framework with adaptive
# iteration (confidence-based early stopping).  When available, custom 3D
# models below allow using it as an alternative backend.
try:
    from pyransac import find_inliers as pyransac_find_inliers, RansacParams
    from pyransac.base import Model as PyransacModel
    HAS_PYRANSAC = True
except ImportError:
    HAS_PYRANSAC = False


# =============================================================================
# Custom pyransac 3D models (adaptive-iteration RANSAC backend)
# =============================================================================

if HAS_PYRANSAC:
    from dataclasses import dataclass

    @dataclass(order=True)
    class Point3D:
        """Simple 3D point used by pyransac models."""
        x: float
        y: float
        z: float

    class Plane3D(PyransacModel):
        """
        3D plane model for pyransac.

        A plane is defined by the equation  Ax + By + Cz + D = 0.
        ``make_model`` computes the plane from exactly 3 non-collinear
        points.  ``calc_error`` returns the perpendicular distance from
        a point to the plane.
        """
        def __init__(self):
            self.a = None
            self.b = None
            self.c = None
            self.d = None

        def make_model(self, points):
            if len(points) != 3:
                raise ValueError(f"Need 3 points to define a plane, got {len(points)}")
            p0, p1, p2 = points
            v1 = np.array([p1.x - p0.x, p1.y - p0.y, p1.z - p0.z])
            v2 = np.array([p2.x - p0.x, p2.y - p0.y, p2.z - p0.z])
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                # Degenerate (collinear) — force large errors
                self.a = self.b = self.c = self.d = 0.0
                return
            normal = normal / norm
            self.a, self.b, self.c = normal
            self.d = -(self.a * p0.x + self.b * p0.y + self.c * p0.z)

        def calc_error(self, point):
            if self.a is None:
                return float('inf')
            return abs(self.a * point.x + self.b * point.y
                       + self.c * point.z + self.d)

    class Line3D(PyransacModel):
        """
        3D line model for pyransac.

        A line is defined by a point ``p0`` and a unit direction ``d``.
        ``make_model`` computes the line from exactly 2 points.
        ``calc_error`` returns the perpendicular distance from a point
        to the infinite line.
        """
        def __init__(self):
            self.p0 = None
            self.direction = None

        def make_model(self, points):
            if len(points) != 2:
                raise ValueError(f"Need 2 points to define a line, got {len(points)}")
            a, b = points
            d = np.array([b.x - a.x, b.y - a.y, b.z - a.z])
            norm = np.linalg.norm(d)
            if norm < 1e-12:
                self.p0 = None
                self.direction = None
                return
            self.direction = d / norm
            self.p0 = np.array([a.x, a.y, a.z])

        def calc_error(self, point):
            if self.p0 is None:
                return float('inf')
            v = np.array([point.x, point.y, point.z]) - self.p0
            proj = np.dot(v, self.direction) * self.direction
            return float(np.linalg.norm(v - proj))


def load_pcd_file(pcd_file):
    """
    Load point cloud from PCD file using Open3D
    
    Parameters:
    -----------
    pcd_file : str
        Path to .pcd file
    
    Returns:
    --------
    points : numpy array (N, 3)
        Point cloud data
    """
    try:
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            raise ValueError("Point cloud is empty")
        
        print(f"Loaded point cloud with {len(points)} points")
        return points
        
    except FileNotFoundError:
        print(f"Error: File '{pcd_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading PCD file: {e}")
        sys.exit(1)


def filter_by_distance(points, max_distance=1.25, origin=None):
    """
    Remove all points beyond a maximum distance from the origin (camera).
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        3D point cloud
    max_distance : float
        Maximum Euclidean distance from origin in meters (default: 1.25)
    origin : numpy array (3,) or None
        Origin point (camera location). If None, uses [0, 0, 0]
    
    Returns:
    --------
    filtered_points : numpy array (M, 3)
        Filtered point cloud with only nearby points
    """
    if origin is None:
        origin = np.zeros(3)
    
    distances = np.linalg.norm(points - origin, axis=1)
    mask = distances <= max_distance
    filtered_points = points[mask]
    
    removed = len(points) - len(filtered_points)
    print(f"  Distance filter: removed {removed} points beyond {max_distance}m "
          f"({len(filtered_points)} remaining)")
    
    return filtered_points


def remove_ground_plane(points, distance_threshold=0.03, max_iterations=1000,
                        vertical_axis=1):
    """
    Detect and remove the ground plane using RANSAC.
    
    Uses pyransac3d to fit a plane, then checks if the plane normal
    is approximately aligned with the vertical axis (i.e. it's a ground/floor
    plane) and removes its inlier points.
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        3D point cloud
    distance_threshold : float
        RANSAC distance threshold for plane inliers (default: 0.03m)
    max_iterations : int
        Maximum RANSAC iterations (default: 1000)
    vertical_axis : int
        Index of vertical axis: 0=X, 1=Y, 2=Z (default: 1=Y)
    
    Returns:
    --------
    filtered_points : numpy array (M, 3)
        Point cloud with ground plane removed
    plane_eq : numpy array (4,) or None
        Plane equation [A, B, C, D] if found, None otherwise
    """
    if len(points) < 3:
        return points, None
    
    plane = pyrsc.Plane()
    plane_eq, inliers = plane.fit(points, thresh=distance_threshold,
                                  maxIteration=max_iterations)
    
    if len(inliers) == 0:
        print("  Ground plane removal: no plane found")
        return points, None
    
    # Check if the detected plane is approximately horizontal (ground)
    # The plane normal is [A, B, C] from Ax + By + Cz + D = 0
    normal = np.array(plane_eq[:3])
    normal = normal / np.linalg.norm(normal)
    
    # Check alignment with vertical axis
    vertical_unit = np.zeros(3)
    vertical_unit[vertical_axis] = 1.0
    alignment = abs(np.dot(normal, vertical_unit))
    
    # If the normal is mostly vertical, it's a ground plane
    angle_from_vertical = np.rad2deg(np.arccos(np.clip(alignment, -1.0, 1.0)))
    
    if angle_from_vertical > 30:
        print(f"  Ground plane removal: detected plane is not horizontal "
              f"(angle from vertical axis: {angle_from_vertical:.1f}°), skipping")
        return points, None
    
    # Remove inlier points (the ground)
    mask = np.ones(len(points), dtype=bool)
    mask[inliers] = False
    filtered_points = points[mask]
    
    axis_names = ['X', 'Y', 'Z']
    print(f"  Ground plane removal: removed {len(inliers)} ground points "
          f"(normal aligned {angle_from_vertical:.1f}° from {axis_names[vertical_axis]}-axis, "
          f"{len(filtered_points)} remaining)")
    print(f"    Plane equation: {plane_eq[0]:.4f}x + {plane_eq[1]:.4f}y + "
          f"{plane_eq[2]:.4f}z + {plane_eq[3]:.4f} = 0")
    
    return filtered_points, plane_eq


def downsample_points(points, voxel_size=0.01):
    """
    Downsample point cloud using voxel grid filtering via Open3D.
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        3D point cloud
    voxel_size : float
        Size of voxel grid cells in meters (default: 0.01 = 1cm)
    
    Returns:
    --------
    downsampled_points : numpy array (M, 3)
        Downsampled point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    
    print(f"  Downsample (voxel {voxel_size}m): {len(points)} -> {len(downsampled_points)} points")
    
    return downsampled_points


# =============================================================================
# Rectangle Detection (side-view climber)
# =============================================================================

def detect_rectangle(points, plane_thresh=0.02, plane_iterations=1000,
                     min_inliers=50, vertical_axis=1,
                     use_pyransac=False, confidence=0.999):
    """
    Detect a rectangular structure in the point cloud.
    
    Strategy:
      1. Use RANSAC to find the dominant plane (the flat face of the climber).
      2. Project the plane's inlier points onto the plane.
      3. Compute the oriented bounding box (OBB) via PCA to get the rectangle's
         center, width, height, corners, and orientation.
    
    This is designed for the FIRST climber side view, where the camera sees
    a large flat rectangular surface.
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        3D point cloud (after preprocessing)
    plane_thresh : float
        RANSAC distance threshold for plane inliers (default: 0.02m)
    plane_iterations : int
        Maximum RANSAC iterations for plane fitting (default: 1000)
    min_inliers : int
        Minimum inliers required for a valid rectangle (default: 50)
    vertical_axis : int
        Index of vertical axis: 0=X, 1=Y, 2=Z (default: 1=Y)
    use_pyransac : bool
        If True, use pyransac's adaptive-iteration RANSAC with confidence-
        based early stopping instead of pyransac3d's fixed-iteration RANSAC.
        This can be faster when the plane is easy to find (high inlier ratio)
        because it stops iterating once the confidence threshold is met.
    confidence : float
        Confidence level for pyransac adaptive stopping (default: 0.999).
        Higher values run more iterations but are more likely to find the
        best model.  Only used when use_pyransac=True.
    
    Returns:
    --------
    result : dict or None
        Dictionary with rectangle parameters:
        - 'center_3d':    Center of the rectangle in 3D (numpy array (3,))
        - 'corners_3d':   4 corner points in 3D (numpy array (4, 3))
        - 'width':        Width of the rectangle (shorter dimension) in meters
        - 'height':       Height of the rectangle (taller dimension) in meters
        - 'normal':       Plane normal vector (numpy array (3,))
        - 'plane_eq':     Plane equation [A, B, C, D]
        - 'up_direction': Direction along the rectangle's height (vertical edge)
        - 'across_direction': Direction along the rectangle's width (horizontal edge)
        - 'inliers':      Indices of inlier points
        - 'n_inliers':    Number of inlier points
        - 'inlier_points': The inlier point coordinates (numpy array (M, 3))
    """
    if len(points) < min_inliers:
        print("  Not enough points for rectangle detection")
        return None
    
    # Step 1: Find the dominant plane with RANSAC
    if use_pyransac and HAS_PYRANSAC:
        # --- pyransac backend: adaptive iteration with early stopping ---
        print(f"\n  Fitting plane with pyransac (thresh={plane_thresh}, "
              f"max_iter={plane_iterations}, confidence={confidence})...")
        point_list = [Point3D(p[0], p[1], p[2]) for p in points]
        params = RansacParams(samples=3, iterations=plane_iterations,
                              confidence=confidence, threshold=plane_thresh)
        model = Plane3D()
        inlier_objs = pyransac_find_inliers(point_list, model, params)

        # Map inlier objects back to indices
        inlier_set = set(id(p) for p in inlier_objs)
        inliers = [i for i, p in enumerate(point_list) if id(p) in inlier_set]

        if len(inliers) < min_inliers:
            print(f"  Plane has only {len(inliers)} inliers (need {min_inliers}), skipping")
            return None

        # Rebuild the plane equation from the final model state
        plane_eq = [model.a, model.b, model.c, model.d]
    else:
        # --- pyransac3d backend: fixed iteration ---
        print(f"\n  Fitting plane with RANSAC (thresh={plane_thresh}, max_iter={plane_iterations})...")
        plane = pyrsc.Plane()
        plane_eq, inliers = plane.fit(points, thresh=plane_thresh,
                                      maxIteration=plane_iterations)
    
    if len(inliers) < min_inliers:
        print(f"  Plane has only {len(inliers)} inliers (need {min_inliers}), skipping")
        return None
    
    inlier_points = points[inliers]
    normal = np.array(plane_eq[:3])
    normal = normal / np.linalg.norm(normal)
    
    print(f"  Plane found: {len(inliers)} inliers")
    print(f"    Equation: {plane_eq[0]:.4f}x + {plane_eq[1]:.4f}y + "
          f"{plane_eq[2]:.4f}z + {plane_eq[3]:.4f} = 0")
    
    # Step 1b: Cluster the plane inliers and keep only the largest cluster.
    # This isolates the single big rectangle (the climber face) from any
    # stray coplanar points belonging to other surfaces.
    pcd_inliers = o3d.geometry.PointCloud()
    pcd_inliers.points = o3d.utility.Vector3dVector(inlier_points)
    
    # DBSCAN clustering — eps is the max distance between neighboring points
    # in a cluster, min_points is the minimum cluster size.
    cluster_eps = plane_thresh * 3  # reasonable neighborhood radius
    labels = np.array(pcd_inliers.cluster_dbscan(
        eps=cluster_eps, min_points=10, print_progress=False
    ))
    
    if len(labels) == 0 or labels.max() < 0:
        print("  Clustering failed — using all inlier points")
    else:
        # Find the largest cluster
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        largest_label = unique_labels[counts.argmax()]
        largest_mask = labels == largest_label
        
        print(f"  Clustered into {len(unique_labels)} group(s), "
              f"keeping largest ({counts.max()} points, "
              f"discarded {len(inlier_points) - counts.max()})")
        
        inlier_points = inlier_points[largest_mask]
        inliers = np.array(inliers)[largest_mask]
    
    # Step 2: Project inlier points onto the plane to get a clean 2D representation
    # Build a local 2D coordinate system on the plane
    # Choose two orthogonal axes on the plane surface
    
    # Pick an initial vector not parallel to the normal
    vertical_unit = np.zeros(3)
    vertical_unit[vertical_axis] = 1.0
    
    # The "up" direction on the plane: project the vertical axis onto the plane
    up_on_plane = vertical_unit - np.dot(vertical_unit, normal) * normal
    up_norm = np.linalg.norm(up_on_plane)
    if up_norm < 1e-10:
        # Normal is parallel to vertical axis — plane is horizontal, pick arbitrary up
        alt = np.array([1.0, 0.0, 0.0]) if vertical_axis != 0 else np.array([0.0, 0.0, 1.0])
        up_on_plane = alt - np.dot(alt, normal) * normal
        up_norm = np.linalg.norm(up_on_plane)
    up_on_plane = up_on_plane / up_norm
    
    # The "across" direction is perpendicular to both normal and up
    across_on_plane = np.cross(normal, up_on_plane)
    across_on_plane = across_on_plane / np.linalg.norm(across_on_plane)
    
    # Step 3: Project inlier points into 2D plane coordinates
    centroid = np.mean(inlier_points, axis=0)
    diff = inlier_points - centroid
    coords_2d = np.column_stack([
        np.dot(diff, across_on_plane),  # u-axis (across)
        np.dot(diff, up_on_plane)       # v-axis (up)
    ])
    
    # Step 4: Compute oriented bounding box via PCA on the 2D coordinates
    # PCA gives us the principal axes of the point distribution
    cov = np.cov(coords_2d.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue descending
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Rotate points into PCA frame
    coords_pca = coords_2d @ eigenvectors
    
    # Bounding box in PCA frame
    pca_min = coords_pca.min(axis=0)
    pca_max = coords_pca.max(axis=0)
    
    dim1 = pca_max[0] - pca_min[0]  # extent along first principal axis
    dim2 = pca_max[1] - pca_min[1]  # extent along second principal axis
    
    # Determine which is width (shorter) and which is height (taller)
    # For the climber, height should be the larger dimension
    if dim1 >= dim2:
        height = dim1
        width = dim2
    else:
        height = dim2
        width = dim1
    
    # Compute 4 corners in PCA 2D space, then transform back to 3D
    corners_pca = np.array([
        [pca_min[0], pca_min[1]],
        [pca_max[0], pca_min[1]],
        [pca_max[0], pca_max[1]],
        [pca_min[0], pca_max[1]],
    ])
    
    # Transform corners back to 2D plane coords
    corners_2d = corners_pca @ eigenvectors.T
    
    # Transform corners back to 3D
    corners_3d = np.array([
        centroid + c[0] * across_on_plane + c[1] * up_on_plane
        for c in corners_2d
    ])
    
    # Center in 3D
    center_3d = centroid + ((pca_min + pca_max) / 2)[0] * (eigenvectors[:, 0] @ np.array([across_on_plane, up_on_plane])) \
                         + ((pca_min + pca_max) / 2)[1] * (eigenvectors[:, 1] @ np.array([across_on_plane, up_on_plane]))
    # Simpler: center is mean of corners
    center_3d = corners_3d.mean(axis=0)
    
    # Determine which rectangle edge direction aligns more with vertical
    # PCA axis 0 in 3D:
    pca_axis0_3d = eigenvectors[0, 0] * across_on_plane + eigenvectors[1, 0] * up_on_plane
    pca_axis1_3d = eigenvectors[0, 1] * across_on_plane + eigenvectors[1, 1] * up_on_plane
    
    # Check which PCA axis is more vertical
    v_align_0 = abs(np.dot(pca_axis0_3d, vertical_unit))
    v_align_1 = abs(np.dot(pca_axis1_3d, vertical_unit))
    
    if v_align_0 >= v_align_1:
        up_direction = pca_axis0_3d
        across_direction = pca_axis1_3d
    else:
        up_direction = pca_axis1_3d
        across_direction = pca_axis0_3d
    
    # Make sure up_direction points in the positive vertical direction
    if np.dot(up_direction, vertical_unit) < 0:
        up_direction = -up_direction
    
    # Normalize
    up_direction = up_direction / np.linalg.norm(up_direction)
    across_direction = across_direction / np.linalg.norm(across_direction)
    
    aspect_ratio = height / width if width > 1e-10 else float('inf')
    
    # Compute the horizontal center (X coordinate of the rectangle center)
    # This is what the robot needs to steer toward
    horizontal_center = center_3d[0]
    
    # Compute distance from the camera (origin) to the rectangle center
    distance_to_center = np.linalg.norm(center_3d)
    
    # Compute depth: perpendicular distance from the camera to the plane of
    # the rectangle.  This is how far "forward" the robot must drive along
    # the plane's normal direction to be on the same plane as the rectangle.
    # Using the plane equation Ax + By + Cz + D = 0 evaluated at origin:
    #   depth = |A*0 + B*0 + C*0 + D| / ||(A,B,C)|| = |D| / ||(A,B,C)||
    # Since we already have a unit normal, this simplifies to |D / 1| = |D|.
    # But we recalculate properly in case of floating point drift.
    plane_normal_len = np.linalg.norm(plane_eq[:3])
    if plane_normal_len > 1e-10:
        depth = abs(plane_eq[3]) / plane_normal_len
    else:
        depth = abs(center_3d[2])  # fallback: use Z coordinate
    
    axis_names = ['X', 'Y', 'Z']
    print(f"  Rectangle detected:")
    print(f"    Center:             ({center_3d[0]:.4f}, {center_3d[1]:.4f}, {center_3d[2]:.4f})")
    print(f"    Horizontal center:  {horizontal_center:+.4f} m (X offset from camera)")
    print(f"    Depth:              {depth:.4f} m (perpendicular distance to plane)")
    print(f"    Distance to center: {distance_to_center:.4f} m (Euclidean)")
    print(f"    Width:              {width:.4f} m")
    print(f"    Height:             {height:.4f} m")
    print(f"    Aspect ratio:       {aspect_ratio:.2f}")
    print(f"    Normal:             ({normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f})")
    print(f"    Inliers:            {len(inliers)}")
    
    return {
        'center_3d': center_3d,
        'horizontal_center': horizontal_center,
        'depth': depth,
        'distance_to_center': distance_to_center,
        'corners_3d': corners_3d,
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
        'normal': normal,
        'plane_eq': plane_eq,
        'up_direction': up_direction,
        'across_direction': across_direction,
        'inliers': inliers,
        'n_inliers': len(inliers),
        'inlier_points': inlier_points,
    }


def visualize_rectangle_3d(points, rect, title="3D Rectangle Detection"):
    """
    Visualize the detected rectangle in the 3D point cloud.
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        Original point cloud
    rect : dict
        Rectangle detection result from detect_rectangle()
    title : str
        Plot title
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               alpha=0.08, s=1, c='lightgray', label='All points')
    
    # Plot inlier points (the rectangle surface)
    inlier_pts = rect['inlier_points']
    ax.scatter(inlier_pts[:, 0], inlier_pts[:, 1], inlier_pts[:, 2],
               alpha=0.3, s=3, c='cyan', label='Rectangle surface')
    
    # Draw rectangle outline
    corners = rect['corners_3d']
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color='red', linewidth=3, alpha=0.9)
    
    # Draw corner markers
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2],
               c='red', s=80, marker='o', label='Corners')
    
    # Draw center
    c = rect['center_3d']
    ax.scatter([c[0]], [c[1]], [c[2]],
               c='yellow', s=150, marker='*', label='Center', zorder=5)
    
    # Draw normal vector from center
    n = rect['normal']
    arrow_len = max(rect['width'], rect['height']) * 0.4
    ax.plot([c[0], c[0] + n[0] * arrow_len],
            [c[1], c[1] + n[1] * arrow_len],
            [c[2], c[2] + n[2] * arrow_len],
            color='green', linewidth=2, alpha=0.9, label='Normal')
    
    # Draw up direction
    u = rect['up_direction']
    ax.plot([c[0], c[0] + u[0] * arrow_len],
            [c[1], c[1] + u[1] * arrow_len],
            [c[2], c[2] + u[2] * arrow_len],
            color='blue', linewidth=2, alpha=0.9, label='Up')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    return fig, ax


def visualize_rectangle_2d_projections(points, rect):
    """
    Visualize the detected rectangle in 2D projections.
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        Original point cloud
    rect : dict
        Rectangle detection result from detect_rectangle()
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    projections = [
        ('XY', 0, 1, 'X', 'Y'),
        ('XZ', 0, 2, 'X', 'Z'),
        ('YZ', 1, 2, 'Y', 'Z')
    ]
    
    corners = rect['corners_3d']
    center = rect['center_3d']
    inlier_pts = rect['inlier_points']
    
    for ax, (name, idx1, idx2, xlabel, ylabel) in zip(axes, projections):
        # Plot all points
        ax.scatter(points[:, idx1], points[:, idx2],
                   alpha=0.08, s=1, c='lightgray')
        
        # Plot inlier points
        ax.scatter(inlier_pts[:, idx1], inlier_pts[:, idx2],
                   alpha=0.3, s=3, c='cyan')
        
        # Draw rectangle outline
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([corners[i, idx1], corners[j, idx1]],
                    [corners[i, idx2], corners[j, idx2]],
                    color='red', linewidth=2.5, alpha=0.9)
        
        # Draw corners
        ax.scatter(corners[:, idx1], corners[:, idx2],
                   c='red', s=60, marker='o', zorder=5)
        
        # Draw center
        ax.scatter([center[idx1]], [center[idx2]],
                   c='yellow', s=120, marker='*', zorder=5)
        
        # Draw normal
        n = rect['normal']
        arrow_len = max(rect['width'], rect['height']) * 0.3
        ax.annotate('', xy=(center[idx1] + n[idx1] * arrow_len,
                            center[idx2] + n[idx2] * arrow_len),
                    xytext=(center[idx1], center[idx2]),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} Projection')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Rectangle edge'),
        Line2D([0], [0], color='green', linewidth=2, label='Normal'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow',
               markersize=12, label='Center'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    fig.suptitle(f"Rectangle: {rect['width']:.3f}m × {rect['height']:.3f}m  "
                 f"(aspect {rect['aspect_ratio']:.2f})", fontsize=12)
    plt.tight_layout()
    return fig, axes


def print_rectangle_summary(rect):
    """Print summary of the detected rectangle with key robot alignment values."""
    print(f"\n{'='*70}")
    print(f"RECTANGLE DETECTION SUMMARY")
    print(f"{'='*70}")
    
    if rect is None:
        print("  No rectangle detected.")
        print(f"{'='*70}")
        return
    
    c = rect['center_3d']
    
    print(f"  --- Key values for robot alignment ---")
    print(f"  Horizontal center (X): {rect['horizontal_center']:+.4f} m")
    print(f"    (negative = rectangle is LEFT of camera,")
    print(f"     positive = rectangle is RIGHT of camera)")
    print(f"  Depth to plane:        {rect['depth']:.4f} m")
    print(f"    (perpendicular distance from camera to the rectangle's plane)")
    print(f"  Distance to center:    {rect['distance_to_center']:.4f} m  (Euclidean)")
    print(f"")
    print(f"  --- Rectangle details ---")
    print(f"  Center 3D:       ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})")
    print(f"  Width:           {rect['width']:.4f} m")
    print(f"  Height:          {rect['height']:.4f} m")
    print(f"  Aspect ratio:    {rect['aspect_ratio']:.2f}")
    n = rect['normal']
    print(f"  Normal:          ({n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f})")
    print(f"  Inlier points:   {rect['n_inliers']}")
    print(f"{'='*70}")


# =============================================================================
# Line Detection (front-view climber)
# =============================================================================

def classify_line_orientation(direction_vector, vertical_axis=1, angle_threshold=15):
    """
    Classify a line as vertical, horizontal, or other based on its direction vector
    
    Parameters:
    -----------
    direction_vector : numpy array (3,)
        Direction vector of the line (slope)
    vertical_axis : int
        Index of the vertical axis (0=X, 1=Y, 2=Z). Default is Y (1).
    angle_threshold : float
        Maximum angle in degrees from the axis to be considered aligned
    
    Returns:
    --------
    orientation : str
        'vertical', 'horizontal', or 'other'
    angle_from_vertical : float
        Angle from vertical axis in degrees
    """
    # Normalize direction vector
    direction = np.array(direction_vector).flatten()
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return 'other', 90.0
    
    direction = direction / norm
    
    # Create unit vector for vertical axis
    vertical_unit = np.zeros(3)
    vertical_unit[vertical_axis] = 1.0
    
    # Calculate angle from vertical axis
    # Use absolute value of dot product since direction can point either way
    cos_angle = abs(np.dot(direction, vertical_unit))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_from_vertical = np.rad2deg(np.arccos(cos_angle))
    
    # Classify based on angle
    if angle_from_vertical <= angle_threshold:
        return 'vertical', angle_from_vertical
    elif angle_from_vertical >= (90 - angle_threshold):
        return 'horizontal', angle_from_vertical
    else:
        return 'other', angle_from_vertical


def detect_single_line(points, thresh=0.05, max_iterations=1000, 
                       vertical_axis=1, angle_threshold=15,
                       use_pyransac=False, confidence=0.999):
    """
    Detect a single line using RANSAC.
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        3D point cloud
    thresh : float
        Distance threshold for inliers
    max_iterations : int
        Maximum RANSAC iterations
    vertical_axis : int
        Index of vertical axis (0=X, 1=Y, 2=Z)
    angle_threshold : float
        Angle threshold for classification in degrees
    use_pyransac : bool
        If True, use pyransac's adaptive-iteration RANSAC (Line3D model)
        instead of pyransac3d's fixed-iteration Line.fit().
    confidence : float
        Confidence level for pyransac adaptive stopping (default: 0.999).
        Only used when use_pyransac=True.
    
    Returns:
    --------
    line_params : dict or None
        Dictionary with line parameters
    """
    if len(points) < 2:
        return None
    
    try:
        if use_pyransac and HAS_PYRANSAC:
            # --- pyransac backend: adaptive iteration ---
            point_list = [Point3D(p[0], p[1], p[2]) for p in points]
            params = RansacParams(samples=2, iterations=max_iterations,
                                  confidence=confidence, threshold=thresh)
            model = Line3D()
            inlier_objs = pyransac_find_inliers(point_list, model, params)

            if len(inlier_objs) < 2:
                return None

            # Map inlier objects back to indices
            inlier_set = set(id(p) for p in inlier_objs)
            inliers = [i for i, p in enumerate(point_list)
                        if id(p) in inlier_set]

            # Direction and point from the final model
            A = model.direction.copy()
            B = model.p0.copy()
        else:
            # --- pyransac3d backend: fixed iteration ---
            line = pyrsc.Line()
            A, B, inliers = line.fit(points, thresh=thresh,
                                     maxIteration=max_iterations)

            if len(inliers) < 2:
                return None

            A = np.array(A).flatten()
            B = np.array(B).flatten()
        
        # Get inlier points
        inlier_points = points[inliers]

        # Direction vector (normalized)
        direction = A / np.linalg.norm(A) if np.linalg.norm(A) > 1e-10 else A
        
        # Project inlier points onto line direction
        projections = np.dot(inlier_points - B, direction)
        t_min, t_max = projections.min(), projections.max()
        
        # Calculate endpoints
        point_start = B + t_min * direction
        point_end = B + t_max * direction
        
        # Classify orientation
        orientation, angle = classify_line_orientation(A, vertical_axis, angle_threshold)
        
        return {
            'direction': A,
            'point': B,
            'inliers': inliers,
            'n_inliers': len(inliers),
            'point_start': point_start,
            'point_end': point_end,
            'orientation': orientation,
            'angle_from_vertical': angle,
            'length': np.linalg.norm(point_end - point_start)
        }
        
    except Exception as e:
        print(f"Warning: Line fitting failed: {e}")
        return None


def detect_multiple_lines(points, max_lines=10, thresh=0.05, max_iterations=1000,
                          min_inliers=20, removal_thresh=None,
                          vertical_axis=1, angle_threshold=15,
                          use_pyransac=False, confidence=0.999):
    """
    Detect multiple lines using sequential RANSAC
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        3D point cloud
    max_lines : int
        Maximum number of lines to detect
    thresh : float
        Distance threshold for inliers
    max_iterations : int
        Maximum RANSAC iterations per line
    min_inliers : int
        Minimum inliers required for a valid line
    removal_thresh : float or None
        Distance threshold for removing points after detection.
        If None, uses thresh * 1.5
    vertical_axis : int
        Index of vertical axis (0=X, 1=Y, 2=Z)
    angle_threshold : float
        Angle threshold for classification in degrees
    use_pyransac : bool
        If True, use pyransac's adaptive-iteration RANSAC instead of
        pyransac3d's fixed-iteration RANSAC for each line.
    confidence : float
        Confidence level for pyransac adaptive stopping (default: 0.999).
        Only used when use_pyransac=True.
    
    Returns:
    --------
    lines : list of dict
        List of detected lines with parameters
    """
    if removal_thresh is None:
        removal_thresh = thresh * 1.5
    
    remaining_points = points.copy()
    remaining_indices = np.arange(len(points))
    detected_lines = []
    
    print(f"\nDetecting lines (max {max_lines})...")
    
    for line_num in range(max_lines):
        if len(remaining_points) < min_inliers:
            print(f"  Stopping: Not enough points remaining ({len(remaining_points)})")
            break
        
        # Detect one line
        line = detect_single_line(
            remaining_points,
            thresh=thresh,
            max_iterations=max_iterations,
            vertical_axis=vertical_axis,
            angle_threshold=angle_threshold,
            use_pyransac=use_pyransac,
            confidence=confidence
        )
        
        if line is None or line['n_inliers'] < min_inliers:
            print(f"  Stopping: No more valid lines found")
            break
        
        # Map inliers back to original indices
        original_inliers = remaining_indices[line['inliers']]
        line['original_inliers'] = original_inliers
        
        detected_lines.append(line)
        
        print(f"  Line {line_num + 1}: {line['orientation'].upper():10s} | "
              f"inliers: {line['n_inliers']:4d} | "
              f"length: {line['length']:.3f} | "
              f"angle from vertical: {line['angle_from_vertical']:.1f}°")
        
        # Remove inliers from remaining points
        # Calculate distance from each remaining point to the detected line
        direction = np.array(line['direction']).flatten()
        direction = direction / np.linalg.norm(direction)
        point_on_line = np.array(line['point']).flatten()
        
        # Distance from point to line: ||(p - p0) - ((p - p0) · d) * d||
        diff = remaining_points - point_on_line
        projections = np.dot(diff, direction).reshape(-1, 1) * direction
        distances = np.linalg.norm(diff - projections, axis=1)
        
        to_keep = distances >= removal_thresh
        remaining_points = remaining_points[to_keep]
        remaining_indices = remaining_indices[to_keep]
        
        if len(remaining_points) == 0:
            break
    
    return detected_lines


def remove_line_points(points, lines, removal_thresh=None, thresh=0.05):
    """
    Remove points belonging to detected lines from the point cloud.
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        3D point cloud
    lines : list of dict
        Detected lines whose inlier points should be removed
    removal_thresh : float or None
        Distance threshold for removing points around each line.
        If None, uses thresh * 1.5
    thresh : float
        Base distance threshold (used if removal_thresh is None)
    
    Returns:
    --------
    filtered_points : numpy array (M, 3)
        Point cloud with line points removed
    """
    if not lines:
        return points
    
    if removal_thresh is None:
        removal_thresh = thresh * 1.5
    
    mask_to_keep = np.ones(len(points), dtype=bool)
    
    for line in lines:
        direction = np.array(line['direction']).flatten()
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction = direction / norm
        point_on_line = np.array(line['point']).flatten()
        
        # Distance from each point to the line
        diff = points - point_on_line
        projections = np.dot(diff, direction).reshape(-1, 1) * direction
        distances = np.linalg.norm(diff - projections, axis=1)
        
        # Also restrict to the line's extent (between start and end)
        proj_scalars = np.dot(diff, direction)
        start_proj = np.dot(line['point_start'] - point_on_line, direction)
        end_proj = np.dot(line['point_end'] - point_on_line, direction)
        t_min = min(start_proj, end_proj) - removal_thresh
        t_max = max(start_proj, end_proj) + removal_thresh
        
        within_extent = (proj_scalars >= t_min) & (proj_scalars <= t_max)
        near_line = distances < removal_thresh
        
        mask_to_keep &= ~(near_line & within_extent)
    
    filtered_points = points[mask_to_keep]
    removed = len(points) - len(filtered_points)
    print(f"  Removed {removed} points belonging to {len(lines)} line(s) "
          f"({len(filtered_points)} remaining)")
    
    return filtered_points


def filter_lines_by_orientation(lines, orientation_filter=None):
    """
    Filter lines by orientation
    
    Parameters:
    -----------
    lines : list of dict
        Detected lines
    orientation_filter : str or None
        'vertical', 'horizontal', 'both', or None (no filter)
    
    Returns:
    --------
    filtered_lines : list of dict
        Filtered lines
    """
    if orientation_filter is None or orientation_filter == 'all':
        return lines
    
    if orientation_filter == 'both':
        return [l for l in lines if l['orientation'] in ['vertical', 'horizontal']]
    
    return [l for l in lines if l['orientation'] == orientation_filter]


def visualize_lines_3d(points, lines, title="3D Point Cloud with Detected Lines"):
    """
    Visualize 3D point cloud with detected lines
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        Original point cloud
    lines : list of dict
        Detected lines
    title : str
        Plot title
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points with low alpha
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               alpha=0.1, s=1, c='lightgray', label='All points')
    
    # Color scheme for different orientations
    colors = {
        'vertical': 'red',
        'horizontal': 'blue',
        'other': 'green'
    }
    
    # Plot each line
    for i, line in enumerate(lines):
        color = colors.get(line['orientation'], 'gray')
        
        # Plot inlier points
        inlier_points = points[line['original_inliers']]
        ax.scatter(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2],
                   alpha=0.5, s=5, c=color,
                   label=f"Line {i+1} ({line['orientation']})" if i < 10 else None)
        
        # Plot the line
        start = line['point_start']
        end = line['point_end']
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=color, linewidth=3, alpha=0.9)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    return fig, ax


def visualize_lines_2d_projections(points, lines):
    """
    Visualize lines in 2D projections (XY, XZ, YZ planes)
    
    Parameters:
    -----------
    points : numpy array (N, 3)
        Original point cloud
    lines : list of dict
        Detected lines
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    projections = [
        ('XY', 0, 1, 'X', 'Y'),
        ('XZ', 0, 2, 'X', 'Z'),
        ('YZ', 1, 2, 'Y', 'Z')
    ]
    
    colors = {
        'vertical': 'red',
        'horizontal': 'blue',
        'other': 'green'
    }
    
    for ax, (name, idx1, idx2, xlabel, ylabel) in zip(axes, projections):
        # Plot all points
        ax.scatter(points[:, idx1], points[:, idx2],
                   alpha=0.1, s=1, c='lightgray')
        
        # Plot each line
        for i, line in enumerate(lines):
            color = colors.get(line['orientation'], 'gray')
            
            # Plot inlier points
            inlier_points = points[line['original_inliers']]
            ax.scatter(inlier_points[:, idx1], inlier_points[:, idx2],
                       alpha=0.5, s=5, c=color)
            
            # Plot the line
            start = line['point_start']
            end = line['point_end']
            ax.plot([start[idx1], end[idx1]], [start[idx2], end[idx2]],
                    color=color, linewidth=2, alpha=0.9)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} Projection')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Vertical'),
        Line2D([0], [0], color='blue', linewidth=2, label='Horizontal'),
        Line2D([0], [0], color='green', linewidth=2, label='Other')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig, axes


def print_summary(lines, vertical_axis_name='Y'):
    """
    Print summary of detected lines
    
    Parameters:
    -----------
    lines : list of dict
        Detected lines
    vertical_axis_name : str
        Name of vertical axis for display
    """
    print(f"\n{'='*70}")
    print(f"DETECTION SUMMARY (Vertical axis: {vertical_axis_name})")
    print(f"{'='*70}")
    
    vertical_lines = [l for l in lines if l['orientation'] == 'vertical']
    horizontal_lines = [l for l in lines if l['orientation'] == 'horizontal']
    other_lines = [l for l in lines if l['orientation'] == 'other']
    
    print(f"Total lines detected: {len(lines)}")
    print(f"  - Vertical lines:   {len(vertical_lines)}")
    print(f"  - Horizontal lines: {len(horizontal_lines)}")
    print(f"  - Other lines:      {len(other_lines)}")
    
    if vertical_lines:
        print(f"\nVertical Lines:")
        for i, line in enumerate(vertical_lines, 1):
            print(f"  {i}. Length: {line['length']:.3f}, "
                  f"Inliers: {line['n_inliers']}, "
                  f"Angle from vertical: {line['angle_from_vertical']:.1f}°")
    
    if horizontal_lines:
        print(f"\nHorizontal Lines:")
        for i, line in enumerate(horizontal_lines, 1):
            print(f"  {i}. Length: {line['length']:.3f}, "
                  f"Inliers: {line['n_inliers']}, "
                  f"Angle from vertical: {line['angle_from_vertical']:.1f}°")
    
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Detect climber structure from a PCD file for FIRST Robotics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
    rectangle  - Detect the flat rectangular face of the climber (side view).
                 Finds the dominant plane and computes its oriented bounding box.
    lines      - Detect vertical/horizontal lines (front view with cylinders).

Examples:
    python line_detector.py scan.pcd --mode rectangle
    python line_detector.py scan.pcd --mode rectangle --plane-threshold 0.03
    python line_detector.py scan.pcd --mode lines --max-lines 20 --filter both
    python line_detector.py scan.pcd --mode rectangle --max-distance 2.0 --downsample 0.01
    python line_detector.py scan.pcd --no-distance-filter --no-remove-ground
        """
    )
    
    parser.add_argument('pcd_file', type=str,
                        help='Path to PCD file containing point cloud')
    parser.add_argument('--mode', type=str, default='rectangle',
                        choices=['rectangle', 'lines'],
                        help='Detection mode (default: rectangle)')
    
    # Preprocessing args
    parser.add_argument('--max-distance', type=float, default=1.25,
                        help='Remove points beyond this distance in meters from origin (default: 1.25)')
    parser.add_argument('--no-distance-filter', action='store_true',
                        help='Disable distance filtering')
    parser.add_argument('--no-remove-ground', action='store_true',
                        help='Disable ground plane removal')
    parser.add_argument('--ground-threshold', type=float, default=0.03,
                        help='Distance threshold for ground plane RANSAC (default: 0.03)')
    parser.add_argument('--downsample', type=float, default=None, metavar='VOXEL_SIZE',
                        help='Downsample using voxel grid with given size in meters (e.g. 0.01 for 1cm)')
    parser.add_argument('--vertical-axis', type=int, default=1, choices=[0, 1, 2],
                        help='Vertical axis index: 0=X, 1=Y, 2=Z (default: 1=Y)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip visualization')
    
    # Rectangle mode args
    rect_group = parser.add_argument_group('rectangle mode options')
    rect_group.add_argument('--plane-threshold', type=float, default=0.02,
                            help='Distance threshold for rectangle plane RANSAC (default: 0.02)')
    rect_group.add_argument('--plane-iterations', type=int, default=1000,
                            help='Max RANSAC iterations for plane fitting (default: 1000)')
    
    # Line mode args
    line_group = parser.add_argument_group('lines mode options')
    line_group.add_argument('--max-lines', type=int, default=10,
                            help='Maximum number of lines to detect (default: 10)')
    line_group.add_argument('--distance-threshold', type=float, default=0.05,
                            help='Distance threshold for line RANSAC inliers (default: 0.05)')
    line_group.add_argument('--max-iterations', type=int, default=1000,
                            help='Maximum RANSAC iterations per line (default: 1000)')
    line_group.add_argument('--min-inliers', type=int, default=20,
                            help='Minimum inliers required for a valid line (default: 20)')
    line_group.add_argument('--angle-threshold', type=float, default=15,
                            help='Max angle deviation for vertical/horizontal classification in degrees (default: 15)')
    line_group.add_argument('--filter', type=str, default='all',
                            choices=['all', 'vertical', 'horizontal', 'both'],
                            help='Filter output by line orientation (default: all)')
    
    # pyransac backend args
    ransac_group = parser.add_argument_group('pyransac backend options')
    ransac_group.add_argument('--use-pyransac', action='store_true',
                              help='Use pyransac adaptive-iteration RANSAC instead of '
                                   'pyransac3d fixed-iteration RANSAC.  This can be '
                                   'faster when the target structure has a high inlier '
                                   'ratio because it stops early once the confidence '
                                   'threshold is met.')
    ransac_group.add_argument('--confidence', type=float, default=0.999,
                              help='Confidence level for adaptive RANSAC stopping '
                                   '(0-1, default: 0.999). Higher values run more '
                                   'iterations. Only used with --use-pyransac.')

    args = parser.parse_args()

    # Validate pyransac availability
    if args.use_pyransac and not HAS_PYRANSAC:
        print("Error: --use-pyransac requested but pyransac is not installed.")
        print("       Install it with: pip install pyransac")
        sys.exit(1)
    
    # Load point cloud
    print(f"Loading PCD file: {args.pcd_file}")
    points = load_pcd_file(args.pcd_file)
    
    # Print raw statistics
    print(f"\nRaw point cloud statistics:")
    print(f"  Number of points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    axis_names = ['X', 'Y', 'Z']
    vertical_axis_name = axis_names[args.vertical_axis]
    print(f"  Vertical axis: {vertical_axis_name}")
    
    # --- Preprocessing ---
    print(f"\nPreprocessing...")
    
    # 1. Distance filter: remove points beyond max_distance
    if not args.no_distance_filter:
        points = filter_by_distance(points, max_distance=args.max_distance)
    
    # 2. Downsample (optional)
    if args.downsample is not None:
        points = downsample_points(points, voxel_size=args.downsample)
    
    # 3. Ground plane removal
    if not args.no_remove_ground:
        points, ground_plane = remove_ground_plane(
            points,
            distance_threshold=args.ground_threshold,
            vertical_axis=args.vertical_axis
        )
    
    # Print post-processing statistics
    print(f"\nProcessed point cloud statistics:")
    print(f"  Number of points: {len(points)}")
    if len(points) > 0:
        print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # =========================================================================
    # Detection
    # =========================================================================
    if args.mode == 'rectangle':
        # --- Rectangle detection (side-view climber) ---
        print(f"\n{'='*70}")
        print(f"RECTANGLE DETECTION MODE (side-view climber)")
        print(f"{'='*70}")
        
        rect = detect_rectangle(
            points,
            plane_thresh=args.plane_threshold,
            plane_iterations=args.plane_iterations,
            min_inliers=args.min_inliers if hasattr(args, 'min_inliers') else 50,
            vertical_axis=args.vertical_axis,
            use_pyransac=args.use_pyransac,
            confidence=args.confidence
        )
        
        print_rectangle_summary(rect)
        
        if not args.no_plot and rect is not None:
            visualize_rectangle_3d(points, rect,
                                   f"Climber Rectangle Detection")
            visualize_rectangle_2d_projections(points, rect)
            plt.show()
        elif rect is None:
            print("\nNo rectangle detected matching the criteria.")
        
        print("\nDetection complete!")
        return rect
    
    else:
        # --- Line detection (front-view climber) ---
        print(f"\n{'='*70}")
        print(f"LINE DETECTION MODE (front-view climber)")
        print(f"{'='*70}")
        
        # Pass 1: Detect vertical lines
        print(f"\n--- Pass 1: Detecting vertical lines ---")
        all_lines = detect_multiple_lines(
            points,
            max_lines=args.max_lines,
            thresh=args.distance_threshold,
            max_iterations=args.max_iterations,
            min_inliers=args.min_inliers,
            vertical_axis=args.vertical_axis,
            angle_threshold=args.angle_threshold,
            use_pyransac=args.use_pyransac,
            confidence=args.confidence
        )
        
        vertical_lines = [l for l in all_lines if l['orientation'] == 'vertical']
        
        print(f"\n  Found {len(vertical_lines)} vertical line(s) in pass 1")
        
        # Remove vertical line points from the cloud
        if vertical_lines:
            print(f"\nRemoving vertical line points from point cloud...")
            points_no_vertical = remove_line_points(
                points, vertical_lines,
                thresh=args.distance_threshold
            )
        else:
            points_no_vertical = points
        
        # Pass 2: Detect horizontal (and other) lines on the cleaned cloud
        print(f"\n--- Pass 2: Detecting horizontal lines (vertical lines removed) ---")
        remaining_lines = detect_multiple_lines(
            points_no_vertical,
            max_lines=args.max_lines,
            thresh=args.distance_threshold,
            max_iterations=args.max_iterations,
            min_inliers=args.min_inliers,
            vertical_axis=args.vertical_axis,
            angle_threshold=args.angle_threshold,
            use_pyransac=args.use_pyransac,
            confidence=args.confidence
        )
        
        horizontal_lines = [l for l in remaining_lines if l['orientation'] == 'horizontal']
        other_lines = [l for l in remaining_lines if l['orientation'] == 'other']
        
        print(f"\n  Found {len(horizontal_lines)} horizontal line(s) and "
              f"{len(other_lines)} other line(s) in pass 2")
        
        # Combine all detected lines
        all_detected_lines = vertical_lines + remaining_lines
        
        # Filter lines if requested
        filtered_lines = filter_lines_by_orientation(all_detected_lines, args.filter)
        
        # Print summary
        print_summary(filtered_lines, vertical_axis_name)
        
        # Visualize
        if not args.no_plot and len(filtered_lines) > 0:
            visualize_lines_3d(points, filtered_lines, 
                              f"3D Point Cloud - {len(filtered_lines)} Lines Detected")
            visualize_lines_2d_projections(points, filtered_lines)
            plt.show()
        elif len(filtered_lines) == 0:
            print("\nNo lines detected matching the criteria.")
        
        print("\nDetection complete!")
        return all_detected_lines, filtered_lines


if __name__ == "__main__":
    main()
