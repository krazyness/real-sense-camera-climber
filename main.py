"""
Climber Rectangle Detector — FIRST Robotics

Runs continuously on a Raspberry Pi. Each loop iteration reads a .pcd file,
detects the flat rectangular face of the climber, and publishes to NetworkTables:
  - /climber/detected           (bool)
  - /climber/horizontal_offset  (double, meters)
  - /climber/depth              (double, meters)
"""

import sys
import time
import numpy as np
import pyransac3d as pyrsc
from scipy import ndimage
from networktables import NetworkTables


# ── PCD I/O ──────────────────────────────────────────────────────────────────

def read_pcd(path):
    """
    Read a binary PCD file with FIELDS x y z (float32).
    Returns an (N, 3) float32 numpy array.
    """
    with open(path, "rb") as f:
        num_points = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("POINTS"):
                num_points = int(line.split()[1])
            if line.startswith("DATA"):
                break
        if num_points == 0:
            return np.empty((0, 3), dtype=np.float32)
        raw = f.read(num_points * 3 * 4)  # 3 floats × 4 bytes each
    pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)
    return pts


# ── Clustering ───────────────────────────────────────────────────────────────

def cluster_largest(points, voxel_size):
    """
    Voxel-grid connected-component clustering.
    Discretises points into a 3D grid, runs scipy.ndimage.label for fast
    connected components, then returns only the points in the largest cluster.
    """
    if len(points) == 0:
        return points

    mins = points.min(axis=0)
    # Map each point to a voxel index
    idx = ((points - mins) / voxel_size).astype(np.int32)
    # Build a dense occupancy grid
    grid_shape = idx.max(axis=0) + 1
    # Safety: if grid would be absurdly large, fall back to returning all points
    if np.prod(grid_shape) > 5_000_000:
        return points
    grid = np.zeros(grid_shape, dtype=np.int32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    # Connected components (26-connectivity)
    struct = ndimage.generate_binary_structure(3, 3)
    labelled, num_features = ndimage.label(grid, structure=struct)
    if num_features == 0:
        return points

    # Find largest component
    point_labels = labelled[idx[:, 0], idx[:, 1], idx[:, 2]]
    unique, counts = np.unique(point_labels[point_labels > 0], return_counts=True)
    if len(unique) == 0:
        return points
    best = unique[counts.argmax()]
    return points[point_labels == best]


def detect_climber(pcd_path, max_distance=1.25, plane_thresh=0.02,
                   ground_thresh=0.03, min_inliers=50):
    """
    Full pipeline: load PCD → filter → remove ground → find rectangle plane
    → cluster largest → compute horizontal offset and depth.

    Returns (horizontal_offset, depth) or None if detection fails.
    """
    # ── Load ──
    pts = read_pcd(pcd_path)
    if len(pts) == 0:
        return None

    # ── Distance filter ──
    pts = pts[np.linalg.norm(pts, axis=1) <= max_distance]
    if len(pts) < min_inliers:
        return None

    # ── Ground removal ──
    plane = pyrsc.Plane()
    eq, inliers = plane.fit(pts, thresh=ground_thresh, maxIteration=1000)
    normal = np.array(eq[:3])
    normal /= np.linalg.norm(normal)
    # Remove only if the plane is roughly horizontal (normal ≈ Y-axis)
    if abs(normal[1]) > np.cos(np.radians(30)):          # within 30° of vertical
        mask = np.ones(len(pts), dtype=bool)
        mask[inliers] = False
        pts = pts[mask]
    if len(pts) < min_inliers:
        return None

    # ── RANSAC plane (the rectangle face) ──
    plane = pyrsc.Plane()
    eq, inliers = plane.fit(pts, thresh=plane_thresh, maxIteration=1000)
    if len(inliers) < min_inliers:
        return None

    inlier_pts = pts[inliers]

    # ── Cluster — keep only the largest connected component ──
    inlier_pts = cluster_largest(inlier_pts, voxel_size=plane_thresh * 3)

    # ── Results ──
    center = inlier_pts.mean(axis=0)
    horizontal_offset = float(center[0])

    n_len = np.linalg.norm(eq[:3])
    depth = float(abs(eq[3]) / n_len) if n_len > 1e-10 else float(abs(center[2]))

    return horizontal_offset, depth


# ── CLI ──
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python main.py <pcd_file> [team_number|roborio_ip]")
        sys.exit(1)

    # NetworkTables setup — connect as a client to the roboRIO
    server = sys.argv[2] if len(sys.argv) > 2 else "10.0.0.2"
    if server.isdigit():
        team = int(server)
        server = f"10.{team // 100}.{team % 100}.2"

    NetworkTables.initialize(server=server)
    table = NetworkTables.getTable("real-sense-camera-climber")

    print(f"NetworkTables connecting to {server} ...")
    print(f"Publishing to /real-sense-camera-climber/  |  Reading PCD: {path}")

    try:
        while True:
            result = detect_climber(path)
            if result is None:
                table.putBoolean("detected", False)
                table.putNumber("horizontal_offset", 0.0)
                table.putNumber("depth", 0.0)
            else:
                h, d = result
                table.putBoolean("detected", True)
                table.putNumber("horizontal_offset", h)
                table.putNumber("depth", d)
            NetworkTables.flush()
            time.sleep(0.02)   # ~50 Hz cap; detection itself is the bottleneck
    except KeyboardInterrupt:
        print("\nStopping.")
        NetworkTables.shutdown()
