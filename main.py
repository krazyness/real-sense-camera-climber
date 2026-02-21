"""
Climber Rectangle Detector — FIRST Robotics

Detects the flat rectangular face of the climber from a RealSense .pcd file.
Returns two values for robot alignment:
  - horizontal_offset: X distance from camera center to rectangle center (m)
  - depth:             perpendicular distance from camera to rectangle plane (m)
"""

import sys
import numpy as np
import pyransac3d as pyrsc
import open3d as o3d


def detect_climber(pcd_path, max_distance=1.25, plane_thresh=0.02,
                   ground_thresh=0.03, min_inliers=50):
    """
    Full pipeline: load PCD → filter → remove ground → find rectangle plane
    → cluster largest → compute horizontal offset and depth.

    Returns (horizontal_offset, depth) or None if detection fails.
    """
    # ── Load ──
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
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

    # ── DBSCAN — keep only the largest cluster ──
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(inlier_pts)
    labels = np.array(pc.cluster_dbscan(eps=plane_thresh * 3,
                                        min_points=10,
                                        print_progress=False))
    if labels.max() >= 0:
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        inlier_pts = inlier_pts[labels == unique[counts.argmax()]]

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
        print("Usage: python main.py <pcd_file>")
        sys.exit(1)

    result = detect_climber(path)
    if result is None:
        print("No rectangle detected.")
        sys.exit(1)

    h, d = result
    print(f"horizontal_offset: {h:+.4f} m")
    print(f"depth:             {d:.4f} m")
