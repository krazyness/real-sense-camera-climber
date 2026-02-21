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
import open3d as o3d
from networktables import NetworkTables


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
