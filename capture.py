"""
RealSense → PCD capture

Continuously captures depth frames from an Intel RealSense camera and
writes the point cloud to a .pcd file that main.py reads each loop.

Usage:
    python capture.py                 # writes to frame.pcd (default)
    python capture.py --output scan.pcd
    python capture.py --width 640 --height 480 --fps 30
"""

import argparse
import time
import numpy as np
import pyrealsense2 as rs
import open3d as o3d


def create_pipeline(width=640, height=480, fps=30):
    """Configure and start the RealSense depth pipeline."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)

    # Let auto-exposure settle
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, profile


def capture_frame(pipeline, profile):
    """
    Grab one depth frame, convert to an Open3D point cloud, return it.
    Returns an open3d.geometry.PointCloud (empty if capture failed).
    """
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return o3d.geometry.PointCloud()

    # Intrinsics needed to deproject pixels → 3D
    intrinsics = (
        depth_frame.profile.as_video_stream_profile().intrinsics
    )

    w, h = intrinsics.width, intrinsics.height
    depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_meters = depth_image * depth_scale

    # Build (u, v) grid
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    u = u.flatten().astype(np.float32)
    v = v.flatten().astype(np.float32)
    z = depth_meters.flatten()

    # Filter out zero-depth (invalid) pixels
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    # Deproject to 3D (camera coordinates)
    fx = intrinsics.fx
    fy = intrinsics.fy
    ppx = intrinsics.ppx
    ppy = intrinsics.ppy

    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy

    points = np.stack((x, y, z), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def main():
    parser = argparse.ArgumentParser(
        description="Capture RealSense depth → PCD file (runs continuously)."
    )
    parser.add_argument(
        "--output", "-o", default="frame.pcd",
        help="Path for the output .pcd file (default: frame.pcd)"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    pipeline, profile = create_pipeline(args.width, args.height, args.fps)
    print(f"RealSense streaming {args.width}×{args.height} @ {args.fps} fps")
    print(f"Writing point clouds to: {args.output}")

    try:
        while True:
            pcd = capture_frame(pipeline, profile)
            if len(pcd.points) > 0:
                o3d.io.write_point_cloud(args.output, pcd, write_ascii=False)
            time.sleep(0.005)  # small yield; camera FPS is the real limiter
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
