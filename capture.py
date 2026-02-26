"""
RealSense → PCD capture

Continuously captures depth frames from an Intel RealSense camera and
writes the point cloud to a .pcd file that main.py reads each loop.

Uses the rs_python C++ pybind11 module (RSCam) instead of pyrealsense2.

Usage:
    python capture.py                 # writes to frame.pcd (default)
    python capture.py --output scan.pcd
"""

import argparse
import os
import tempfile
import time
import numpy as np
import rs_python


def write_pcd(path, points):
    """
    Write an (N, 3) float32 array to a binary PCD file.
    Uses atomic write (temp file + rename) so readers never see a partial file.
    """
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".pcd")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(points.astype(np.float32).tobytes())
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def capture_frame(cam, K):
    """
    Grab one depth frame via RSCam, deproject to 3D, return (N, 3) float32 array.
    Returns an empty array if no valid points.
    """
    depth_raw = cam.GetDepth()  # uint16 numpy array (h, w), values in mm

    h, w = depth_raw.shape
    depth_meters = depth_raw.astype(np.float32) / 1000.0  # mm → m

    # Intrinsics from K matrix (3×3 list-of-lists)
    fx = K[0][0]
    fy = K[1][1]
    ppx = K[0][2]
    ppy = K[1][2]

    # Build (u, v) grid
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    z = depth_meters.flatten()

    # Filter out zero-depth (invalid) pixels
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Deproject to 3D (camera coordinates)
    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy

    return np.stack((x, y, z), axis=-1)


def main():
    parser = argparse.ArgumentParser(
        description="Capture RealSense depth → PCD file (runs continuously)."
    )
    parser.add_argument(
        "--output", "-o", default="frame.pcd",
        help="Path for the output .pcd file (default: frame.pcd)"
    )
    args = parser.parse_args()

    # RSCam handles pipeline init + auto-exposure internally
    cam = rs_python.RSCam(enable_imu=False)
    K = cam.GetK(depth=True)  # depth intrinsic matrix (3×3)

    print("RealSense camera initialised")
    print(f"Writing point clouds to: {args.output}")

    try:
        while True:
            points = capture_frame(cam, K)
            if len(points) > 0:
                write_pcd(args.output, points)
            time.sleep(0.005)  # small yield; camera FPS is the real limiter
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cam.Stop()


if __name__ == "__main__":
    main()
