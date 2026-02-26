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


def write_pcd(path, points, tmp_path):
    """
    Write an (N, 3) float32 array to a binary PCD file.
    Uses atomic write (write to tmp then rename) so readers never see partial data.
    """
    n = len(points)
    header = (
        f"# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\n"
        f"FIELDS x y z\n"
        f"SIZE 4 4 4\n"
        f"TYPE F F F\n"
        f"COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA binary\n"
    )
    with open(tmp_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.tobytes())      # already float32, no copy needed
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(
        description="Capture RealSense depth → PCD file (runs continuously)."
    )
    parser.add_argument(
        "--output", "-o", default="frame.pcd",
        help="Path for the output .pcd file (default: frame.pcd)"
    )
    args = parser.parse_args()

    # ── Camera init ──
    cam = rs_python.RSCam(enable_imu=False)
    K = cam.GetK(depth=True)

    fx = np.float32(K[0][0])
    fy = np.float32(K[1][1])
    ppx = np.float32(K[0][2])
    ppy = np.float32(K[1][2])

    # Grab one frame to learn resolution, then pre-compute the pixel grid once
    first = cam.GetDepth()
    h, w = first.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)
    u_flat = (u.flatten() - ppx) / fx   # pre-baked (u-ppx)/fx
    v_flat = (v.flatten() - ppy) / fy   # pre-baked (v-ppy)/fy

    # Pre-allocate output buffer (worst case: every pixel valid)
    buf = np.empty((w * h, 3), dtype=np.float32)

    # Temp file path for atomic writes (reuse, don't mkstemp each frame)
    dir_name = os.path.dirname(os.path.abspath(args.output))
    tmp_path = os.path.join(dir_name, ".capture_tmp.pcd")

    print(f"RealSense {w}×{h} ready")
    print(f"Writing point clouds to: {args.output}")

    frame_count = 0
    t0 = time.monotonic()
    try:
        while True:
            # ── Grab depth ──
            depth_raw = cam.GetDepth()
            z = depth_raw.ravel().astype(np.float32)
            z *= 0.001  # mm → m  (in-place)

            # ── Filter invalid pixels & deproject ──
            valid = z > 0
            zv = z[valid]
            n = len(zv)
            if n > 0:
                buf[:n, 0] = u_flat[valid] * zv   # x
                buf[:n, 1] = v_flat[valid] * zv   # y
                buf[:n, 2] = zv                    # z
                write_pcd(args.output, buf[:n], tmp_path)

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - t0
                fps = frame_count / elapsed
                print(f"  {frame_count} frames  |  {fps:.1f} fps  |  {n} pts")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cam.Stop()
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
