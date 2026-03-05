"""
RealSense PCD snapshot tool  (Raspberry Pi)

Press SPACEBAR to save a point cloud snapshot. Press q to quit.
Camera streams continuously in the background to stay warm.

Usage:
    python capture.py                        # saves snap_001.pcd, snap_002.pcd, ...
    python capture.py --prefix myscan        # saves myscan_001.pcd, myscan_002.pcd, ...
"""

import argparse
import os
import select
import sys
import termios
import tty
import numpy as np
import rs_python


def write_pcd(path, points):
    """Write an (N, 3) float32 array to a binary PCD file."""
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
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.tobytes())


def main():
    parser = argparse.ArgumentParser(description="RealSense PCD snapshots.")
    parser.add_argument(
        "--prefix", "-p", default="snap",
        help="Filename prefix (default: snap -> snap_001.pcd, snap_002.pcd, ...)"
    )
    args = parser.parse_args()

    # ── Camera init ──
    cam = rs_python.RSCam(enable_imu=False)
    K = cam.GetK(depth=True)

    fx = np.float32(K[0][0])
    fy = np.float32(K[1][1])
    ppx = np.float32(K[0][2])
    ppy = np.float32(K[1][2])

    # First frame to learn resolution; pre-compute pixel grid once
    first = cam.GetDepth()
    h, w = first.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)
    u_flat = (u.flatten() - ppx) / fx
    v_flat = (v.flatten() - ppy) / fy
    buf = np.empty((w * h, 3), dtype=np.float32)

    def grab():
        """Grab one depth frame, deproject, return buf[:n] view."""
        depth_raw = cam.GetDepth()
        z = depth_raw.ravel().astype(np.float32)
        z *= 0.001
        valid = z > 0
        zv = z[valid]
        n = len(zv)
        if n == 0:
            return buf[:0]
        buf[:n, 0] = u_flat[valid] * zv
        buf[:n, 1] = v_flat[valid] * zv
        buf[:n, 2] = zv
        return buf[:n]

    print(f"RealSense {w}x{h} ready")
    print("Press SPACEBAR to save a snapshot, q to quit.")

    # Put terminal in cbreak mode for single-keypress detection
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    snap_num = 0
    try:
        while True:
            grab()  # keep camera streaming (discards frame)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == " ":
                    points = grab().copy()  # copy since buf is reused
                    if len(points) > 0:
                        snap_num += 1
                        path = f"{args.prefix}_{snap_num:03d}.pcd"
                        write_pcd(path, points)
                        print(f"  [{snap_num}] saved {path}  ({len(points)} pts)")
                    else:
                        print("  empty frame, try again")
                elif key == "q":
                    break
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        cam.Stop()


if __name__ == "__main__":
    main()
