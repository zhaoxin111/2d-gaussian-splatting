#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project COLMAP points3D.txt onto images.

Inputs:
    cameras.txt
    images.txt
    points3D.txt
    original image folder

Pose convention:
    COLMAP images.txt stores world-to-camera pose:
        X_cam = R_cw @ X_world + t_cw
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# -----------------------------
# COLMAP parsing
# -----------------------------

def read_cameras_txt(path):
    """
    Parse COLMAP cameras.txt

    Format:
        CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    """
    cameras = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=np.float64)

            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }

    return cameras


def read_images_txt(path):
    """
    Parse COLMAP images.txt

    Each image has two lines:
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        POINTS2D[] as X Y POINT3D_ID
    """
    images = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    i = 0
    while i < len(lines):
        header = lines[i]
        parts = header.split()

        image_id = int(parts[0])

        qvec = np.array(
            [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
            dtype=np.float64,
        )

        tvec = np.array(
            [float(parts[5]), float(parts[6]), float(parts[7])],
            dtype=np.float64,
        )

        camera_id = int(parts[8])

        # Image name may theoretically contain spaces.
        image_name = " ".join(parts[9:])

        images.append({
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": image_name,
        })

        # Skip 2D observation line.
        i += 2

    return images


def read_points3D_txt(path):
    """
    Parse COLMAP points3D.txt

    Format:
        POINT3D_ID X Y Z R G B ERROR TRACK[]
    """
    xyz_list = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            xyz_list.append([x, y, z])

    if len(xyz_list) == 0:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(xyz_list, dtype=np.float64)


# -----------------------------
# Geometry
# -----------------------------

def qvec_to_rotmat(qvec):
    """
    COLMAP quaternion convention:
        qvec = [qw, qx, qy, qz]

    Returns:
        R_cw
    """
    qw, qx, qy, qz = qvec

    return np.array([
        [
            1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
            2.0 * qx * qy - 2.0 * qz * qw,
            2.0 * qx * qz + 2.0 * qy * qw,
        ],
        [
            2.0 * qx * qy + 2.0 * qz * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
            2.0 * qy * qz - 2.0 * qx * qw,
        ],
        [
            2.0 * qx * qz - 2.0 * qy * qw,
            2.0 * qy * qz + 2.0 * qx * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
        ],
    ], dtype=np.float64)


def project_points(camera, points_cam):
    """
    Project camera-coordinate 3D points to pixels.

    points_cam:
        shape [N, 3]

    Returns:
        uv: shape [N, 2]
    """
    model = camera["model"].upper()
    params = camera["params"]

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    x = X / Z
    y = Y / Z

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        u = f * x + cx
        v = f * y + cy

    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        u = fx * x + cx
        v = fy * y + cy

    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[:4]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2
        xd = x * radial
        yd = y * radial
        u = f * xd + cx
        v = f * yd + cy

    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        r2 = x * x + y * y
        r4 = r2 * r2
        radial = 1.0 + k1 * r2 + k2 * r4
        xd = x * radial
        yd = y * radial
        u = f * xd + cx
        v = f * yd + cy

    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        r2 = x * x + y * y
        r4 = r2 * r2

        radial = 1.0 + k1 * r2 + k2 * r4

        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        u = fx * xd + cx
        v = fy * yd + cy

    elif model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params[:12]

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        radial_den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        radial = radial_num / radial_den

        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        u = fx * xd + cx
        v = fy * yd + cy

    elif model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = params[:8]

        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)

        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        theta_d = theta * (
            1.0
            + k1 * theta2
            + k2 * theta4
            + k3 * theta6
            + k4 * theta8
        )

        scale = np.ones_like(r)
        valid = r > 1e-12
        scale[valid] = theta_d[valid] / r[valid]

        xd = x * scale
        yd = y * scale

        u = fx * xd + cx
        v = fy * yd + cy

    else:
        raise NotImplementedError(
            f"Unsupported camera model: {model}. "
            f"You can undistort images with COLMAP first, then use PINHOLE cameras."
        )

    return np.stack([u, v], axis=1)


# -----------------------------
# Visualization
# -----------------------------

def distance_to_colors(distances, colormap_name="turbo", min_dist=None, max_dist=None):
    """
    Convert distance values to OpenCV BGR colors.
    """
    distances = np.asarray(distances, dtype=np.float64)

    if min_dist is None:
        min_dist = np.percentile(distances, 2)

    if max_dist is None:
        max_dist = np.percentile(distances, 98)

    if max_dist <= min_dist:
        max_dist = min_dist + 1e-6

    values = (distances - min_dist) / (max_dist - min_dist)
    values = np.clip(values, 0.0, 1.0)
    values = (values * 255).astype(np.uint8)

    if colormap_name.lower() == "turbo" and hasattr(cv2, "COLORMAP_TURBO"):
        cmap = cv2.COLORMAP_TURBO
    elif colormap_name.lower() == "jet":
        cmap = cv2.COLORMAP_JET
    elif colormap_name.lower() == "hot":
        cmap = cv2.COLORMAP_HOT
    else:
        cmap = cv2.COLORMAP_JET

    colors = cv2.applyColorMap(values.reshape(-1, 1), cmap)
    colors = colors.reshape(-1, 3)

    return colors


def draw_projected_points(
    image,
    uv,
    distances,
    radius=2,
    alpha=1.0,
    colormap_name="turbo",
    min_dist=None,
    max_dist=None,
):
    """
    Draw projected points on image.

    Far points are drawn first, near points are drawn later.
    This gives a weak occlusion-like visualization.
    """
    output = image.copy()
    overlay = image.copy()

    colors = distance_to_colors(
        distances,
        colormap_name=colormap_name,
        min_dist=min_dist,
        max_dist=max_dist,
    )

    # Draw far-to-near.
    order = np.argsort(-distances)

    for idx in order:
        u, v = uv[idx]
        color = colors[idx].tolist()

        cv2.circle(
            overlay,
            center=(int(round(u)), int(round(v))),
            radius=radius,
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    if alpha >= 1.0:
        output = overlay
    else:
        output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)

    return output


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sparse_dir",
        type=str,
        required=True,
        help="Directory containing cameras.txt, images.txt, points3D.txt",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing original images",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Projected point radius in pixels",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Overlay alpha. 1.0 means directly draw on image.",
    )

    parser.add_argument(
        "--distance_type",
        type=str,
        default="euclidean",
        choices=["euclidean", "depth"],
        help="Color by euclidean camera distance or camera depth Z",
    )

    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        choices=["turbo", "jet", "hot"],
        help="OpenCV colormap",
    )

    parser.add_argument(
        "--min_dist",
        type=float,
        default=None,
        help="Manual minimum distance for color normalization",
    )

    parser.add_argument(
        "--max_dist",
        type=float,
        default=None,
        help="Manual maximum distance for color normalization",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=-1,
        help="Only process first N images. -1 means all images.",
    )

    parser.add_argument(
        "--sample_points",
        type=int,
        default=-1,
        help="Randomly sample N 3D points. -1 means use all points.",
    )
    parser.add_argument(
    "--min_euclidean_dist",
    type=float,
    default=None,
    help="Only project points whose Euclidean distance to current camera is >= this value",
    )

    parser.add_argument(
        "--max_euclidean_dist",
        type=float,
        default=None,
        help="Only project points whose Euclidean distance to current camera is <= this value",
    )

    args = parser.parse_args()

    sparse_dir = Path(args.sparse_dir)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras_path = sparse_dir / "cameras.txt"
    images_path = sparse_dir / "images.txt"
    points_path = sparse_dir / "points3D.txt"

    cameras = read_cameras_txt(cameras_path)
    images = read_images_txt(images_path)
    points_world = read_points3D_txt(points_path)

    print(f"[INFO] cameras: {len(cameras)}")
    print(f"[INFO] images: {len(images)}")
    print(f"[INFO] points3D: {len(points_world)}")

    if args.sample_points > 0 and args.sample_points < len(points_world):
        rng = np.random.default_rng(0)
        indices = rng.choice(len(points_world), size=args.sample_points, replace=False)
        points_world = points_world[indices]
        print(f"[INFO] sampled points3D: {len(points_world)}")

    if args.max_images > 0:
        images = images[:args.max_images]

    for idx, img_info in enumerate(images):
        image_name = img_info["name"]
        image_path = image_dir / image_name

        if not image_path.exists():
            print(f"[WARN] image not found: {image_path}")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] failed to read image: {image_path}")
            continue

        h, w = image.shape[:2]

        camera = cameras[img_info["camera_id"]]

        R_cw = qvec_to_rotmat(img_info["qvec"])
        t_cw = img_info["tvec"].reshape(3, 1)

        # World to camera:
        # points_cam = R_cw @ points_world + t_cw
        points_cam = (R_cw @ points_world.T + t_cw).T

        # Keep points in front of camera.
        z = points_cam[:, 2]
        valid_front = z > 1e-6

        points_cam_valid = points_cam[valid_front]
        if len(points_cam_valid) == 0:
            print(f"[WARN] no front points for image: {image_name}")
            continue

        # ------------------------------------------------------------
        # Filter by Euclidean distance to current camera center.
        # Since points_cam is already in camera coordinate system,
        # Euclidean distance = sqrt(X_cam^2 + Y_cam^2 + Z_cam^2)
        # ------------------------------------------------------------
        euclidean_dist = np.linalg.norm(points_cam_valid, axis=1)

        valid_dist = np.ones(len(points_cam_valid), dtype=bool)

        if args.min_euclidean_dist is not None:
            valid_dist &= euclidean_dist >= args.min_euclidean_dist

        if args.max_euclidean_dist is not None:
            valid_dist &= euclidean_dist <= args.max_euclidean_dist

        points_cam_valid = points_cam_valid[valid_dist]

        if len(points_cam_valid) == 0:
            print(
                f"[WARN] no points within euclidean distance range for image: {image_name}, "
                f"range=[{args.min_euclidean_dist}, {args.max_euclidean_dist}]"
            )
            continue

        uv = project_points(camera, points_cam_valid)

        u = uv[:, 0]
        v = uv[:, 1]

        valid_img = (
            (u >= 0)
            & (u < w)
            & (v >= 0)
            & (v < h)
            & np.isfinite(u)
            & np.isfinite(v)
        )

        uv = uv[valid_img]
        points_cam_visible = points_cam_valid[valid_img]

        if len(uv) == 0:
            print(f"[WARN] no projected points inside image: {image_name}")
            continue

        if args.distance_type == "depth":
            distances = points_cam_visible[:, 2]
        else:
            distances = np.linalg.norm(points_cam_visible, axis=1)

        vis = draw_projected_points(
            image=image,
            uv=uv,
            distances=distances,
            radius=args.radius,
            alpha=args.alpha,
            colormap_name=args.colormap,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
        )

        out_path = out_dir / image_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_path), vis)

        print(
            f"[{idx + 1:04d}/{len(images):04d}] "
            f"{image_name} | projected points: {len(uv)} | saved: {out_path}"
        )

    print("[DONE]")


if __name__ == "__main__":
    main()