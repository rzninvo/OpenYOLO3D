#!/usr/bin/env python3
"""
Convert NavVis dataset format to OpenYOLO3D format
"""

import os
import shutil
import numpy as np
from pathlib import Path
import argparse


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix"""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def pose_to_matrix(qw, qx, qy, qz, tx, ty, tz):
    """Convert quaternion + translation to 4x4 transformation matrix"""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    T[:3, 3] = [tx, ty, tz]
    return T


def parse_trajectories(traj_file, camera_id):
    """Parse trajectories.txt and extract poses for specific camera"""
    poses = []
    with open(traj_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(', ')
            if len(parts) < 9:
                continue

            timestamp = parts[0]
            cam_id = parts[1]

            if cam_id == camera_id:
                qw, qx, qy, qz = map(float, parts[2:6])
                tx, ty, tz = map(float, parts[6:9])
                poses.append({
                    'timestamp': timestamp,
                    'matrix': pose_to_matrix(qw, qx, qy, qz, tx, ty, tz)
                })

    return poses


def parse_intrinsics(sensors_file, camera_id):
    """Parse sensors.txt and extract intrinsics for specific camera"""
    with open(sensors_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(', ')
            if len(parts) < 8:
                continue

            cam_id = parts[0]
            if cam_id == camera_id:
                # Format: sensor_id, name, sensor_type, model, width, height, fx, fy, cx, cy
                width = int(parts[4])
                height = int(parts[5])
                fx = float(parts[6])
                fy = float(parts[7])
                cx = float(parts[8])
                cy = float(parts[9])

                return {
                    'width': width,
                    'height': height,
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy
                }

    return None


def parse_images(images_file, camera_id):
    """Parse images.txt and get image paths for specific camera"""
    images = []
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(', ')
            if len(parts) < 3:
                continue

            timestamp = parts[0]
            cam_id = parts[1]
            img_path = parts[2]

            if cam_id == camera_id:
                images.append({
                    'timestamp': timestamp,
                    'path': img_path
                })

    return images


def convert_dataset(navvis_path, output_path, camera_id='cam0_center', use_symlinks=True):
    """
    Convert NavVis dataset to OpenYOLO3D format

    Args:
        navvis_path: Path to NavVis dataset
        output_path: Output path for OpenYOLO3D format
        camera_id: Which camera to use (cam0_center, cam1_center, cam2_center, cam3_center)
        use_symlinks: Use symlinks instead of copying files (saves space)
    """
    navvis_path = Path(navvis_path)
    output_path = Path(output_path)

    print(f"Converting NavVis dataset from {navvis_path}")
    print(f"Output path: {output_path}")
    print(f"Using camera: {camera_id}")
    print(f"Using symlinks: {use_symlinks}")

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    color_dir = output_path / 'color'
    depth_dir = output_path / 'depth'
    poses_dir = output_path / 'poses'

    color_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    poses_dir.mkdir(exist_ok=True)

    # Parse images
    print("\nParsing images...")
    images_file = navvis_path / 'images.txt'
    images = parse_images(images_file, camera_id)
    print(f"Found {len(images)} images for {camera_id}")

    # Parse poses
    print("\nParsing trajectories...")
    traj_file = navvis_path / 'trajectories.txt'
    poses = parse_trajectories(traj_file, camera_id)
    print(f"Found {len(poses)} poses for {camera_id}")

    # Parse intrinsics
    print("\nParsing intrinsics...")
    sensors_file = navvis_path / 'sensors.txt'
    intrinsics = parse_intrinsics(sensors_file, camera_id)

    if intrinsics:
        print(f"Camera intrinsics:")
        print(f"  Resolution: {intrinsics['width']}x{intrinsics['height']}")
        print(f"  fx={intrinsics['fx']}, fy={intrinsics['fy']}")
        print(f"  cx={intrinsics['cx']}, cy={intrinsics['cy']}")

        # Write intrinsics.txt
        intrinsics_file = output_path / 'intrinsics.txt'
        with open(intrinsics_file, 'w') as f:
            f.write(f"{intrinsics['fx']} {intrinsics['fy']} {intrinsics['cx']} {intrinsics['cy']}\n")
        print(f"Wrote intrinsics to {intrinsics_file}")

    # Verify we have matching number of images and poses
    if len(images) != len(poses):
        print(f"WARNING: Number of images ({len(images)}) != number of poses ({len(poses)})")
        print("Using minimum of both...")
        n_frames = min(len(images), len(poses))
    else:
        n_frames = len(images)

    # Convert images, depth maps, and poses
    print(f"\nProcessing {n_frames} frames...")

    raw_data_path = navvis_path / 'raw_data' / 'images_undistr_center'
    depth_maps_path = navvis_path / 'depth_maps'

    for i in range(n_frames):
        if i % 100 == 0:
            print(f"  Processing frame {i}/{n_frames}...")

        # Process RGB image
        img_basename = images[i]['path'].split('/')[-1]
        src_img = raw_data_path / img_basename

        # Convert jpg to match expected format
        dst_img = color_dir / f"{i}.jpg"

        if src_img.exists():
            if use_symlinks:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                dst_img.symlink_to(src_img.absolute())
            else:
                shutil.copy2(src_img, dst_img)
        else:
            print(f"WARNING: Image not found: {src_img}")

        # Process depth map
        depth_basename = img_basename.replace('.jpg', '.png')
        src_depth = depth_maps_path / depth_basename
        dst_depth = depth_dir / f"{i}.png"

        if src_depth.exists():
            if use_symlinks:
                if dst_depth.exists() or dst_depth.is_symlink():
                    dst_depth.unlink()
                dst_depth.symlink_to(src_depth.absolute())
            else:
                shutil.copy2(src_depth, dst_depth)
        else:
            print(f"WARNING: Depth map not found: {src_depth}")

        # Write pose as 4x4 matrix
        pose_file = poses_dir / f"{i}.txt"
        np.savetxt(pose_file, poses[i]['matrix'], fmt='%.8f')

    # Copy mesh if exists
    mesh_src = navvis_path / 'proc' / 'meshes' / 'mesh.ply'
    if mesh_src.exists():
        mesh_dst = output_path / f"{output_path.name}.ply"
        if use_symlinks:
            if mesh_dst.exists() or mesh_dst.is_symlink():
                mesh_dst.unlink()
            mesh_dst.symlink_to(mesh_src.absolute())
        else:
            shutil.copy2(mesh_src, mesh_dst)
        print(f"\nCopied mesh to {mesh_dst}")

    print(f"\nConversion complete!")
    print(f"Output directory: {output_path}")
    print(f"  - {n_frames} RGB images in color/")
    print(f"  - {n_frames} depth maps in depth/")
    print(f"  - {n_frames} poses in poses/")
    print(f"  - intrinsics.txt")
    if mesh_src.exists():
        print(f"  - {output_path.name}.ply (mesh)")


def main():
    parser = argparse.ArgumentParser(description='Convert NavVis dataset to OpenYOLO3D format')
    parser.add_argument('navvis_path', type=str, help='Path to NavVis dataset')
    parser.add_argument('output_path', type=str, help='Output path for OpenYOLO3D format')
    parser.add_argument('--camera', type=str, default='cam0_center',
                       choices=['cam0_center', 'cam1_center', 'cam2_center', 'cam3_center'],
                       help='Which camera to use (default: cam0_center)')
    parser.add_argument('--copy', action='store_true',
                       help='Copy files instead of using symlinks (uses more disk space)')

    args = parser.parse_args()

    convert_dataset(
        navvis_path=args.navvis_path,
        output_path=args.output_path,
        camera_id=args.camera,
        use_symlinks=not args.copy
    )


if __name__ == '__main__':
    main()
