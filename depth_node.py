#!/usr/bin/env python3
"""
depth_node.py
-------------
ROS 2 node simulating "UniDepth as a perception node" in a mobile robot's
software stack. Subscribes to /camera/image_raw (published by
image_publisher.py), matches each incoming frame to its PRECOMPUTED UniDepth
depth map (exported from Colab, since running UniDepth live inside WSL2/Docker
is unnecessary risk with limited time), and republishes it on /unidepth/depth.

This models the real robot architecture (camera -> depth perception node ->
downstream planner) without requiring GPU/UniDepth install inside WSL2.

If you DO get GPU-in-Docker working and want live inference instead, swap the
`load_precomputed_depth()` call for an actual model.infer() call — the ROS2
node structure/topics stay identical either way.

Usage:
    python3 depth_node.py --export-dir ~/mr_project/ros2_export
"""

import argparse
import os
import glob

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header


def imgmsg_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image (rgb8) back to an HxWx3 uint8 numpy array."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return arr.reshape(msg.height, msg.width, 3)


def depth_to_imgmsg(depth: np.ndarray, frame_id: str = "camera_frame") -> Image:
    """Convert a float32 HxW depth map (metres) to a sensor_msgs/Image
    (encoding '32FC1'), without cv_bridge."""
    msg = Image()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.height, msg.width = depth.shape
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 4  # 4 bytes per float32
    msg.data = depth.astype(np.float32).tobytes()
    return msg


class DepthNode(Node):
    def __init__(self, export_dir: str):
        super().__init__('unidepth_perception_node')

        self.frame_paths = sorted(glob.glob(os.path.join(export_dir, 'frame_*.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(export_dir, 'depth_*.npy')))
        if not self.depth_paths:
            self.get_logger().error(f"No depth_*.npy files found in {export_dir}")
            raise FileNotFoundError(f"No precomputed depth files in {export_dir}")

        # Map basename index -> depth array, so we can match incoming frames
        # to the correct precomputed depth regardless of publish order.
        self.depth_lookup = {}
        for i, dpath in enumerate(self.depth_paths):
            idx_str = os.path.basename(dpath).replace('depth_', '').replace('.npy', '')
            self.depth_lookup[idx_str] = np.load(dpath)

        self.frame_index_cycle = [
            os.path.basename(p).replace('frame_', '').replace('.png', '')
            for p in self.frame_paths
        ]
        self.get_logger().info(
            f"Loaded {len(self.depth_lookup)} precomputed depth maps from {export_dir}"
        )

        self.received_count = 0
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.depth_pub = self.create_publisher(Image, '/unidepth/depth', 10)

    def image_callback(self, msg: Image):
        # Match this incoming frame to its precomputed depth via cyclic index
        # (mirrors the publisher's loop order).
        idx_str = self.frame_index_cycle[self.received_count % len(self.frame_index_cycle)]
        depth = self.depth_lookup[idx_str]

        depth_msg = depth_to_imgmsg(depth)
        self.depth_pub.publish(depth_msg)

        self.get_logger().info(
            f"[unidepth/depth] frame {idx_str} | "
            f"min:{depth.min():.1f}m max:{depth.max():.1f}m mean:{depth.mean():.1f}m"
        )
        self.received_count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-dir', type=str, required=True,
                         help='Path to folder containing frame_XX.png and depth_XX.npy files')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = DepthNode(export_dir=args.export_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
