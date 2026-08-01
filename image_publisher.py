#!/usr/bin/env python3
"""
image_publisher.py
-------------------
ROS 2 node simulating a mobile robot's forward-facing camera.
Publishes each exported KITTI frame on /camera/image_raw at ~2 Hz, looping.

Usage (inside ros:humble container / WSL2 with ROS2 sourced):
    python3 image_publisher.py --export-dir ~/mr_project/ros2_export
"""

import argparse
import os
import glob

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header


def numpy_to_imgmsg(img_np: np.ndarray, frame_id: str = "camera_frame") -> Image:
    """Convert an HxWx3 uint8 RGB numpy array to a sensor_msgs/Image, without
    needing cv_bridge (kept dependency-light for a quick WSL2/Docker setup)."""
    msg = Image()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.height, msg.width, _ = img_np.shape
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = img_np.tobytes()
    return msg


class ImagePublisher(Node):
    def __init__(self, export_dir: str, rate_hz: float = 2.0):
        super().__init__('kitti_image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        self.frame_paths = sorted(glob.glob(os.path.join(export_dir, 'frame_*.png')))
        if not self.frame_paths:
            self.get_logger().error(f"No frame_*.png files found in {export_dir}")
            raise FileNotFoundError(f"No frames in {export_dir}")

        self.get_logger().info(f"Loaded {len(self.frame_paths)} frames from {export_dir}")

        self.index = 0
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_next_frame)

    def publish_next_frame(self):
        path = self.frame_paths[self.index]
        img_np = np.array(PILImage.open(path).convert('RGB'))
        msg = numpy_to_imgmsg(img_np)
        self.publisher_.publish(msg)
        self.get_logger().info(
            f"Published frame {self.index+1}/{len(self.frame_paths)}: {os.path.basename(path)}"
        )
        self.index = (self.index + 1) % len(self.frame_paths)  # loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-dir', type=str, required=True,
                         help='Path to folder containing frame_XX.png files')
    parser.add_argument('--rate', type=float, default=2.0,
                         help='Publish rate in Hz (default: 2.0)')
    args, _ = parser.parse_known_args()  # ros2 run passes extra args; ignore unknown

    rclpy.init()
    node = ImagePublisher(export_dir=args.export_dir, rate_hz=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
