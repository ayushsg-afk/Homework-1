#!/usr/bin/env python3
"""
tf_broadcaster_node.py -- HW1 Part 2, Task 3: TF Broadcaster for Body vs Fixed Frame
Animates elemental rotations smoothly and allows live toggling between 
'current' (intrinsic) and 'fixed' (extrinsic) frame composition via ROS 2 parameters.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


ELEMENTARY_ROTATIONS = {"x": Rx, "y": Ry, "z": Rz}

# Rotation sequence matching Task 1
STEP_SEQUENCE = [
    ("z", np.deg2rad(90)),
    ("x", np.deg2rad(90)),
    ("y", np.deg2rad(60)),
]


def R_to_quat_xyzw(R):
    """3x3 rotation matrix -> ROS-convention xyzw quaternion."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


class Hw01TfBroadcaster(Node):
    def __init__(self):
        super().__init__("hw01_tf_broadcaster")
        self.declare_parameter("compose_frame", "current")  # "current" or "fixed"
        self.declare_parameter("publish_rate", 30.0)        # 30 Hz timer rate
        self.declare_parameter("step_duration", 2.0)       # Seconds per elemental step

        self.tf_broadcaster = TransformBroadcaster(self)

        # State tracking
        self.R_base = np.eye(3)        # Rotation at the start of current step
        self.R_current = np.eye(3)     # Animated current rotation
        self.step_index = 0
        self.step_elapsed = 0.0

        publish_rate = self.get_parameter("publish_rate").value
        self.dt = 1.0 / publish_rate
        self.timer = self.create_timer(self.dt, self.on_timer)

        self.get_logger().info(
            "hw01_tf_broadcaster active. Toggle live using:\n"
            "  ros2 param set /hw01_tf_broadcaster compose_frame fixed\n"
            "  ros2 param set /hw01_tf_broadcaster compose_frame current"
        )

    def on_timer(self):
        # 1. Broadcast fixed space frame (static origin)
        self.broadcast_frame("world", "space_frame", np.eye(3))

        step_duration = self.get_parameter("step_duration").value
        axis, total_angle = STEP_SEQUENCE[self.step_index % len(STEP_SEQUENCE)]

        # 2. Advance time and compute interpolated angle
        self.step_elapsed += self.dt
        progress = min(1.0, self.step_elapsed / step_duration)
        angle_step = total_angle * progress
        R_step = ELEMENTARY_ROTATIONS[axis](angle_step)

        # 3. Read live ROS parameter for composition mode
        compose_frame = self.get_parameter("compose_frame").value.lower()

        # 4. Composition logic matching Task 1
        if compose_frame == "fixed":
            # Extrinsic: Pre-multiplication
            self.R_current = R_step @ self.R_base
        else:
            # Intrinsic: Post-multiplication
            self.R_current = self.R_base @ R_step

        # 5. Handle step completion and state transition
        if progress >= 1.0:
            self.R_base = self.R_current.copy()
            self.step_elapsed = 0.0
            self.step_index += 1

        # 6. Broadcast moving body frame
        self.broadcast_frame("world", "body_frame", self.R_current)

    def broadcast_frame(self, parent, child, R):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.0 if child == "body_frame" else 0.0
        qx, qy, qz, qw = R_to_quat_xyzw(R)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = Hw01TfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
