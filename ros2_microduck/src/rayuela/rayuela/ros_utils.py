"""Helpers shared by the rayuela rclpy nodes: message builders and ``main()``."""

import array
import signal

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image

from rayuela import board_geometry


def seconds(duration: Duration) -> float:
    """A Duration (e.g. ``clock.now() - stamp``) in float seconds."""
    return duration.nanoseconds * 1e-9


def camera_info_msg() -> CameraInfo:
    """CameraInfo for the simulated angled camera (no distortion)."""
    fx, fy, cx, cy = board_geometry.camera_intrinsics()
    msg = CameraInfo()
    msg.header.frame_id = "angled_cam"
    msg.width = board_geometry.CAMERA_WIDTH
    msg.height = board_geometry.CAMERA_HEIGHT
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def rgb8_image_msg(rgb_bytes: bytes, width: int, height: int, header) -> Image:
    """Image message for a packed rgb8 frame.

    ``header`` is a std_msgs/Header whose stamp and frame_id are copied.
    """
    msg = Image()
    msg.header.stamp = header.stamp
    msg.header.frame_id = header.frame_id
    msg.height = height
    msg.width = width
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = width * 3
    # array.array('B', ...), NOT bytes. rosidl's generated setter for a
    # uint8[] field fast-paths an array.array with typecode 'B' straight into
    # the message; anything else falls into a __debug__ branch that validates
    # EVERY element in Python. For a 2.76MB frame that is 163ms per image vs
    # 0.06ms — measured, and at ~16 Hz it alone would eat most of the 20 ms
    # control step of the node publishing the frames.
    msg.data = array.array('B', rgb_bytes)
    return msg


def pose_msg(stamp, x: float, y: float, z: float,
             qx: float, qy: float, qz: float, qw: float) -> PoseStamped:
    """Duck pose in the world frame (quaternion in ROS order: x, y, z, w)."""
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = "world"
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def spin_node(node_cls, args=None) -> None:
    """Init rclpy, spin one ``node_cls()`` until Ctrl+C, and shut down quietly.

    On Ctrl+C, rclpy shuts the context down from its own signal handler:
    spin() then raises ExternalShutdownException (not KeyboardInterrupt), a
    callback that was mid-flight can fail to publish ("publisher's context is
    invalid"), and a second rclpy.shutdown() raises "already called". None of
    that is a real error, so it is swallowed — but only once the context is
    gone; anything raised while it is still up propagates.
    """
    rclpy.init(args=args)
    node = node_cls()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        # A Ctrl+C reaches each node twice (the terminal signals the whole
        # process group and ros2 launch forwards its own); the second must
        # not abort the cleanup half-way, or resources such as an EGL renderer
        # are left to __del__ at interpreter exit and fail there.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
