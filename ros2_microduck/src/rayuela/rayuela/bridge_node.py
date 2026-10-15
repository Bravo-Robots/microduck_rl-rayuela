"""ROS2-side half of the sim_worker bridge.

sim_worker.py owns the MuJoCo + PolicyInference simulation and runs under
the project's venv (Python 3.12 — has mujoco/onnxruntime/bam). rclpy is only
importable under this machine's system Python 3.10, so it can't live in the
same process as sim_worker. This node is the rclpy-side counterpart: it
listens on a Unix domain socket (rayuela_ipc.SOCKET_PATH), republishes what
sim_worker sends (pose, camera frames) as normal ROS2 topics, and forwards
/rayuela/cmd_vel + /rayuela/behavior_cmd down to sim_worker over the same
socket. Everything else in the rayuela stack (vision_node, control_node,
teleop_keyboard) is unaffected — they only ever see real ROS2 topics.
"""

import os
import socket
import threading

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String

from rayuela import rayuela_ipc as ipc, ros_utils


class RayuelaBridgeNode(Node):
    def __init__(self):
        super().__init__("rayuela_bridge_node")

        self._client_sock: socket.socket | None = None
        self._client_lock = threading.Lock()
        self._warned_no_client = False

        latched_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.image_pub = self.create_publisher(Image, "/rayuela/angled_camera/image_raw", 10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, "/rayuela/angled_camera/camera_info", latched_qos
        )
        self.topdown_debug_pub = self.create_publisher(
            Image, "/rayuela/topdown_camera_groundtruth/image_raw", 10
        )
        self.pose_pub = self.create_publisher(PoseStamped, "/rayuela/duck_pose", 10)
        self.policy_state_pub = self.create_publisher(String, "/rayuela/current_policy", 10)
        self.camera_info_pub.publish(ros_utils.camera_info_msg())

        self.create_subscription(Twist, "/rayuela/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(String, "/rayuela/behavior_cmd", self._on_behavior_cmd, 10)

        self._stop = threading.Event()
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        self.get_logger().info(f"rayuela_bridge_node listening on {ipc.SOCKET_PATH}")

    def _accept_loop(self) -> None:
        try:
            os.unlink(ipc.SOCKET_PATH)
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(ipc.SOCKET_PATH)
        server.listen(1)
        while not self._stop.is_set():
            conn, _ = server.accept()
            self.get_logger().info("sim_worker connected")
            with self._client_lock:
                self._client_sock = conn
                self._warned_no_client = False
            try:
                self._read_loop(conn)
            except Exception:
                # Shutting down: the socket was closed or the ROS context
                # went away under the reader. Anything else is a real error.
                if self._stop.is_set() or not rclpy.ok():
                    return
                raise
            with self._client_lock:
                if self._client_sock is conn:
                    self._client_sock = None
            self.get_logger().warn("sim_worker disconnected — waiting for reconnect")

    def _read_loop(self, conn: socket.socket) -> None:
        while not self._stop.is_set():
            frame = ipc.read_frame(conn)
            if frame is None:
                return
            msg_type, body = frame
            if msg_type == ipc.MSG_POSE:
                self._publish_pose(*ipc.unpack_pose(body))
            elif msg_type == ipc.MSG_IMAGE:
                self._publish_image(*ipc.unpack_image(body))
            elif msg_type == ipc.MSG_POLICY_STATE:
                self.policy_state_pub.publish(String(data=ipc.unpack_policy_state(body)))

    def _publish_pose(self, stamp_sec, x, y, z, qx, qy, qz, qw) -> None:
        stamp = self.get_clock().now().to_msg()
        self.pose_pub.publish(ros_utils.pose_msg(stamp, x, y, z, qx, qy, qz, qw))

    def _publish_image(self, cam_id, stamp_sec, width, height, rgb_bytes) -> None:
        stamp = self.get_clock().now().to_msg()
        if cam_id == ipc.CAM_ANGLED:
            pub, frame_id = self.image_pub, "angled_cam"
        elif cam_id == ipc.CAM_TOPDOWN_DEBUG:
            pub, frame_id = self.topdown_debug_pub, "topdown_debug"
        else:
            return
        pub.publish(ros_utils.rgb8_image_msg(
            rgb_bytes, width, height, Header(stamp=stamp, frame_id=frame_id)
        ))

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._send(ipc.MSG_CMD_VEL, ipc.pack_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z))

    def _on_behavior_cmd(self, msg: String) -> None:
        self._send(ipc.MSG_BEHAVIOR, ipc.pack_behavior(msg.data.strip()))

    def _send(self, msg_type: int, body: bytes) -> None:
        with self._client_lock:
            sock = self._client_sock
            if sock is None:
                if not self._warned_no_client:
                    self.get_logger().warn("No sim_worker connected — dropping command(s)")
                    self._warned_no_client = True
                return
            try:
                ipc.send_frame(sock, msg_type, body)
            except OSError as e:
                self.get_logger().warn(f"Failed to send to sim_worker: {e}")

    def destroy_node(self) -> bool:
        self._stop.set()
        with self._client_lock:
            if self._client_sock is not None:
                self._client_sock.close()
        return super().destroy_node()


def main(args=None):
    ros_utils.spin_node(RayuelaBridgeNode, args)


if __name__ == "__main__":
    main()
