"""The rayuela simulation as a ROS 2 node.

MuJoCo with the BAM actuators the policies were trained on, driven by the
ONNX PolicyInference stack (scripts/infer_policy.py) at 50 Hz. Publishes the
angled camera, the ground-truth duck pose and the active policy; consumes
velocity and behavior commands. How the duck behaves is in sim_setup.py; this
file is only the node around it.

Runs in the project venv (Python 3.12, created with --system-site-packages so
it also sees rclpy) — see the README.
"""

from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String

from rayuela import paths, ros_utils, sim_setup
from rayuela.sim_setup import (
    CONTROL_DECIMATION, IMAGE_PUBLISH_EVERY_N_CONTROL_STEPS, PUBLISH_TOPDOWN_GROUNDTRUTH,
)


class RayuelaSimNode(Node):
    def __init__(self):
        super().__init__("rayuela_sim_node")

        self.declare_parameter("use_viewer", False)
        self.declare_parameter("xml_path", str(paths.SCENE_XML))
        self.declare_parameter("policies_dir", str(paths.POLICIES_DIR))

        self.model, self.data, self.bam_ctrl = sim_setup.load_bam_sim(
            self.get_parameter("xml_path").value
        )
        self.policy = sim_setup.build_policy(
            self.model, self.data, self.get_parameter("policies_dir").value, self.bam_ctrl
        )
        self._vel_gate = sim_setup.VelCommandGate(self.policy)
        self._qpos_adr = sim_setup.place_duck_at_home(self.model, self.data, self.policy)
        self.bam_ctrl.reset(self.data.qpos)

        self.viewer = None
        if self.get_parameter("use_viewer").value:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
        self._cameras = sim_setup.SimCameras(self.model)

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

        self.control_dt = CONTROL_DECIMATION * self.model.opt.timestep
        self._control_step_count = 0

        # Once-per-second status line instead of a per-message flood: a 1s
        # moving average of the velocity the duck ACHIEVED (body frame, so it
        # is directly comparable to the command) next to what was commanded.
        freejoint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self._trunk_qvel_adr = int(self.model.jnt_dofadr[freejoint_id])
        self._status_window = max(1, int(round(1.0 / self.control_dt)))
        self._vel_history: deque = deque(maxlen=self._status_window)
        self._last_status_policy: str | None = None

        self.timer = self.create_timer(self.control_dt, self._on_timer)
        self.get_logger().info(
            f"rayuela_sim_node ready — BAM actuators, control_dt={self.control_dt:.4f}s, "
            f"viewer={'on' if self.viewer is not None else 'off'}"
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._vel_gate.apply(msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_behavior_cmd(self, msg: String) -> None:
        sim_setup.handle_behavior_cmd(
            self.policy, msg.data.strip(), duck_x=float(self.data.qpos[self._qpos_adr])
        )

    def _on_timer(self) -> None:
        if self.viewer is not None and not self.viewer.is_running():
            # Closing the viewer window ends the simulation (and, through the
            # launch file's on_exit, the whole game).
            self.get_logger().info("Viewer closed — shutting down")
            rclpy.shutdown()
            return

        stamp = self.get_clock().now().to_msg()
        sim_setup.step_behavior_timers(self.policy, self.control_dt)

        action = self.policy.infer()
        self.policy.apply_action(action)
        for _ in range(CONTROL_DECIMATION):
            self.bam_ctrl.update()
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        self._control_step_count += 1
        qpos = self.data.qpos[self._qpos_adr:self._qpos_adr + 7]
        # MuJoCo quat is (w, x, y, z); ROS is (x, y, z, w).
        self.pose_pub.publish(ros_utils.pose_msg(stamp, *qpos[:3], *qpos[4:7], qpos[3]))
        self.policy_state_pub.publish(String(data=self.policy.current_policy))
        self._update_status()
        if self._control_step_count % IMAGE_PUBLISH_EVERY_N_CONTROL_STEPS == 0:
            self._publish_images(stamp)

    def _publish_images(self, stamp) -> None:
        self._publish_frame(self.image_pub, self._cameras.render_angled(self.data),
                            Header(stamp=stamp, frame_id="angled_cam"))
        if PUBLISH_TOPDOWN_GROUNDTRUTH:
            self._publish_frame(self.topdown_debug_pub, self._cameras.render_topdown_debug(self.data),
                                Header(stamp=stamp, frame_id="topdown_debug"))

    @staticmethod
    def _publish_frame(pub, frame, header: Header) -> None:
        height, width, _ = frame.shape
        pub.publish(ros_utils.rgb8_image_msg(frame.tobytes(), width, height, header))

    def _update_status(self) -> None:
        """One status line per second: 1s-averaged achieved velocity (body
        frame, so it lines up with the command) vs what was commanded. Also
        logs immediately on a policy change, so switches are never buried."""
        adr, dadr = self._qpos_adr, self._trunk_qvel_adr
        quat = self.data.qpos[adr + 3:adr + 7].astype(np.float32)
        v_world = self.data.qvel[dadr:dadr + 3].astype(np.float32)
        v_body = self.policy.quat_rotate_inverse(quat, v_world)
        self._vel_history.append(
            (float(v_body[0]), float(v_body[1]), float(self.data.qvel[dadr + 5]))
        )

        policy_changed = self.policy.current_policy != self._last_status_policy
        if not policy_changed and self._control_step_count % self._status_window:
            return
        self._last_status_policy = self.policy.current_policy

        n = len(self._vel_history)
        avg = [sum(v[i] for v in self._vel_history) / n for i in range(3)]
        cmd = self.policy.vel_cmd
        self.get_logger().info(
            f"[{self.policy.current_policy:<9s}] "
            f"fwd {avg[0]:+.2f}/{cmd[0]:+.2f}  "
            f"lat {avg[1]:+.2f}/{cmd[1]:+.2f} m/s  "
            f"yaw {avg[2]:+.2f}/{cmd[2]:+.2f} rad/s  "
            f"z {self.data.qpos[adr + 2] * 1000:5.1f} mm"
            + ("   <- policy change" if policy_changed else "")
        )

    def destroy_node(self) -> bool:
        if self.viewer is not None:
            self.viewer.close()
        self._cameras.close()
        return super().destroy_node()


def main(args=None):
    ros_utils.spin_node(RayuelaSimNode, args)


if __name__ == "__main__":
    main()
