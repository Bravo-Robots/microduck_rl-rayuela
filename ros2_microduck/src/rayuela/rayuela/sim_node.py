"""ROS2 entry point for the rayuela MuJoCo world.

Owns the MuJoCo model/data and the ONNX PolicyInference stack (same setup as
``run_env.py``, minus the blocking viewer loop and terminal teleop — those
become topics so other nodes, or a human via teleop_keyboard, can drive the
duck). Publishes the angled-camera view and ground-truth duck pose; consumes
velocity and behavior commands.
"""

import array
import math

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from rayuela import board_geometry, kick_command, repo_paths

import sys
sys.path.append(repo_paths.scripts_dir())
from infer_policy import PolicyInference  # noqa: E402

XML_PATH = repo_paths.scene_xml()
POLICIES_DIR = repo_paths.policies_dir()

CONTROL_DECIMATION = 4
IMAGE_PUBLISH_EVERY_N_CONTROL_STEPS = 3  # ~50Hz / 3 ≈ 16Hz


class RayuelaSimNode(Node):
    def __init__(self):
        super().__init__("rayuela_sim_node")

        self.declare_parameter("use_viewer", False)
        self.declare_parameter("xml_path", XML_PATH)
        self.declare_parameter("policies_dir", POLICIES_DIR)

        xml_path = self.get_parameter("xml_path").value
        policies_dir = self.get_parameter("policies_dir").value

        kick_left, kick_right, kick_speed_commanded = kick_command.select_kick_policies(policies_dir)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        # Policies are trained at 50Hz with decimation=4 (0.005s * 4 = 0.02s).
        # MuJoCo's default timestep is 0.002s — without this override the
        # walking gait ran at 125Hz (2.5x too fast), which was destabilizing
        # it into falls. Matches infer_policy.py's --no-bam / BAM setup paths.
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)

        self.policy = PolicyInference(
            self.model, self.data,
            bam_ctrl=None,
            walking_onnx_path=f'{policies_dir}/alpha_walking.onnx',
            action_scale=1.0,
            delay_min_lag=0,
            delay_max_lag=0,
            standing_onnx_path=f'{policies_dir}/alpha_standing.onnx',
            # See sim_worker.py: 0.5 exceeds vel_max_x=0.3, so a pure-forward
            # command never crosses it and the duck never leaves "standing".
            switch_threshold=0.15,
            # TRUE, matching infer_policy.py's CLI default
            # (use_projected_gravity=not args.raw_accelerometer). The False
            # inherited from run_env.py fed the policy the raw IMU
            # accelerometer instead: same unit magnitude, but contaminated by
            # body acceleration, so accelerating forward tilts the "apparent
            # gravity" backward and the policy leans forward to correct a
            # tilt that isn't there — the slow, head-forward gait that kept
            # ending on the floor.
            use_projected_gravity=True,
            ground_pick_onnx_path=f'{policies_dir}/alpha_ground_pick.onnx',
            ground_pick_period=1.0,
            sit_onnx_path=None,
            new_cmd_obs=True,
            slope_onnx_path=None,
            sitstand_onnx_path=f'{policies_dir}/alpha_sitstand.onnx',
            kick_left_onnx_path=kick_left,
            kick_right_onnx_path=kick_right,
            kick_speed_commanded=kick_speed_commanded,
            roulade_onnx_path=f'{policies_dir}/roulade.onnx',
            kick_duration=3.0,
            roulade_duration=2.0,
        )
        self.policy.set_vel_cmd(0.0, 0.0, 0.0)
        # Force standing from the very first control step — with both walking
        # and standing sessions loaded, the constructor otherwise defaults
        # current_policy to "walking" (zero command, not a settled stand),
        # which was pitching the duck forward before the first cmd_vel arrived.
        if self.policy.standing_session is not None:
            self.policy.current_policy = "standing"
            self.policy.ort_session = self.policy.standing_session
            self.policy._update_command()
        self.policy.vel_max_x = 0.3
        self.policy.vel_min_x = -0.3
        self.policy.vel_max_y = 0.2
        self.policy.vel_min_y = -0.2
        self.policy.vel_max_ang = 1.5

        self._freejoint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
        )
        self._qpos_adr = self.model.jnt_qposadr[self._freejoint_id]
        self.data.qpos[self._qpos_adr + 0] = 0.0
        self.data.qpos[self._qpos_adr + 1] = 0.0
        self.data.qpos[self._qpos_adr + 2] = 0.125
        self.data.qpos[self._qpos_adr + 3:self._qpos_adr + 7] = [1, 0, 0, 0]
        for i, qpos_idx in enumerate(self.policy.joint_qpos_indices):
            self.data.qpos[qpos_idx] = self.policy.default_pose[i]
        self.policy.set_position_targets(self.policy.default_pose)
        mujoco.mj_forward(self.model, self.data)

        self.use_viewer = self.get_parameter("use_viewer").value
        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )

        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, board_geometry.CAMERA_NAME
        )
        self._renderer = mujoco.Renderer(
            self.model, height=board_geometry.CAMERA_HEIGHT, width=board_geometry.CAMERA_WIDTH
        )

        # Sim-only ground-truth top-down view: this MuJoCo build has no
        # per-fixed-camera orthographic mode, so a free MjvCamera is aimed
        # straight down in orthographic mode instead. Purely a debug feed to
        # sanity-check vision_node's fiducial-homography rectification — a
        # real deployment has no ceiling camera and doesn't use this path.
        self._topdown_debug_cam = mujoco.MjvCamera()
        self._topdown_debug_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._topdown_debug_cam.orthographic = 1
        self._topdown_debug_cam.lookat = list(board_geometry.TOPDOWN_DEBUG_CAM_LOOKAT)
        self._topdown_debug_cam.distance = board_geometry.TOPDOWN_DEBUG_CAM_DISTANCE
        self._topdown_debug_cam.azimuth = board_geometry.TOPDOWN_DEBUG_CAM_AZIMUTH
        self._topdown_debug_cam.elevation = board_geometry.TOPDOWN_DEBUG_CAM_ELEVATION

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

        self._camera_info_msg = self._build_camera_info()
        self.camera_info_pub.publish(self._camera_info_msg)

        self.create_subscription(Twist, "/rayuela/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(String, "/rayuela/behavior_cmd", self._on_behavior_cmd, 10)

        self.control_dt = CONTROL_DECIMATION * self.model.opt.timestep
        self._control_step_count = 0
        self._prev_time = self.get_clock().now()
        self.timer = self.create_timer(self.control_dt, self._on_timer)

        self.get_logger().info(
            f"rayuela_sim_node ready — control_dt={self.control_dt:.4f}s "
            f"viewer={'on' if self.use_viewer else 'off'}"
        )

    def _build_camera_info(self) -> CameraInfo:
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

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.policy.set_vel_cmd(msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_behavior_cmd(self, msg: String) -> None:
        name = msg.data.strip()
        if name == "ground_pick":
            self.policy.trigger_ground_pick()
        elif name == "toggle_sit":
            self.policy.toggle_sit()
        elif name == "stop":
            self.policy.stop_all()
        elif kick_command.is_kick_command(name):
            try:
                kick, speed = kick_command.parse_kick_command(
                    name, duck_x=float(self.data.qpos[self._qpos_adr])
                )
            except ValueError as e:
                self.get_logger().warn(f"Bad kick command '{name}': {e}")
                return
            if speed is not None and not self.policy.kick_speed_commanded:
                self.get_logger().warn(f"'{name}': fixed-speed kick policies loaded, target speed ignored")
            self.policy.trigger_behavior(kick, kick_speed=speed)
        elif name in self.policy.behavior_sessions:
            self.policy.trigger_behavior(name)
        else:
            self.get_logger().warn(f"Unknown behavior_cmd '{name}'")

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        self._prev_time = now

        # See sim_worker.py: SIM time, not wall time. mj_step advances the
        # world by exactly control_dt, so the trick timers must drain on the
        # same clock — otherwise render/publish load cuts behaviors short.
        self.policy.update_ground_pick_phase(self.control_dt)
        self.policy.update_behavior(self.control_dt)

        action = self.policy.infer()
        self.policy.apply_action(action)

        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        self._control_step_count += 1
        self._publish_pose(now)
        self.policy_state_pub.publish(String(data=self.policy.current_policy))
        if self._control_step_count % IMAGE_PUBLISH_EVERY_N_CONTROL_STEPS == 0:
            self._publish_image(now)

    def _publish_pose(self, stamp) -> None:
        qpos = self.data.qpos[self._qpos_adr:self._qpos_adr + 7]
        msg = PoseStamped()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(qpos[0])
        msg.pose.position.y = float(qpos[1])
        msg.pose.position.z = float(qpos[2])
        # MuJoCo quat is (w, x, y, z); ROS is (x, y, z, w).
        msg.pose.orientation.w = float(qpos[3])
        msg.pose.orientation.x = float(qpos[4])
        msg.pose.orientation.y = float(qpos[5])
        msg.pose.orientation.z = float(qpos[6])
        self.pose_pub.publish(msg)

    def _publish_image(self, stamp) -> None:
        self._renderer.update_scene(self.data, camera=self._cam_id)
        frame = self._renderer.render()  # HxWx3 uint8, RGB
        self.image_pub.publish(self._make_image_msg(frame, stamp, "angled_cam"))

        self._renderer.update_scene(self.data, camera=self._topdown_debug_cam)
        topdown_frame = self._renderer.render()
        self.topdown_debug_pub.publish(
            self._make_image_msg(topdown_frame, stamp, "topdown_debug")
        )

    @staticmethod
    def _make_image_msg(frame: np.ndarray, stamp, frame_id: str) -> Image:
        height, width, _ = frame.shape
        img_msg = Image()
        img_msg.header.stamp = stamp.to_msg()
        img_msg.header.frame_id = frame_id
        img_msg.height = height
        img_msg.width = width
        img_msg.encoding = "rgb8"
        img_msg.is_bigendian = 0
        img_msg.step = width * 3
        # array.array('B'), not bytes — see bridge_node._publish_image:
        # the bytes path costs 163ms per 2.76MB frame vs 0.06ms.
        img_msg.data = array.array('B', np.ascontiguousarray(frame).tobytes())
        return img_msg

    def destroy_node(self) -> bool:
        if self.viewer is not None:
            self.viewer.close()
        self._renderer.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RayuelaSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
