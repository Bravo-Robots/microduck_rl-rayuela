"""MuJoCo + PolicyInference worker — runs under the project's venv (Python
3.12: mujoco, onnxruntime, bam), NOT under ROS2's system Python. Talks to
rayuela_bridge_node over a Unix socket (see rayuela_ipc.py) instead of
rclpy: sends duck pose + camera frames, receives cmd_vel + behavior_cmd.

Uses real BAM actuators (bam.mujoco.MujocoController) — the model the
policies were actually trained against, unlike the plain-XML-actuator
fallback sim_node.py uses (kept for machines where a working `bam` install
isn't available to whichever Python has rclpy).

Run with the venv interpreter, e.g.:
    /home/robot/microduck_rl/.venv/bin/python3 sim_worker.py --use-viewer
"""

import argparse
import math
import os
import socket
import sys
import threading
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src/rayuela
sys.path.insert(0, _PKG_DIR)
from rayuela import board_geometry, kick_command, rayuela_ipc as ipc  # noqa: E402

sys.path.append('/home/robot/microduck_rl/scripts')
from infer_policy import (  # noqa: E402
    BAM_KP_FW, BAM_VIN_MIN, PolicyInference, load_bam_model, load_mujoco_with_bam,
)

XML_PATH = "/home/robot/microduck_rl/src/mjlab_microduck/robot/microduck/escena_rayuela.xml"
POLICIES_DIR = "/home/robot/microduck_rl/policies-v1"

CONTROL_DECIMATION = 4
CONTROL_TIMESTEP = 0.005  # policies trained at 50Hz = decimation(4) * 0.005s
IMAGE_PUBLISH_EVERY_N_CONTROL_STEPS = 3  # ~50Hz / 3 ≈ 16Hz
# The ground-truth top-down feed is redundant now that vision_node rectifies
# its own top-down canvas from the fiducials, and it doubles the per-publish
# cost (a second 1280x720 render + a second 2.76MB socket send + a second
# ROS Image on the bridge). Off by default; flip to True to get the debug
# feed on /rayuela/topdown_camera_groundtruth/image_raw again.
PUBLISH_TOPDOWN_GROUNDTRUTH = False

# BAM M6/XL330 defaults, matching infer_policy.py's CLI defaults (--kp-fw,
# --vin, --vin-drop-gain, --current-limit).
BAM_VIN = 7.4
BAM_VIN_DROP_GAIN = 0.1
BAM_CURRENT_LIMIT = 0.0  # <=0 -> None (no firmware current limit)

# Hysteresis on the walking<->standing switch. PolicyInference compares the
# command magnitude against a SINGLE switch_threshold, so a command sitting
# near it flaps the policy every tick ("Switched to standing (0.147)" /
# "Switched to walking (0.159)" / ...). Since _update_policy_session re-reads
# self.switch_threshold on every call, raising the bar to ENTER while standing
# and lowering it to EXIT while walking turns that one threshold into a proper
# Schmitt trigger: it takes a deliberate command to start moving, and a
# deliberate stop to settle, with no chatter in between.
SWITCH_ENTER = 0.1   # command magnitude needed to START walking
SWITCH_EXIT = 0.05    # must drop below this to fall back to standing


class SimWorker:
    def __init__(self, xml_path: str, policies_dir: str, use_viewer: bool):
        bam_model = load_bam_model(BAM_KP_FW, BAM_VIN, BAM_CURRENT_LIMIT)
        self.model, self.data, self.bam_ctrl, _names = load_mujoco_with_bam(
            xml_path, bam_model, CONTROL_TIMESTEP, BAM_VIN_DROP_GAIN, BAM_VIN_MIN
        )
        kick_left, kick_right, kick_speed_commanded = kick_command.select_kick_policies(policies_dir)
        print(f"Kick policies: {'SPEED-COMMANDED' if kick_speed_commanded else 'fixed-speed'} "
              f"({os.path.basename(kick_left)}, {os.path.basename(kick_right)})")

        self.policy = PolicyInference(
            self.model, self.data,
            bam_ctrl=self.bam_ctrl,
            walking_onnx_path=f'{policies_dir}/alpha_walking.onnx',
            action_scale=1.0,
            delay_min_lag=0,
            delay_max_lag=0,
            standing_onnx_path=f'{policies_dir}/alpha_standing.onnx',
            # 0.5 (the run_env.py/original default) exceeds vel_max_x=0.3 below,
            # so a pure-forward command's norm never crosses it and the duck
            # never leaves "standing" no matter what control_node commands —
            # lowered so any deliberate walk/turn command actually engages
            # the walking gait, while exact-zero idle still reads as standing.
            switch_threshold=SWITCH_ENTER,
            # See sim_node.py: TRUE matches infer_policy.py's CLI default.
            # False fed the raw IMU accelerometer, whose apparent-gravity
            # direction tilts with body acceleration — the policy then leans
            # forward to correct a tilt that isn't there.
            use_projected_gravity=True,
            ground_pick_onnx_path=f'{policies_dir}/alpha_ground_pick.onnx',
            ground_pick_period=1.0,
            sit_onnx_path=None,
            new_cmd_obs=True,
            slope_onnx_path=None,
            sitstand_onnx_path=f'{policies_dir}/alpha_sitstand.onnx',
            kick_left_onnx_path=kick_left,
            kick_right_onnx_path=kick_right,
            roulade_onnx_path=f'{policies_dir}/roulade.onnx',
            kick_duration=2.0,
            roulade_duration=2.0,
            kick_speed_commanded=kick_speed_commanded,
        )
        self.policy.set_vel_cmd(0.0, 0.0, 0.0)
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
        self.bam_ctrl.reset(self.data.qpos)

        self.viewer = None
        if use_viewer:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )

        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, board_geometry.CAMERA_NAME
        )
        self._renderer = mujoco.Renderer(
            self.model, height=board_geometry.CAMERA_HEIGHT, width=board_geometry.CAMERA_WIDTH
        )
        self._topdown_cam = mujoco.MjvCamera()
        self._topdown_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._topdown_cam.orthographic = 1
        self._topdown_cam.lookat = list(board_geometry.TOPDOWN_DEBUG_CAM_LOOKAT)
        self._topdown_cam.distance = board_geometry.TOPDOWN_DEBUG_CAM_DISTANCE
        self._topdown_cam.azimuth = board_geometry.TOPDOWN_DEBUG_CAM_AZIMUTH
        self._topdown_cam.elevation = board_geometry.TOPDOWN_DEBUG_CAM_ELEVATION

        self.control_dt = CONTROL_DECIMATION * self.model.opt.timestep
        self._control_step_count = 0

        # Once-per-second status line instead of a per-message flood: a 1s
        # moving average of the velocity the duck ACHIEVED (body frame, so it
        # is directly comparable to the command) next to what was commanded.
        # Ported from run_env.py, which prints the same summary.
        self._trunk_qvel_adr = int(self.model.jnt_dofadr[self._freejoint_id])
        self._status_window = max(1, int(round(1.0 / self.control_dt)))
        self._vel_history: deque = deque(maxlen=self._status_window)
        self._last_status_policy: str | None = None

        self.sock: socket.socket | None = None
        self._sock_lock = threading.Lock()
        self._last_cmd_key = None  # dedup identical cmd_vel re-applies

    def connect(self) -> None:
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(ipc.SOCKET_PATH)
                self.sock = sock
                print(f"Connected to bridge at {ipc.SOCKET_PATH}")
                threading.Thread(target=self._recv_loop, daemon=True).start()
                return
            except OSError as e:
                print(f"Waiting for bridge ({ipc.SOCKET_PATH}): {e}")
                time.sleep(1.0)

    def _recv_loop(self) -> None:
        while True:
            frame = ipc.read_frame(self.sock)
            if frame is None:
                print("Bridge disconnected")
                return
            msg_type, body = frame
            if msg_type == ipc.MSG_CMD_VEL:
                vx, vy, wz = ipc.unpack_cmd_vel(body)
                # control_node republishes at 20Hz whether or not the command
                # changed, and PolicyInference.set_vel_cmd prints on every
                # call — that is the wall of identical "Vel cmd: ..." lines.
                # Skip identical re-applies (they are idempotent anyway).
                # current_policy is part of the key on purpose: _end_behavior
                # zeroes vel_cmd when a kick/roulade finishes, so without it a
                # still-unchanged walk command would never be re-applied and
                # the duck would just stand there after a trick.
                key = (vx, vy, wz, self.policy.current_policy)
                if key != self._last_cmd_key:
                    self._last_cmd_key = key
                    # Schmitt trigger (see SWITCH_ENTER/SWITCH_EXIT): the bar to
                    # start walking is higher than the bar to keep walking.
                    self.policy.switch_threshold = (
                        SWITCH_EXIT if self.policy.current_policy == "walking"
                        else SWITCH_ENTER
                    )
                    self.policy.set_vel_cmd(vx, vy, wz)
            elif msg_type == ipc.MSG_BEHAVIOR:
                self._on_behavior(ipc.unpack_behavior(body))

    def _on_behavior(self, name: str) -> None:
        if name == "ground_pick":
            self.policy.trigger_ground_pick()
        elif name == "toggle_sit":
            self.policy.toggle_sit()
        elif kick_command.is_kick_command(name):
            try:
                kick, speed = kick_command.parse_kick_command(
                    name, duck_x=float(self.data.qpos[self._qpos_adr])
                )
            except ValueError as e:
                print(f"Bad kick command '{name}': {e}")
                return
            if speed is not None and not self.policy.kick_speed_commanded:
                print(f"'{name}': fixed-speed kick policies loaded, target speed ignored")
            print(f"{kick}" + (f" at {speed:.2f} m/s" if speed is not None else ""))
            self.policy.trigger_behavior(kick, kick_speed=speed)
        elif name == "stop":
            # Not a behavior session: stop_all ends whatever is running (kick,
            # roulade, sit, pick) and drops into the standing policy.
            self.policy.stop_all()
        elif name in self.policy.behavior_sessions:
            self.policy.trigger_behavior(name)
        else:
            print(f"Unknown behavior_cmd '{name}'")

    def _send(self, msg_type: int, body: bytes) -> None:
        with self._sock_lock:
            try:
                ipc.send_frame(self.sock, msg_type, body)
            except OSError as e:
                print(f"Send failed: {e}")

    def run(self) -> None:
        print("sim_worker running — Ctrl+C to stop")
        prev_time = time.time()
        try:
            while True:
                step_start = time.time()
                prev_time = step_start

                # SIM time, not wall time. mj_step below advances the world by
                # exactly control_dt, so the behavior timers have to drain on
                # the same clock. Feeding them wall-clock dt (what run_env.py /
                # infer_policy.py do, where it's harmless because their loop is
                # just a viewer sync) made trick duration depend on render and
                # socket load: this loop ships two 2.7MB frames every 3rd step,
                # so it routinely overruns its 20ms budget and the roulade was
                # being cut off after ~0.6s of roll instead of running to
                # completion — it only travelled 0.14m instead of 0.57m.
                self.policy.update_ground_pick_phase(self.control_dt)
                self.policy.update_behavior(self.control_dt)
                # Sim time, like update_behavior: a fall while sitting has to
                # hand control back to the standing policy, or the duck lies
                # there under a policy that cannot get up (see
                # infer_policy.SIT_FALL_TILT_RAD).
                self.policy.update_sit_fall_watchdog(self.control_dt)

                action = self.policy.infer()
                self.policy.apply_action(action)

                for _ in range(CONTROL_DECIMATION):
                    self.bam_ctrl.update()
                    mujoco.mj_step(self.model, self.data)

                if self.viewer is not None:
                    if not self.viewer.is_running():
                        break
                    self.viewer.sync()

                self._control_step_count += 1
                self._send_pose(step_start)
                self._send(ipc.MSG_POLICY_STATE, ipc.pack_policy_state(self.policy.current_policy))
                self._update_status()
                if self._control_step_count % IMAGE_PUBLISH_EVERY_N_CONTROL_STEPS == 0:
                    self._send_images(step_start)

                elapsed = time.time() - step_start
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            if self.viewer is not None:
                self.viewer.close()
            self._renderer.close()

    def _update_status(self) -> None:
        """One status line per second: 1s-averaged achieved velocity (body
        frame, so it lines up with the command) vs what was commanded. Also
        prints immediately on a policy change, so switches are never buried."""
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
        print(
            f"[{self.policy.current_policy:<9s}] "
            f"fwd {avg[0]:+.2f}/{cmd[0]:+.2f}  "
            f"lat {avg[1]:+.2f}/{cmd[1]:+.2f} m/s  "
            f"yaw {avg[2]:+.2f}/{cmd[2]:+.2f} rad/s  "
            f"z {self.data.qpos[adr + 2] * 1000:5.1f} mm"
            + ("   <- policy change" if policy_changed else ""),
            flush=True,
        )

    def _send_pose(self, stamp: float) -> None:
        qpos = self.data.qpos[self._qpos_adr:self._qpos_adr + 7]
        # MuJoCo quat is (w, x, y, z); ROS is (x, y, z, w).
        body = ipc.pack_pose(
            stamp, float(qpos[0]), float(qpos[1]), float(qpos[2]),
            float(qpos[4]), float(qpos[5]), float(qpos[6]), float(qpos[3]),
        )
        self._send(ipc.MSG_POSE, body)

    def _send_images(self, stamp: float) -> None:
        self._renderer.update_scene(self.data, camera=self._cam_id)
        frame = self._renderer.render()
        h, w, _ = frame.shape
        self._send(ipc.MSG_IMAGE, ipc.pack_image(
            ipc.CAM_ANGLED, stamp, w, h, np.ascontiguousarray(frame).tobytes()
        ))

        if not PUBLISH_TOPDOWN_GROUNDTRUTH:
            return
        self._renderer.update_scene(self.data, camera=self._topdown_cam)
        topdown = self._renderer.render()
        h, w, _ = topdown.shape
        self._send(ipc.MSG_IMAGE, ipc.pack_image(
            ipc.CAM_TOPDOWN_DEBUG, stamp, w, h, np.ascontiguousarray(topdown).tobytes()
        ))


def main():
    parser = argparse.ArgumentParser(description="rayuela sim_worker (BAM actuators, talks to rayuela_bridge_node)")
    parser.add_argument("--use-viewer", action="store_true")
    parser.add_argument("--xml-path", default=XML_PATH)
    parser.add_argument("--policies-dir", default=POLICIES_DIR)
    args = parser.parse_args()

    worker = SimWorker(args.xml_path, args.policies_dir, args.use_viewer)
    worker.connect()
    worker.run()


if __name__ == "__main__":
    main()
