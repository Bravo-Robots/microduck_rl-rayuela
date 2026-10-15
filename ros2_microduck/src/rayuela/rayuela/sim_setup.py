"""How the simulated duck behaves: which policies are loaded and how,
walking<->standing thresholds, durations, command handling and the cameras.

Used by sim_worker.py, which is only the process around it (control loop,
socket, status line). Runs under the project venv (Python 3.12), so it may use
mujoco/numpy/onnxruntime, never rclpy.
"""

import os
import sys

import mujoco
import numpy as np

from rayuela import board_geometry, kick_command, paths

if str(paths.SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(paths.SCRIPTS_DIR))
from infer_policy import PolicyInference  # noqa: E402

CONTROL_DECIMATION = 4
CONTROL_TIMESTEP = 0.005  # policies trained at 50Hz = decimation(4) * 0.005s
IMAGE_PUBLISH_EVERY_N_CONTROL_STEPS = 3  # ~50Hz / 3 ≈ 16Hz
# The ground-truth top-down feed is redundant now that vision_node rectifies
# its own top-down canvas from the fiducials, and it doubles the per-publish
# cost (a second 1280x720 render + a second 2.76MB frame). Off by default;
# flip to True to get the debug feed on
# /rayuela/topdown_camera_groundtruth/image_raw again.
PUBLISH_TOPDOWN_GROUNDTRUTH = False

# How long a trick policy stays in control before handing back to
# standing/walking. KICK_DURATION_S must match the kick_duration that
# scripts/kick_sweep.py measured board_geometry's kick tables with (2.0).
KICK_DURATION_S = 2.0
ROULADE_DURATION_S = 2.0

# Hysteresis on the walking<->standing switch. PolicyInference compares the
# command magnitude against a SINGLE switch_threshold, so a command sitting
# near it flaps the policy every tick ("Switched to standing (0.147)" /
# "Switched to walking (0.159)" / ...). Since _update_policy_session re-reads
# switch_threshold on every call, raising the bar to ENTER while standing and
# lowering it to EXIT while walking turns that one threshold into a proper
# Schmitt trigger: it takes a deliberate command to start moving, and a
# deliberate stop to settle, with no chatter in between.
# 0.5 (the original default) exceeds VEL_MAX_X=0.3 below, so a pure-forward
# command's norm never crossed it and the duck never left "standing".
SWITCH_ENTER = 0.1   # command magnitude needed to START walking
SWITCH_EXIT = 0.05   # must drop below this to fall back to standing

VEL_MAX_X = 0.3
VEL_MIN_X = -0.3
VEL_MAX_Y = 0.2
VEL_MIN_Y = -0.2
VEL_MAX_ANG = 1.5

HOME_TRUNK_Z = 0.125


def select_kick_policies(policies_dir: str) -> tuple[str, str, bool]:
    """(left onnx, right onnx, speed_commanded).

    Uses the BallKickSpeed policies only when BOTH feet are present:
    PolicyInference has one kick_speed_commanded switch for every kick, so a
    speed policy on one foot and a fixed-speed one on the other would feed
    the fixed one a command it was never trained on.
    """
    speed = [os.path.join(policies_dir, f"ball_kick_speed_{f}.onnx") for f in ("left", "right")]
    if all(os.path.exists(p) for p in speed):
        return speed[0], speed[1], True
    return (
        os.path.join(policies_dir, "ball_kick_left.onnx"),
        os.path.join(policies_dir, "ball_kick_right.onnx"),
        False,
    )


def build_policy(model, data, policies_dir: str, bam_ctrl) -> PolicyInference:
    """PolicyInference with every rayuela policy loaded, already standing."""
    kick_left, kick_right, kick_speed_commanded = select_kick_policies(policies_dir)
    print(f"Kick policies: {'SPEED-COMMANDED' if kick_speed_commanded else 'fixed-speed'} "
          f"({os.path.basename(kick_left)}, {os.path.basename(kick_right)})")
    policy = PolicyInference(
        model, data,
        bam_ctrl=bam_ctrl,
        walking_onnx_path=f'{policies_dir}/alpha_walking.onnx',
        action_scale=1.0,
        delay_min_lag=0,
        delay_max_lag=0,
        standing_onnx_path=f'{policies_dir}/alpha_standing.onnx',
        switch_threshold=SWITCH_ENTER,
        # TRUE, matching infer_policy.py's CLI default. False fed the policy
        # the raw IMU accelerometer instead: same unit magnitude, but
        # contaminated by body acceleration, so accelerating forward tilts the
        # "apparent gravity" backward and the policy leans forward to correct
        # a tilt that isn't there — a slow, head-forward gait that kept ending
        # on the floor.
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
        kick_duration=KICK_DURATION_S,
        roulade_duration=ROULADE_DURATION_S,
    )
    policy.set_vel_cmd(0.0, 0.0, 0.0)
    # Force standing from the very first control step — with both walking and
    # standing sessions loaded, the constructor otherwise defaults
    # current_policy to "walking" (zero command, not a settled stand), which
    # was pitching the duck forward before the first cmd_vel arrived.
    if policy.standing_session is not None:
        policy.current_policy = "standing"
        policy.ort_session = policy.standing_session
        policy._update_command()
    policy.vel_max_x = VEL_MAX_X
    policy.vel_min_x = VEL_MIN_X
    policy.vel_max_y = VEL_MAX_Y
    policy.vel_min_y = VEL_MIN_Y
    policy.vel_max_ang = VEL_MAX_ANG
    return policy


def place_duck_at_home(model, data, policy: PolicyInference) -> int:
    """Put the duck upright at HOME in its default pose. Returns the trunk
    freejoint's qpos address."""
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = int(model.jnt_qposadr[freejoint_id])
    data.qpos[adr:adr + 3] = [0.0, 0.0, HOME_TRUNK_Z]
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    return adr


class SimCameras:
    """Renders the angled camera vision_node works from, and the sim-only
    ground-truth top-down view (see PUBLISH_TOPDOWN_GROUNDTRUTH). Frames are
    HxWx3 uint8 RGB, C-contiguous."""

    def __init__(self, model):
        self._renderer = mujoco.Renderer(
            model, height=board_geometry.CAMERA_HEIGHT, width=board_geometry.CAMERA_WIDTH
        )
        self._angled = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, board_geometry.CAMERA_NAME)
        # This MuJoCo build has no per-fixed-camera orthographic mode, so a
        # free MjvCamera is aimed straight down in orthographic mode instead.
        # Purely a debug feed to sanity-check vision_node's fiducial-homography
        # rectification — a real deployment has no ceiling camera.
        self._topdown = mujoco.MjvCamera()
        self._topdown.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._topdown.orthographic = 1
        self._topdown.lookat = list(board_geometry.TOPDOWN_DEBUG_CAM_LOOKAT)
        self._topdown.distance = board_geometry.TOPDOWN_DEBUG_CAM_DISTANCE
        self._topdown.azimuth = board_geometry.TOPDOWN_DEBUG_CAM_AZIMUTH
        self._topdown.elevation = board_geometry.TOPDOWN_DEBUG_CAM_ELEVATION

    def _render(self, data, camera) -> np.ndarray:
        self._renderer.update_scene(data, camera=camera)
        return np.ascontiguousarray(self._renderer.render())

    def render_angled(self, data) -> np.ndarray:
        return self._render(data, self._angled)

    def render_topdown_debug(self, data) -> np.ndarray:
        return self._render(data, self._topdown)

    def close(self) -> None:
        self._renderer.close()


class VelCommandGate:
    """Applies /rayuela/cmd_vel to PolicyInference.

    control_node republishes at 20Hz whether or not the command changed, and
    PolicyInference.set_vel_cmd prints on every call — that is the wall of
    identical "Vel cmd: ..." lines. Identical re-applies are skipped (they are
    idempotent anyway). current_policy is part of the key on purpose:
    _end_behavior zeroes vel_cmd when a kick/roulade finishes, so without it a
    still-unchanged walk command would never be re-applied and the duck would
    just stand there after a trick.
    """

    def __init__(self, policy: PolicyInference):
        self._policy = policy
        self._last_key = None

    def apply(self, vx: float, vy: float, wz: float) -> None:
        key = (vx, vy, wz, self._policy.current_policy)
        if key == self._last_key:
            return
        self._last_key = key
        # Schmitt trigger (see SWITCH_ENTER/SWITCH_EXIT): the bar to start
        # walking is higher than the bar to keep walking.
        self._policy.switch_threshold = (
            SWITCH_EXIT if self._policy.current_policy == "walking" else SWITCH_ENTER
        )
        self._policy.set_vel_cmd(vx, vy, wz)


def step_behavior_timers(policy: PolicyInference, control_dt: float) -> None:
    """Advance the trick timers by one control step of SIM time.

    Sim time, not wall time: mj_step advances the world by exactly control_dt,
    so the behavior timers have to drain on the same clock. Feeding them
    wall-clock dt made trick duration depend on render and socket load — the
    loop routinely overruns its 20ms budget, and the roulade was being cut off
    after ~0.6s of roll instead of running to completion (0.14m of travel
    instead of 0.57m).
    """
    policy.update_ground_pick_phase(control_dt)
    policy.update_behavior(control_dt)
    # A fall while sitting has to hand control back to the standing policy,
    # or the duck lies there under a policy that cannot get up (see
    # infer_policy.SIT_FALL_TILT_RAD).
    policy.update_sit_fall_watchdog(control_dt)


def handle_behavior_cmd(policy: PolicyInference, name: str, duck_x: float) -> None:
    """Dispatch one /rayuela/behavior_cmd string (see kick_command for the
    kick grammar). ``duck_x``: current trunk x, for kick_casilla."""
    if name == "ground_pick":
        policy.trigger_ground_pick()
    elif name == "toggle_sit":
        policy.toggle_sit()
    elif name == "stop":
        # Not a behavior session: stop_all ends whatever is running (kick,
        # roulade, sit, pick) and drops into the standing policy.
        policy.stop_all()
    elif kick_command.is_kick_command(name):
        try:
            kick, speed = kick_command.parse_kick_command(name, duck_x=duck_x)
        except ValueError as e:
            print(f"Bad kick command '{name}': {e}")
            return
        if speed is not None and not policy.kick_speed_commanded:
            print(f"'{name}': fixed-speed kick policies loaded, target speed ignored")
        print(f"{kick}" + (f" at {speed:.2f} m/s" if speed is not None else ""))
        policy.trigger_behavior(kick, kick_speed=speed)
    elif name in policy.behavior_sessions:
        policy.trigger_behavior(name)
    else:
        print(f"Unknown behavior_cmd '{name}'")
