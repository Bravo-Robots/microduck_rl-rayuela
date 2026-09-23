"""Walks the duck to the center of whatever casilla vision_node reports.

Simple go-to-point controller on ground-truth duck pose (published by
sim_node — a real robot would swap this input for its own odometry node,
out of scope here): turn to face the target, then walk forward, capped at
the walking policy's own velocity limits. Stops within tolerance. No
automatic kick triggering (MVP scope) — that stays a manual pub or
teleop_keyboard.

A `/rayuela/go_home` (Bool, data=true) message overrides whatever casilla
target is active and sends the duck to board_geometry.HOME_POSITION — the
light-purple start_spot marker, which is always where the duck starts and
must return to.

Stability rules, each learned from watching this fall over:
  - The walking policy was trained on commands that are held roughly
    constant for whole resampling intervals (seconds), not recomputed every
    tick. A closed-loop steering law that jumps discontinuously (e.g. an
    if/else that snaps from "turn in place" to "walk+turn" the instant
    heading error crosses a threshold) or has no rate limit feeds it a much
    noisier, faster-changing command than anything in training — so both
    the linear/angular blend below and the published cmd_vel are smoothed.
  - There is no "stand up from the ground" policy in policies-v1 (alpha_
    ground_pick picks something up, it isn't a fall recovery). So if the
    duck is fallen (checked via trunk tilt from duck_pose, not just height),
    the only honest thing to do is stop commanding it and wait — NOT keep
    walking a fallen robot into the ground — until it's upright again
    (recovered by a person, teleop, or a future recovery policy).
  - PolicyInference.current_policy (mirrored here over /rayuela/current_policy
    by sim_node/bridge_node) is the sim's authoritative state — "walking",
    "standing", "sit", "ground_pick", "slope", or a behavior name
    ("kick_left", "kick_right", "roulade"). A steering command only means
    something while the duck is "standing" (about to start — a big-enough
    vel_cmd is exactly what PolicyInference._update_policy_session switches
    on) or "walking" (already going); it CANNOT mean "walking exclusively",
    since that would deadlock — nothing would ever be allowed to send the
    command that flips standing -> walking in the first place. Mid-kick,
    sitting, ground_pick, or slope mode, a steering command wouldn't be
    honored anyway and could only interfere, so this node sends an explicit
    zero Twist there instead of a stale/guessed non-zero one.
"""

import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String

from rayuela import board_geometry
from rayuela_msgs.msg import TargetCasilla

POS_TOLERANCE_M = 0.04
# MEASURED dead zone: alpha_walking produces no net forward motion below
# ~0.3 m/s commanded — 0.15 and 0.20 both march in place (0.001 m/s over 10s
# of sim), 0.30 gives 0.092 m/s and 0.40 gives 0.151 m/s. Training marks a
# fraction of envs as standing (is_standing_env in mdp.py), so the policy
# learned to hold still rather than step for small commands. Keep this at
# 0.3 or above, and never taper lin_vel down into the dead zone on approach —
# the duck would simply stop short of the target and never arrive.
MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 2.0
# Taper lin_vel to 0 as distance -> POS_TOLERANCE_M inside this radius, so the
# duck settles into the tolerance circle instead of repeatedly overshooting
# it at full speed and oscillating around the target (go_home path only —
# see the casilla state machine below for the other case).
SLOWDOWN_RADIUS_M = 0.3

CONTROL_DT_S = 0.05  # matches self.timer period below
# Per-tick change caps on the published cmd_vel, so it ramps rather than
# jumps — the walking policy expects a roughly-held setpoint, not a value
# that can flip between "0" and "max" between two 50ms control steps.

# Casilla approach is a 3-stage state machine, not the continuous go-to-point
# law above: walk mostly-straight at fixed speed, then once close switch to
# turning in place to face the casilla center exactly, then hand off to the
# roulade behavior to actually land on it.
#   "straight": lin_vel fixed at MAX_LIN_VEL, only a gentle (STRAIGHT_KP)
#     heading nudge — deliberately not the aggressive TURN_KP correction, so
#     the approach reads as walking straight rather than curving in.
#   "orient": turn in place (lin_vel=0) with TURN_KP until facing the casilla
#     center within ROULADE_ALIGN_TOLERANCE_RAD.
#   "roulade_wait": behavior_cmd "roulade" published once, cmd_vel held at
#     zero (also true anyway once current_policy flips to "roulade" — see
#     AMBULATORY_POLICY_STATES below — but held explicitly here too to cover
#     the tick or two before that topic update lands) until the roulade
#     finishes (current_policy leaves "roulade" again) or ROULADE_WAIT_TIMEOUT_S
#     elapses with it never starting (e.g. another behavior was already
#     running), then back to "straight" to re-approach.
STRAIGHT_KP = 1.5
# Small on purpose, current focus: get the duck to the target without
# falling, and the rotational twist doesn't need to be aggressive here —
# 2.0 was producing a sharp proportional response to heading error;
# tune back up once arrival is reliable.
TURN_KP = 1.5
# Derivative gain on the heading error. Pure P saturated MAX_ANG_VEL for most
# of a big turn and then overshot past zero error and came back — the "it just
# keeps spinning" symptom. The D term bleeds the command off as the error
# closes. There is no yaw-rate feedback to use (duck_pose is a PoseStamped,
# pose only), so the rate is the wrapped tick-to-tick difference of the error,
# EMA-filtered: at 20Hz the raw difference is noisy enough to fight the P term.
TURN_KD = 0.5
HEADING_D_FILTER = 0.3   # EMA weight on the new sample (1.0 = no filtering)
# Linear PD, used inside SLOWDOWN_RADIUS_M. WALK_KP * SLOWDOWN_RADIUS_M is the
# command at the edge of the circle; WALK_KD brakes early when closing fast
# (the error rate is the closing speed, negative while approaching).
WALK_KP = 1.5
WALK_KD = 0.8
WALK_D_FILTER = 0.3
# THE floor that makes this work: below ~0.3 m/s commanded the gait produces
# no net motion at all (0.15 and 0.20 both measured 0.001 m/s). A taper that
# fades smoothly to zero therefore spends its last 15cm issuing commands the
# duck cannot act on, while the heading PD keeps steering — the duck circles
# in place instead of arriving. So the linear command is effectively binary:
# either >= this floor, or a clean zero once inside POS_TOLERANCE_M.
WALK_MIN_VEL = 0.11
# MEASURED forward travel of one roulade, from a settled stand: +0.559 m in
# scene_ball.xml and +0.567 m in escena_rayuela.xml — the two scenes agree, so
# the roll itself is scene-independent. (An earlier "it goes 0.24 m BACKWARD"
# reading was an artifact: the ball used to spawn at the right foot, inside
# the roll path, and the duck bounced off it. See pelota_rayuela.xml.)
ROULADE_ROLL_DISTANCE_M = 0.57
# Fire the roulade only from roughly one roll-length out, so it LANDS on the
# casilla instead of overshooting it. Triggering at the old ORIENT_RADIUS_M
# (0.4 m) overshot by ~0.14 m, and since that left the duck outside
# POS_TOLERANCE_M it just rolled again — oscillating past the target forever.
# Outside this band the duck simply walks, which also closes out whatever the
# roll leaves over.
ROULADE_TRIGGER_BAND_M = 0.07
ROULADE_ALIGN_TOLERANCE_RAD = math.radians(5.0)
# MEASURED: turning in place (vx=0) is dead below ~1.5 rad/s commanded. At
# 0.4 and 0.8 the duck achieves 0.011 and 0.018 rad/s — 3-6 degrees in SIX
# seconds — while 1.5 achieves 0.77 rad/s. (Turning WHILE walking is a
# different story: 0.4 commanded already gives 0.175 rad/s.) So an in-place
# turn has to be commanded bang-bang at this magnitude; a proportional
# ang_vel = kp * heading_error stalls the moment the error shrinks, which is
# exactly why _rotate_home never reached its tolerance and left _go_home
# latched forever, blocking every later casilla target.
TURN_IN_PLACE_CMD = 1.5
# At the achieved ~0.77 rad/s one 50ms control tick covers ~2.2 deg, so a
# tolerance tighter than that just makes it hunt back and forth.
TURN_IN_PLACE_TOL_RAD = math.radians(2.0)
ENABLE_ROULADE_FINISH = True
# Once a roulade starts, treat the duck as "mid-trick" for at least this long
# REGARDLESS of what current_policy says. PolicyInference hands the roulade
# back to standing/walking after its own roulade_duration (2.0s, "~the roll
# itself"), but the duck is still tumbling at that moment — without this latch
# the tilt check below immediately reads the tumble as a fall, latches
# _was_fallen and holds control hostage through the settle. The latch only
# governs THIS node's fall detection and commands; it cannot postpone
# PolicyInference's internal 2s handoff (that lives in sim_node/sim_worker).
ROULADE_HOLD_S = 4.0
# Must stay above ROULADE_HOLD_S, or the "roulade never started" bailout in
# _run_casilla_approach fires while the latch is still legitimately held.
ROULADE_WAIT_TIMEOUT_S = 8.0

# Trunk tilt (angle between the body's up axis and world-up) beyond which
# the duck is considered fallen, not just leaning into a turn.
FALL_TILT_RAD = math.radians(60)
# Recovery from a real fall (measured: alpha_standing rights itself from
# lying face-down in ~6-8s, but NOT monotonically — tilt can dip under 60deg
# then swing back up past it mid-recovery before finally settling). Clearing
# "_was_fallen" the instant tilt first dips below FALL_TILT_RAD was resuming
# real steering commands while the duck was still mid-recovery and
# unstable — interrupting it and often causing another fall. Instead,
# require tilt to stay under this MUCH stricter threshold continuously for
# RECOVERY_SETTLE_S before resuming control.
RECOVERY_SETTLE_RAD = math.radians(20)
RECOVERY_SETTLE_S = 1.5

# Steering commands are meaningful in these two states only: "standing" is
# the pre-walk state a big-enough vel_cmd switches OUT of (blocking it here
# would deadlock — nothing could ever start a walk), "walking" is the gait
# actually running. Every other state (sit/ground_pick/slope/a behavior name
# like kick_left) is busy doing something a cmd_vel can't steer.
AMBULATORY_POLICY_STATES = ("standing", "walking")

# Behaviors that are SUPPOSED to tip the trunk past FALL_TILT_RAD as part of
# the trick (roulade is a barrel roll) — exempt from fall detection, and
# don't early-return past the roulade_wait bookkeeping in
# _run_casilla_approach, or it can never observe current_policy=="roulade"
# to know the trick actually ran.
DYNAMIC_BEHAVIOR_STATES = ("kick_left", "kick_right", "roulade")


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Angle (rad) between the body's local +z axis and world +z."""
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, up_z)))


class RayuelaControlNode(Node):
    def __init__(self):
        super().__init__("rayuela_control_node")

        self._duck_pose: PoseStamped | None = None
        self._target: TargetCasilla | None = None
        self._go_home = False
        self._current_policy: str | None = None
        self._was_fallen = False
        self._pd_prev_err = 0.0
        self._pd_err_rate = 0.0
        self._pd_last_t = None
        self._walk_prev_dist = 0.0
        self._walk_dist_rate = 0.0
        self._walk_last_t = None
        self._recovery_settle_start = None
        self._home_oriented = True

        # Casilla-approach state machine (see constants above).
        self._approach_state = "straight"
        self._roulade_confirmed = False
        self._roulade_wait_start = None
        self.now_roulade = None

        self.cmd_pub = self.create_publisher(Twist, "/rayuela/cmd_vel", 10)
        self.behavior_pub = self.create_publisher(String, "/rayuela/behavior_cmd", 10)
        self.create_subscription(PoseStamped, "/rayuela/duck_pose", self._on_duck_pose, 10)
        self.create_subscription(TargetCasilla, "/rayuela/target_casilla", self._on_target, 10)
        self.create_subscription(Bool, "/rayuela/go_home", self._on_go_home, 10)
        self.create_subscription(String, "/rayuela/current_policy", self._on_current_policy, 10)

        self.timer = self.create_timer(CONTROL_DT_S, self._on_timer)
        self.get_logger().info("rayuela_control_node ready")

    def _on_duck_pose(self, msg: PoseStamped) -> None:
        self._duck_pose = msg

    def _on_current_policy(self, msg: String) -> None:
        self._current_policy = msg.data

    def _on_target(self, msg: TargetCasilla) -> None:
        self._target = msg
        self._approach_state = "straight"
        self._roulade_confirmed = False
        self._roulade_wait_start = None
        self.get_logger().info(
            f"New target: casilla {msg.casilla_id} at "
            f"({msg.center.x:.2f}, {msg.center.y:.2f})"
        )

    def _on_go_home(self, msg: Bool) -> None:
        self._go_home = msg.data
        if self._go_home:
            self._approach_state = "straight"
            # Walk home FIRST, then spin. Without this reset a second go-home
            # would still see the False left by the previous round trip and
            # jump straight to _rotate_home without ever walking back.
            self._home_oriented = True
            self.get_logger().info("Go-home requested — heading to start_spot")

    def _on_timer(self) -> None:
        if self._duck_pose is None:
            return
        
        now = self.get_clock().now()
        if self._current_policy in DYNAMIC_BEHAVIOR_STATES:
            # (Re)arm the hold for as long as the trick is actually running,
            # so the window is measured from when it ENDS, not when it began.
            self.now_roulade = now

        q = self._duck_pose.pose.orientation
        tilt = tilt_from_quat(q.x, q.y, q.z, q.w)

        # The hold exists to stop the tumble being read as a fall, so it ends
        # the moment the duck is demonstrably upright again — ROULADE_HOLD_S is
        # only the cap for a roll that lands badly. Waiting out the full fixed
        # window cost ~4s of the duck just standing there after every roulade.
        holding_for_trick = (
            self.now_roulade is not None
            and (now - self.now_roulade).nanoseconds * 1e-9 < ROULADE_HOLD_S
            and tilt > RECOVERY_SETTLE_RAD
        )
        in_dynamic_behavior = self._current_policy in DYNAMIC_BEHAVIOR_STATES or holding_for_trick

        if holding_for_trick and self._current_policy not in DYNAMIC_BEHAVIOR_STATES:
            # Trick handed back to standing/walking but still tipped over from
            # the roll — keep quiet and let it finish settling.
            self.pub_vel_cmd(0.0, 0.0)
            return

        if not in_dynamic_behavior and tilt > FALL_TILT_RAD:
            if not self._was_fallen:
                self.get_logger().warn(
                    "Duck appears fallen (trunk tilt past threshold) — holding "
                    "cmd_vel at zero so alpha_standing can right itself (measured:"
                    " ~6-8s from flat, not monotonic — it needs the whole window "
                    "undisturbed, so this waits for it to actually settle before "
                    "resuming control, not just dip under the threshold once)."
                )
                self._was_fallen = True
            self._recovery_settle_start = None  # still clearly down, no settling yet
            self.pub_vel_cmd(0.0, 0.0)
            return

        if self._was_fallen and not in_dynamic_behavior:
            if tilt > RECOVERY_SETTLE_RAD:
                # Mid-recovery: under FALL_TILT_RAD but not upright yet — keep
                # holding zero and don't count any settle time yet.
                self._recovery_settle_start = None
                self.pub_vel_cmd(0.0, 0.0)
                return
            if self._recovery_settle_start is None:
                self._recovery_settle_start = self.get_clock().now()
            settled_s = (self.get_clock().now() - self._recovery_settle_start).nanoseconds * 1e-9
            if settled_s < RECOVERY_SETTLE_S:
                self.pub_vel_cmd(0.0, 0.0)
                return
            self.get_logger().info("Duck upright and settled — resuming")
            self._was_fallen = False
            self._recovery_settle_start = None

        if self._approach_state == "arrived":
            return
        
        if self._go_home and self._home_oriented:
            self._run_go_home(q)
            return
        elif self._go_home and not self._home_oriented:
            self._rotate_home(q)
            return

        if self._target is not None:
            self._run_casilla_approach(q)
        # Idle (no target, no go-home): publish NOTHING. /rayuela/cmd_vel is
        # shared with teleop_keyboard, which sends a command once per keypress;
        # streaming zeros here at 20Hz overwrote every key on the next tick, so
        # manual driving flapped walking->standing and never moved. Silence is
        # safe: PolicyInference holds the last command it got, which at startup
        # is sim_worker's explicit zero, and every path into idle goes through
        # an arrival branch that already sends its own final stop.

    def _heading_pd(self, heading_error: float) -> float:
        """PD on the heading error -> angular velocity command (clamped).

        Only for steering WHILE WALKING. An in-place turn still has to be
        bang-bang at TURN_IN_PLACE_CMD: below ~1.5 rad/s the duck does not
        rotate at all standing still, so a PD there would stall at whatever
        small command it converges to.
        """
        now = self.get_clock().now()
        gap = None if self._pd_last_t is None else (now - self._pd_last_t).nanoseconds * 1e-9
        if gap is None or gap > 3 * CONTROL_DT_S:
            # First call, or resuming after a fall / trick hold: no usable
            # history, and a stale sample would inject a huge fake rate.
            self._pd_err_rate = 0.0
        else:
            d_err = math.atan2(
                math.sin(heading_error - self._pd_prev_err),
                math.cos(heading_error - self._pd_prev_err),
            ) / max(gap, 1e-3)
            self._pd_err_rate += HEADING_D_FILTER * (d_err - self._pd_err_rate)
        self._pd_prev_err = heading_error
        self._pd_last_t = now

        if abs(heading_error) > math.radians(15.0):
            ang_vel = TURN_KP * heading_error + TURN_KD * self._pd_err_rate
        else:
            ang_vel = TURN_KD * self._pd_err_rate

        return max(-MAX_ANG_VEL, min(MAX_ANG_VEL, ang_vel))
    
    def _walking_pd(self, distance: float) -> float:
        """PD on the distance to target -> forward velocity command.

        Far away the command just saturates at MAX_LIN_VEL; the PD only does
        something inside SLOWDOWN_RADIUS_M, where it eases off (and the D term
        brakes earlier the faster the duck is closing) so it stops near the
        target instead of coasting past it and having to come back.

        The output is then snapped to the gait's dead zone (see WALK_MIN_VEL):
        anything the duck cannot act on becomes either the floor or a clean
        zero. Without that the tail of the taper is a command that walks
        nowhere while the heading PD keeps turning — the duck circling near
        the target rather than reaching it.
        """
        now = self.get_clock().now()
        gap = None if self._walk_last_t is None else (now - self._walk_last_t).nanoseconds * 1e-9
        if gap is None or gap > 3 * CONTROL_DT_S:
            # First call, or resuming after a fall / trick hold: a stale sample
            # would inject a huge fake closing rate.
            self._walk_dist_rate = 0.0
        else:
            d_rate = (distance - self._walk_prev_dist) / max(gap, 1e-3)
            self._walk_dist_rate += WALK_D_FILTER * (d_rate - self._walk_dist_rate)
        self._walk_prev_dist = distance
        self._walk_last_t = now

        lin_vel = WALK_KP * distance + WALK_KD * self._walk_dist_rate
        lin_vel = max(0.0, min(MAX_LIN_VEL, lin_vel))
        if lin_vel < WALK_MIN_VEL:
            lin_vel = 0.0 if distance <= POS_TOLERANCE_M else WALK_MIN_VEL
        return lin_vel

    def _gated(self, lin_vel: float, ang_vel: float) -> tuple[float, float]:
        """Zero out a steering command if the sim isn't in a state that can
        act on it (sit/ground_pick/slope/a kick or roulade in progress)."""
        if self._current_policy is not None and self._current_policy not in AMBULATORY_POLICY_STATES:
            return 0.0, 0.0
        return lin_vel, ang_vel

    def _rotate_home(self, q) -> None:
        """Spin in place at home until facing down the board (world +x)."""
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        heading_error = math.atan2(math.sin(-yaw), math.cos(-yaw))  # target yaw = 0

        if abs(heading_error) <= TURN_IN_PLACE_TOL_RAD:
            self.pub_vel_cmd(0.0, 0.0)
            self._go_home = False
            self._home_oriented = True  # ready for the next go-home round trip
            self._approach_state = "arrived"
            self.get_logger().info(
                f"Arrived home and facing the board ({math.degrees(heading_error):+.1f} deg off)"
            )
            return
        # Bang-bang, NOT proportional — see TURN_IN_PLACE_CMD.
        ang_vel = math.copysign(TURN_IN_PLACE_CMD, heading_error)
        self.pub_vel_cmd(*self._gated(0.0, ang_vel))

    def _run_go_home(self, q) -> None:
        target_x, target_y = board_geometry.HOME_POSITION
        target_x -= 0.03
        dx = target_x - self._duck_pose.pose.position.x
        dy = target_y - self._duck_pose.pose.position.y
        distance = math.hypot(dx, dy)

        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        target_yaw = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)
        )
        
        if distance <= POS_TOLERANCE_M:
            self.pub_vel_cmd(0.0, 0.0)
            self._home_oriented = False
            return
        elif distance > POS_TOLERANCE_M and distance <= POS_TOLERANCE_M + 0.2:
            ang_vel = self._heading_pd(heading_error)
            lin_vel = self._walking_pd(distance)
            self.pub_vel_cmd(*self._gated(lin_vel, ang_vel))
            return

        ang_vel = self._heading_pd(heading_error)
        
        # Full speed or nothing — NOT a proportional taper. STRAIGHT_KP*distance
        # produced 0.15, 0.14, 0.11... as it closed in, which is doubly useless:
        # those are inside the gait's dead zone (no net motion below ~0.3, so
        # the duck just crawled) AND they sit right on switch_threshold, so
        # PolicyInference flapped walking<->standing every tick. Arrival is the
        # POS_TOLERANCE_M check above, which commands a clean zero.
        self.pub_vel_cmd(*self._gated(MAX_LIN_VEL, ang_vel))

    def _run_casilla_approach(self, q) -> None:
        target_x, target_y = self._target.center.x, self._target.center.y
        dx = target_x - self._duck_pose.pose.position.x
        dy = target_y - self._duck_pose.pose.position.y
        distance = math.hypot(dx, dy)

        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        target_yaw = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)
        )

        if distance <= POS_TOLERANCE_M:
            self.pub_vel_cmd(0.0, 0.0)
            self._approach_state = "arrived"  # reset for the next target
            self.get_logger().info(
                f"Arrived at casilla {self._target.casilla_id} ({distance:.3f}m)"
            )
            return
        elif distance > POS_TOLERANCE_M and distance <= POS_TOLERANCE_M + 0.2:
            ang_vel = self._heading_pd(heading_error)
            lin_vel = self._walking_pd(distance)
            self.pub_vel_cmd(*self._gated(lin_vel, ang_vel))
            return

        if self._approach_state == "roulade_wait":
            self.pub_vel_cmd(0.0, 0.0)
            now = self.get_clock().now()
            if self._current_policy == "roulade":
                self._roulade_confirmed = True
            elif self._roulade_confirmed:
                self.get_logger().info(
                    f"Roulade finished — now {distance:.3f}m from the casilla"
                )
                self._approach_state = "straight"
            elif (now - self._roulade_wait_start).nanoseconds * 1e-9 > ROULADE_WAIT_TIMEOUT_S:
                self.get_logger().warn(
                    "Roulade never started (behavior_cmd ignored — something "
                    "else running?) — retrying approach"
                )
                self._approach_state = "straight"
            return

        in_roulade_band = abs(distance - ROULADE_ROLL_DISTANCE_M) <= ROULADE_TRIGGER_BAND_M

        if self._approach_state == "straight":
            if ENABLE_ROULADE_FINISH and in_roulade_band:
                self._approach_state = "orient"
            else:
                # Walk the whole way in (no taper — see MAX_LIN_VEL's dead-zone
                # note; scaling down on approach just stops the duck short).
                ang_vel = self._heading_pd(heading_error)
                self.pub_vel_cmd(*self._gated(MAX_LIN_VEL, ang_vel))
                return

        if self._approach_state == "orient":
            if not in_roulade_band:
                # Drifted out of one roll-length while turning — walk again
                # rather than fire a roll that can no longer land on target.
                self._approach_state = "straight"
                self.pub_vel_cmd(*self._gated(MAX_LIN_VEL, 0.0))
                return
            if abs(heading_error) <= ROULADE_ALIGN_TOLERANCE_RAD:
                self.get_logger().info(
                    f"Aligned — triggering roulade at {distance:.3f}m "
                    f"(roll covers ~{ROULADE_ROLL_DISTANCE_M:.2f}m)"
                )
                self.behavior_pub.publish(String(data="roulade"))
                self._approach_state = "roulade_wait"
                self._roulade_confirmed = False
                self._roulade_wait_start = self.get_clock().now()
                # Arm the hold now, not when current_policy first reports
                # "roulade" — the duck is already committed and the tilt can
                # cross FALL_TILT_RAD before that topic update lands.
                self.now_roulade = self._roulade_wait_start
                self.pub_vel_cmd(0.0, 0.0)
            else:
                # Bang-bang like _rotate_home: this is an in-place turn, and a
                # proportional command stalls below ~1.5 rad/s (see
                # TURN_IN_PLACE_CMD). Same latent bug that froze _rotate_home.
                ang_vel = math.copysign(TURN_IN_PLACE_CMD, heading_error)
                self.pub_vel_cmd(*self._gated(0.0, ang_vel))

    def pub_vel_cmd(self, lin_vel: float, ang_vel: float) -> None:

        cmd = Twist()
        cmd.linear.x = lin_vel
        cmd.angular.z = ang_vel
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RayuelaControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
