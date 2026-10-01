"""Walks the duck to the centre of whatever casilla vision_node reports.

Go-to-point controller on ground-truth duck pose from sim_node (a real robot
would swap that input for its own odometry). ``/rayuela/go_home`` (Bool true)
overrides the active target and sends the duck back to HOME.

Three rules shape everything below, each measured:

* **The gait has dead zones.** Commands the duck cannot act on look exactly
  like a frozen controller, so every command here is bang-bang: at or above a
  measured floor, or a clean zero. Tapering to zero is what made it circle
  near the target instead of arriving.
* **There is no fall-recovery policy.** ``alpha_standing`` rights itself, so a
  fallen duck gets zero commands and time, never a command into the ground.
* **``current_policy``** (mirrored from the sim) is authoritative. Steering
  only means something in ``AMBULATORY_POLICY_STATES``; anywhere else this
  node publishes an explicit zero.
"""

import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool, String

from rayuela import board_geometry
from rayuela_msgs.msg import TargetCasilla

CONTROL_DT_S = 0.05
POS_TOLERANCE_M = 0.08
MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 2.0
MIN_ANG_VEL = 0.4

# ── Measured gait limits ────────────────────────────────────────────────────
# Forward, holding a constant vx for 8 s (escena_rayuela.xml):
#     cmd 0.08-0.20 -> 0.000 m/s | 0.25 -> 0.078 | 0.30 -> 0.107 | 0.50 -> 0.215
# So WALK_MIN_VEL must clear BOTH the 0.20 dead zone and sim_worker's
# SWITCH_ENTER (0.15), below which the sim never leaves the standing policy.
WALK_MIN_VEL = 0.30
# Turning IN PLACE (vx=0): 0.4 -> 0.011 rad/s, 0.8 -> 0.018, 1.5 -> 0.77. An
# in-place turn is therefore bang-bang at 1.5; a proportional command stalls
# as the error shrinks and the branch never exits (this froze _rotate_home).
# Turning WHILE walking is a different regime and works from ~0.4.
TURN_IN_PLACE_CMD = 1.5
TURN_IN_PLACE_TOL_RAD = math.radians(2.0)  # ~1 tick of travel at 0.77 rad/s

# ── Heading PD (steering while walking only) ────────────────────────────────
TURN_KP = 1.5
TURN_KD = 0.4
HEADING_D_FILTER = 0.3
# The P and D terms run on a FILTERED error: walking at 0.5 the trunk yaw
# carries +/-4.1 deg of oscillation at ~2.6 Hz (the gait's step frequency,
# unavoidable). Sampled at 20 Hz the D term differentiates it and pumps the
# command at step rate. This filter sits below 2.6 Hz; it costs ~0.2 s of lag.
HEADING_ERR_FILTER = 0.25
# Small errors command a clean zero. The old law used ONLY the D term under
# 15 deg — a pure differentiator whose entire output was that oscillation, and
# which left 5-15 deg with no restoring action (the duck drifted +70 deg in
# 10 s of walking straight).
HEADING_DEADBAND_RAD = math.radians(5.0)
# Rate limit on the published angular command: 12 rad/s^2 -> 3. Measured over
# one trajectory it cuts the mean tick-to-tick jump 0.078 -> 0.029 rad/s and
# sign changes 7 -> 3 at the same mean magnitude (1.53 -> 1.52). Not applied to
# the linear command, which is binary by design.
ANG_SLEW_PER_TICK = 0.15

# ── Approach geometry ───────────────────────────────────────────────────────
# Badly misaligned: turn in place until facing the target again.
ALIGN_TURN_ENTER_RAD = math.radians(30.0)
ALIGN_TURN_EXIT_RAD = math.radians(5.0)
# Inside this radius the bearing is ill-conditioned (a few cm of lateral error
# is tens of degrees), so _final_approach alternates turn-in-place and walking
# straight with NO steering, with hysteresis between the two.
FINAL_APPROACH_M = 0.3
FINAL_TURN_ENTER_RAD = math.radians(20.0)
FINAL_TURN_EXIT_RAD = math.radians(8.0)

# ── Roulade finish ──────────────────────────────────────────────────────────
ENABLE_ROULADE_FINISH = True
# Measured forward travel of one roll from a settled stand: +0.559 m in
# scene_ball.xml, +0.567 m in escena_rayuela.xml — scene-independent.
ROULADE_ROLL_DISTANCE_M = 0.57
# Fire when the roll would LAND ON the casilla, not when the distance matches
# the roll length: a casilla is 0.30 m long, and casilla 1 sits 0.48 m from
# HOME, already too close to ever match 0.57 m. Tolerance = how far from the
# centre the landing may be.
ROULADE_LANDING_TOLERANCE_M = 0.12
ROULADE_ALIGN_TOLERANCE_RAD = math.radians(5.0)
# Backing up when closer than one roll-length. Reverse is the weakest thing
# this gait does: 0.000 m/s at -0.30, ~0.08 m/s at -0.40 while veering. So it
# is capped in distance and time, and on failure the roll is dropped for this
# target and the duck walks in. The cap covers every start from 0.31 m out.
REVERSE_VEL = -0.4
REVERSE_MAX_M = 0.20
REVERSE_TIMEOUT_S = 5.0
REVERSE_EXIT_TOLERANCE_M = 0.06
# Stand still before turning and before rolling: the in-place turn fights
# residual motion, and the roll launches from whatever pose it finds. These
# publish explicit zeros across ticks — NEVER time.sleep, which blocks the
# single-threaded executor while the sim keeps acting on the last command (in
# "orient", a 1.5 rad/s spin: one second of sleep adds ~42 deg and destroys
# the alignment it was meant to protect).
REVERSE_SETTLE_S = 1.0
ROULADE_PRE_SETTLE_S = 1.0
# The roulade hands back to standing after 2.0 s while the duck is still
# tumbling, so hold "mid-trick" longer than that or the tilt check reads the
# tumble as a fall. The wait timeout must stay above the hold.
ROULADE_HOLD_S = 4.0
ROULADE_WAIT_TIMEOUT_S = 8.0

# ── Fall handling ───────────────────────────────────────────────────────────
FALL_TILT_RAD = math.radians(60)
# alpha_standing rights itself from face-down in ~6-8 s, but NOT monotonically:
# tilt dips under 60 deg and swings back past it mid-recovery. So resume only
# after it holds a much stricter angle continuously, or the node interrupts the
# recovery and causes another fall.
RECOVERY_SETTLE_RAD = math.radians(20)
RECOVERY_SETTLE_S = 1.5

# Steering only means something here. "standing" must be included: it is the
# state a big-enough command switches OUT of, so blocking it would deadlock.
AMBULATORY_POLICY_STATES = ("standing", "walking")
# Tricks that are SUPPOSED to tip the trunk past FALL_TILT_RAD — exempt from
# fall detection, and they must still reach the roulade_wait bookkeeping.
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
        self._pd_err_filt = 0.0
        self._pd_last_t = None
        # Last angular command actually published (see ANG_SLEW_PER_TICK).
        self._last_ang_cmd = 0.0
        self._final_turning = False
        self._aligning = False
        self._reversing = False
        self._reverse_start = None
        self._reverse_start_error = 0.0
        self._settle_until = None
        # "No (more) rolling for this target": set by a roll that has already
        # been attempted, or by a reverse that could not get into range. Reset
        # on every new target and on arrival.
        self._skip_roulade = False
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
        self.create_subscription(String, "/rayuela/behavior_cmd", self._on_behavior_cmd, 10)

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
        # A new casilla gets a fresh approach: the reverse latch and the
        # "this one is not worth rolling to" verdict both belonged to the
        # previous target.
        self._reversing = False
        self._settle_until = None
        self._aligning = False
        self._final_turning = False
        self._skip_roulade = False
        self.get_logger().info(
            f"New target: casilla {msg.casilla_id} at "
            f"({msg.center.x:.2f}, {msg.center.y:.2f})"
        )

    def _on_behavior_cmd(self, msg: String) -> None:
        """"stop" (teleop X) cancels the target and go-home.

        The sim drops into standing on its own; this node's job is to stop
        steering, or its next tick would set the duck walking again. It ends
        with an explicit zero because idle publishes nothing and the sim holds
        the last command it got — a steering command already in flight would
        otherwise be the one it keeps.
        """
        if msg.data.strip() != "stop":
            return
        self._target = None
        self._go_home = False
        self._home_oriented = False
        self._approach_state = "arrived"
        self._reversing = False
        self._settle_until = None
        self._aligning = False
        self._final_turning = False
        self._skip_roulade = False
        self.now_roulade = None
        self.pub_vel_cmd(0.0, 0.0)
        self.get_logger().info("Stop — target and go-home cancelled")

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

        The P and D terms both run on the FILTERED error (HEADING_ERR_FILTER),
        so the controller does not chase the gait's own +/-4 deg yaw
        oscillation, and small errors command a clean zero
        (HEADING_DEADBAND_RAD) instead of the old pure-derivative branch. The
        published command is additionally rate-limited in pub_vel_cmd.
        """
        now = self.get_clock().now()
        gap = None if self._pd_last_t is None else (now - self._pd_last_t).nanoseconds * 1e-9
        if gap is None or gap > 3 * CONTROL_DT_S:
            # First call, or resuming after a fall / trick hold: no usable
            # history, and a stale sample would inject a huge fake rate.
            self._pd_err_rate = 0.0
            self._pd_err_filt = heading_error
        else:
            # Filter the error itself (see HEADING_ERR_FILTER), wrapping the
            # step so the EMA cannot be dragged the long way round at +/-pi.
            step = math.atan2(
                math.sin(heading_error - self._pd_err_filt),
                math.cos(heading_error - self._pd_err_filt),
            )
            self._pd_err_filt = math.atan2(
                math.sin(self._pd_err_filt + HEADING_ERR_FILTER * step),
                math.cos(self._pd_err_filt + HEADING_ERR_FILTER * step),
            )
        if gap is not None and gap <= 3 * CONTROL_DT_S:
            d_err = math.atan2(
                math.sin(self._pd_err_filt - self._pd_prev_err),
                math.cos(self._pd_err_filt - self._pd_prev_err),
            ) / max(gap, 1e-3)
            self._pd_err_rate += HEADING_D_FILTER * (d_err - self._pd_err_rate)
        self._pd_prev_err = self._pd_err_filt
        self._pd_last_t = now

        if abs(self._pd_err_filt) < HEADING_DEADBAND_RAD:
            ang_vel = 0.0
        else:
            ang_vel = TURN_KP * self._pd_err_filt + TURN_KD * self._pd_err_rate
        if abs(ang_vel) < MIN_ANG_VEL:
            ang_vel = math.copysign(MIN_ANG_VEL, ang_vel)
        return max(-MAX_ANG_VEL, min(MAX_ANG_VEL, ang_vel))
    
    def _should_reverse(self, roll_error: float) -> bool:
        """Whether to back up so a roll can land on the casilla.

        Latched: entered when the duck is nearer than one roll-length by more
        than ROULADE_LANDING_TOLERANCE_M, left once the roll error is back
        inside the tighter REVERSE_EXIT_TOLERANCE_M — which then starts the
        settle pause (REVERSE_SETTLE_S) before orienting.

        It is allowed to FAIL, and must be: reverse is the weakest thing this
        gait does. If REVERSE_MAX_M of backing up or REVERSE_TIMEOUT_S go by
        without reaching roll range, the roulade is dropped for this target and
        the duck walks in. Without that, this branch would hold the approach
        forever — the same trap that froze _rotate_home and the realign branch.
        """
        if not ENABLE_ROULADE_FINISH or self._skip_roulade:
            return False

        if not self._reversing:
            if roll_error < -ROULADE_LANDING_TOLERANCE_M:
                self._reversing = True
                self._reverse_start = self.get_clock().now()
                self._reverse_start_error = roll_error
                self.get_logger().info(
                    f"Too close to roll by {-roll_error:.2f}m — backing up"
                )
            return self._reversing

        if abs(roll_error) <= REVERSE_EXIT_TOLERANCE_M:
            self._reversing = False
            self._settle_until = self.get_clock().now() + Duration(
                seconds=REVERSE_SETTLE_S
            )
            self.get_logger().info(
                f"Back in roll range ({roll_error:+.2f}m) — settling "
                f"{REVERSE_SETTLE_S:.0f}s before orienting"
            )
            return False

        backed_up = roll_error - self._reverse_start_error
        elapsed = (self.get_clock().now() - self._reverse_start).nanoseconds * 1e-9
        if backed_up > REVERSE_MAX_M or elapsed > REVERSE_TIMEOUT_S:
            self._reversing = False
            self._skip_roulade = True
            self.get_logger().warn(
                f"Backed up {backed_up:.2f}m in {elapsed:.0f}s and still "
                f"{-roll_error:.2f}m too close — walking in instead"
            )
            return False
        return True

    def _settling(self) -> bool:
        """Holding still after a reverse (see REVERSE_SETTLE_S)."""
        if self._settle_until is None:
            return False
        if self.get_clock().now() >= self._settle_until:
            self._settle_until = None
            return False
        return True

    def _turn_in_place(self, heading_error: float) -> tuple[float, float]:
        """(0, +/-TURN_IN_PLACE_CMD): the only in-place turn that moves the duck."""
        return 0.0, math.copysign(TURN_IN_PLACE_CMD, heading_error)

    def _needs_realign(self, heading_error: float) -> bool:
        """Latched: enter at ALIGN_TURN_ENTER_RAD, leave at ...EXIT_RAD."""
        if self._aligning:
            if abs(heading_error) < ALIGN_TURN_EXIT_RAD:
                self._aligning = False
        elif abs(heading_error) > ALIGN_TURN_ENTER_RAD:
            self._aligning = True
        return self._aligning

    def _final_approach(self, distance: float, heading_error: float) -> tuple[float, float]:
        """(lin_vel, ang_vel) for the last FINAL_APPROACH_M to a target.

        Alternates between turning in place and walking straight (see
        FINAL_APPROACH_M). Replaces a proportional taper on lin_vel, whose
        whole output range sat inside the gait's dead zone: the duck circled
        near the target and never arrived.
        """
        if self._final_turning:
            if abs(heading_error) < FINAL_TURN_EXIT_RAD:
                self._final_turning = False
        elif abs(heading_error) > FINAL_TURN_ENTER_RAD:
            self._final_turning = True

        if self._final_turning:
            return self._turn_in_place(heading_error)
        # Facing it: close the gap without steering. Over 24 cm the heading
        # cannot drift far, and chasing the bearing is exactly what spins it.
        return WALK_MIN_VEL, 0.0

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

        if abs(heading_error) > math.radians(30):
            ang_vel = self._heading_pd(heading_error)
            self.pub_vel_cmd(0.0, ang_vel)
            return

        if distance <= (POS_TOLERANCE_M-0.04):
            self.pub_vel_cmd(0.0, 0.0)
            self._home_oriented = False
            self._final_turning = False
            return
        elif distance <= FINAL_APPROACH_M:
            self.pub_vel_cmd(*self._gated(*self._final_approach(distance, heading_error)))
            return

        ang_vel = self._heading_pd(heading_error)
        
        # Full speed or nothing. A proportional taper produced 0.15, 0.14,
        # 0.11... on approach: inside the dead zone AND right on
        # switch_threshold, so the sim flapped walking<->standing every tick.
        # Arrival is the POS_TOLERANCE_M check above, which commands zero.
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
        # Badly misaligned: face the target before walking at it. Bang-bang —
        # see ALIGN_TURN_ENTER_RAD for why a PD here freezes the duck.
        if self._needs_realign(heading_error):
            self.pub_vel_cmd(*self._gated(*self._turn_in_place(heading_error)))
            return

        if distance <= POS_TOLERANCE_M:
            self.pub_vel_cmd(0.0, 0.0)
            self._approach_state = "arrived"  # reset for the next target
            self._final_turning = False
            self._aligning = False
            self._reversing = False
            self._settle_until = None
            self._skip_roulade = False
            self.get_logger().info(
                f"Arrived at casilla {self._target.casilla_id} ({distance:.3f}m)"
            )
            return
        elif distance <= FINAL_APPROACH_M:
            self.pub_vel_cmd(*self._gated(*self._final_approach(distance, heading_error)))
            return

        if self._approach_state == "pre_roll":
            # Standing still between the in-place turn and the roll. Alignment
            # is NOT re-checked when the pause ends: the duck is holding a zero
            # command, so it cannot have turned away, and re-checking would let
            # a few tenths of a degree of drift send it back to orient and
            # around again.
            self.pub_vel_cmd(0.0, 0.0)
            if self._settling():
                return
            self.behavior_pub.publish(String(data="roulade"))
            self._approach_state = "roulade_wait"
            self._roulade_confirmed = False
            self._roulade_wait_start = self.get_clock().now()
            # Arm the hold now, not when current_policy first reports
            # "roulade" — the duck is already committed and the tilt can
            # cross FALL_TILT_RAD before that topic update lands.
            self.now_roulade = self._roulade_wait_start
            return

        if self._approach_state == "roulade_wait":
            self.pub_vel_cmd(0.0, 0.0)
            now = self.get_clock().now()
            if self._current_policy == "roulade":
                self._roulade_confirmed = True
                # ONE roll per target, and it is spent the moment the roll
                # actually starts. A roll that ends far from the casilla used
                # to put the duck back in "straight", which could walk it into
                # roll range again and fire a second one — and a third. The
                # walk-in always works, so a bad roll costs a few seconds
                # instead of an unbounded loop of tumbles.
                self._skip_roulade = True
            elif self._roulade_confirmed:
                self.get_logger().info(
                    f"Roulade finished — now {distance:.3f}m from the casilla; "
                    f"walking the rest (one roll per target)"
                )
                self._approach_state = "straight"
            elif (now - self._roulade_wait_start).nanoseconds * 1e-9 > ROULADE_WAIT_TIMEOUT_S:
                self.get_logger().warn(
                    "Roulade never started (behavior_cmd ignored — something "
                    "else running?) — retrying approach"
                )
                self._approach_state = "straight"
            return

        # Three cases, and only the middle one rolls:
        #   too far  -> walk forward until a roll would land on the casilla
        #   in range -> orient, then roll
        #   too near -> back up until it would land on the square again
        #               (REVERSE_VEL); if the reverse does not progress, give
        #               up on the roll and walk in.
        roll_error = distance - ROULADE_ROLL_DISTANCE_M
        roulade_lands_on_target = abs(roll_error) <= ROULADE_LANDING_TOLERANCE_M

        if self._should_reverse(roll_error):
            # Hold the heading while reversing: the gait veers badly backwards,
            # and orient inherits whatever heading this leaves behind.
            self.pub_vel_cmd(*self._gated(REVERSE_VEL, 0.0))
            return

        if self._settling():
            # Explicit zeros, not just "publish nothing": the sim holds the
            # last command it was given, so silence here would keep reversing.
            self.pub_vel_cmd(0.0, 0.0)
            return

        if self._approach_state == "straight":
            if ENABLE_ROULADE_FINISH and roulade_lands_on_target \
                    and not self._skip_roulade:
                self._approach_state = "orient"
            else:
                # Walk the whole way in (no taper — see WALK_MIN_VEL's
                # dead-zone note; scaling down on approach stops the duck
                # short). This is also the path when the roll would overshoot.
                ang_vel = self._heading_pd(heading_error)
                self.pub_vel_cmd(*self._gated(MAX_LIN_VEL, ang_vel))
                return

        if self._approach_state == "orient":
            if not roulade_lands_on_target:
                # Drifted while turning and a roll would now miss the square.
                # Walk (forward — there is no usable reverse); if the duck is
                # now TOO NEAR, "straight" walks it in without rolling.
                self._approach_state = "straight"
                self.pub_vel_cmd(*self._gated(MAX_LIN_VEL,
                                              self._heading_pd(heading_error)))
                return
            if abs(heading_error) <= ROULADE_ALIGN_TOLERANCE_RAD:
                self.get_logger().info(
                    f"Aligned at {distance:.3f}m — settling "
                    f"{ROULADE_PRE_SETTLE_S:.0f}s, then rolling "
                    f"(covers ~{ROULADE_ROLL_DISTANCE_M:.2f}m)"
                )
                self._approach_state = "pre_roll"
                self._settle_until = self.get_clock().now() + Duration(
                    seconds=ROULADE_PRE_SETTLE_S
                )
                self.pub_vel_cmd(0.0, 0.0)
                self.pub_vel_cmd(0.0, 0.0)
            else:
                # Bang-bang like _rotate_home: this is an in-place turn, and a
                # proportional command stalls below ~1.5 rad/s (see
                # TURN_IN_PLACE_CMD). Same latent bug that froze _rotate_home.
                self.pub_vel_cmd(*self._gated(*self._turn_in_place(heading_error)))

    def pub_vel_cmd(self, lin_vel: float, ang_vel: float) -> None:
        """Publish cmd_vel, rate-limiting the ANGULAR component.

        The walking policy expects a roughly-held setpoint, not a value that
        can flip between two 50 ms ticks; see ANG_SLEW_PER_TICK. The linear
        command is passed through unlimited on purpose — it is binary by
        design (WALK_MIN_VEL), and ramping it would only spend ticks inside the
        gait's dead zone.

        A commanded zero is honoured immediately rather than ramped: every
        stop in this node (arrival, fall, trick hold, gating) is a safety
        stop, and sliding down to it through the dead zone would leave the
        duck walking for another few ticks.
        """
        if ang_vel == 0.0:
            self._last_ang_cmd = 0.0
        else:
            delta = ang_vel - self._last_ang_cmd
            self._last_ang_cmd += max(-ANG_SLEW_PER_TICK,
                                      min(ANG_SLEW_PER_TICK, delta))

        cmd = Twist()
        cmd.linear.x = lin_vel
        cmd.angular.z = self._last_ang_cmd
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
