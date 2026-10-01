"""Microduck one-legged tasks: STAND on one foot, then HOP on it.

Two tasks, one curriculum:

* ``Mjlab-OneLeggedStand-Flat-MicroDuck`` — balance on the right foot, the
  left one tucked up (a flamingo). Cheap to train, easy to inspect, and the
  prerequisite for everything else.
* ``Mjlab-OneLeggedHop-Flat-MicroDuck`` — the same stand phase for the first
  STAND_PHASE_ITERS iterations, then flight of the support foot, then forward
  speed.

A second finding, from evaluating the first Stand run: it HAD learned to
stand, and was being knocked off by pushes sized for walking (see
PUSH_STAGES). Both share one experiment directory, and mjlab restores the
  curriculum step counter on resume, so the hop can be warm-started straight
  from a stand checkpoint and picks up exactly where the stand phase ends.

Built on the velocity recipe, so DR / obs noise / NaN guards / BAM wiring stay
identical and the 61D obs layout is unchanged.

Why the first attempt (4000 iterations) learned nothing, measured from its
curves and cfg — each point is addressed below:

1. It never SAW a one-legged state: episodes started on two feet and ended
   ~1 s later on the swing-foot termination, every time. -> reset_one_leg_stance.
2. Its pose term pinned hip_roll with std 0.05 rad: a one-leg stance is ~10
   std away, a flat zero with no gradient. -> a Gaussian toward the STANCE.
3. Half of its positive reward (velocity tracking at 2.0 + 2.0) paid for
   standing still on two feet. -> cut.

The stance itself (servo order below) was measured with a floating base and
the support foot FLAT: the leg below the hip has only pitch joints, so the
trunk has to lean — 27 deg of roll, which puts the support hip_roll AT its
-22 deg limit. That alone leaves the CoM 1.2 mm inside the edge of the 49x37 mm
sole; turning the head (yaw -90, roll -12) and the tucked leg (yaw +30) widens
it to 5.8 mm. Open-loop, the head-assisted stance stays one-legged for 0.7-1.0 s
before toppling (without the head the free foot is back down in 0.05 s) — no
static equilibrium, but a real window for feedback to act in.
"""

import dataclasses
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import _JOINT_PERM, _JOINT_SIGN

# ── The measured stance (RIGHT foot supports) ───────────────────────────────
# Servo order: 0-4 left leg, 5-8 neck/head, 9-13 right leg.
STANCE_RIGHT = (
    0.5236, -0.3491, -1.5051, -1.0521, 1.5002,   # left leg: tucked up, yaw +30
    0.3491, 0.3491, -1.5708, -0.2094,            # head: yaw -90, roll -12
    0.0000, -0.3839, 0.4579, 0.0049, -0.4530,    # right leg: hip_roll at -22 limit
)
STANCE_BASE_ROLL = math.radians(27.0)   # trunk roll with the support foot flat
STANCE_BASE_Z = 0.1255                  # measured 0.1225, +3 mm to settle cleanly

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# ── Spawn mix ───────────────────────────────────────────────────────────────
# Mostly one-legged at first, so the balance gets learned; the two-foot share
# grows once it exists, because deployment ALWAYS starts from a two-foot stand
# and the lift into the stance has to be trained, not assumed.
STANCE_SPAWN_STAGES = [
    {"step": 0,        "params": {"prob": 0.8}},
    {"step": 600 * 24, "params": {"prob": 0.5}},
]

# ── Stand phase rewards ─────────────────────────────────────────────────────
STANCE_POSE_WEIGHT = 2.0
STANCE_POSE_STD = 0.6       # generous: the tucked leg starts ~1 rad away
SWING_GROUNDED_WEIGHT = -1.0
SWING_GRACE_S = 1.0         # time a two-foot spawn has to lift the foot
TRACK_LINEAR_WEIGHT = 1.0
TRACK_ANGULAR_WEIGHT = 0.5
# The stance REQUIRES 27 deg of roll, which the stock upright (std 0.22) pays
# almost nothing for (~0.01). Widened so it still discourages real falls
# without punishing the one posture that works.
UPRIGHT_WEIGHT = 1.0
UPRIGHT_STD = 0.7
# The head is the reason this works (margin 1.2 -> 5.8 mm), so nothing may pin
# it to a pose; kept alive at a token weight for the obs contract.
HEAD_FREE_WEIGHT = 0.05
# Motion-blockers, kept LOW: balancing and hopping need trunk rotation and
# angular momentum, and a head swing IS angular momentum (AGENTS.md).
BODY_ANG_VEL_WEIGHT = -0.01
ANGULAR_MOMENTUM_WEIGHT = -0.002

# ── Pushes ──────────────────────────────────────────────────────────────────
# The velocity recipe shoves at up to 0.3 m/s every 3-6 s. A walker recovers
# by STEPPING; on one leg, stepping IS putting the swing foot down, which ends
# the episode. Measured on the first Stand run (model_1499), per push size:
#     0      -> 81% of episodes reach the 20 s timeout, median 20.0 s
#     0.05   -> 73%        0.10 -> 49%        0.15 -> 15%
#     0.20   ->  2%        0.30 ->  0%, median 4.1 s  (<- what it trained on)
# So it HAD learned the stand; the pushes were knocking it off, 94-96% of the
# time inward (toward the swing foot, where the support hip_roll is already at
# its limit and cannot lean further). Pushes are kept, but in the range a
# one-legged stance can actually absorb. Staged so a fine-tune resumed from a
# 1500-iteration stand walks through them in order.
def _push(v: float) -> dict:
    return {"x": (-v, v), "y": (-v, v)}


PUSH_STAGES = [
    {"step": 0,          "velocity_range": _push(0.03)},
    {"step": 1600 * 24,  "velocity_range": _push(0.06)},
    {"step": 1900 * 24,  "velocity_range": _push(0.10)},
]

# ── Hop phase (Hop task only) ───────────────────────────────────────────────
# Starts after the stand has trained through the last push stage. A hop run
# warm-started from a stand checkpoint continues from that checkpoint's
# iteration, so this must match how long the stand actually trained: with the
# fine-tune below (1500 + 800) it lands exactly here. Starting too early would
# switch on full flight weight AND forward speed at once, skipping the
# hop-in-place stage.
STAND_PHASE_ITERS = 2300
_H = STAND_PHASE_ITERS * 24
HOP_FLIGHT_MIN_S = 0.04     # a hop, not a leap
HOP_FLIGHT_MAX_S = 0.30
HOP_FLIGHT_STAGES = [
    {"step": 0,                "weight": 0.0},
    {"step": _H,               "weight": 2.0},
    {"step": _H + 500 * 24,    "weight": 4.0},
]
# The support leg has to flex to jump, so the stance target lets go of it.
STANCE_POSE_HOP_STAGES = [
    {"step": 0,                "weight": STANCE_POSE_WEIGHT},
    {"step": _H,               "weight": 1.0},
    {"step": _H + 500 * 24,    "weight": 0.5},
]
# Forward speed only once hopping in place exists.
HOP_SPEED_STAGES = [
    {"step": 0,                "lin_vel_range": 0.02, "ang_vel_range": 0.1},
    {"step": _H + 500 * 24,    "lin_vel_range": 0.12, "ang_vel_range": 0.2},
    {"step": _H + 1100 * 24,   "lin_vel_range": 0.25, "ang_vel_range": 0.3},
]
HOP_TRACK_LINEAR_WEIGHT = 2.0
# A small hop is exactly how a one-legged stance recovers a shove too big to
# lean against, so larger pushes come in only once hopping exists.
HOP_PUSH_STAGES = PUSH_STAGES + [
    {"step": _H + 1100 * 24,   "velocity_range": _push(0.15)},
]


def mirror_stance(stance: tuple) -> tuple:
    """The same stance on the OTHER foot (tables from tasks/symmetry.py)."""
    return tuple(stance[p] * s for p, s in zip(_JOINT_PERM, _JOINT_SIGN))


def _foot_sensor(name: str, foot: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode="geom", pattern=rf"^{foot}_foot_collision$",
                             entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )


def make_microduck_onelegged_env_cfg(
    play: bool = False,
    rough: bool = False,
    hop: bool = False,
    support_foot: str = "right",
) -> ManagerBasedRlEnvCfg:
    assert support_foot in ("right", "left")
    swing_foot = "left" if support_foot == "right" else "right"
    stance = STANCE_RIGHT if support_foot == "right" else mirror_stance(STANCE_RIGHT)
    roll = STANCE_BASE_ROLL if support_foot == "right" else -STANCE_BASE_ROLL

    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        _foot_sensor("support_contact", support_foot),
        _foot_sensor("swing_contact", swing_foot),
    )

    # ── Spawn ───────────────────────────────────────────────────────────────
    cfg.events["reset_one_leg_stance"] = EventTermCfg(
        func=microduck_mdp.reset_one_leg_stance,
        mode="reset",
        params={
            "stance": stance,
            "base_roll": roll,
            "base_z": STANCE_BASE_Z,
            "prob": STANCE_SPAWN_STAGES[0]["params"]["prob"],
        },
    )
    cfg.curriculum["stance_spawn"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={"event_name": "reset_one_leg_stance",
                "param_stages": STANCE_SPAWN_STAGES},
    )

    # ── Stand on one foot ───────────────────────────────────────────────────
    stance_legs = {i: stance[i] for i in _LEG_JOINTS}
    cfg.rewards["stance_pose"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=STANCE_POSE_WEIGHT,
        params={"std": STANCE_POSE_STD, "joint_indices": _LEG_JOINTS,
                "target_overrides": stance_legs},
    )
    cfg.rewards["swing_foot_lifted"] = RewardTermCfg(
        func=microduck_mdp.single_foot_grounded_reward,
        weight=SWING_GROUNDED_WEIGHT,
        params={"sensor_name": "swing_contact"},
    )
    # Touching down on the swing foot ends the episode: as a mere cost, two
    # feet are cheaper than one and the task dissolves into standing.
    cfg.terminations["swing_foot_down"] = TerminationTermCfg(
        func=microduck_mdp.swing_foot_touchdown,
        params={"sensor_name": "swing_contact", "grace_s": SWING_GRACE_S},
        time_out=False,
    )

    # ── What the velocity recipe pays for, and this task must not ───────────
    cfg.rewards["pose"].weight = 0.0            # pins hip_roll to HOME (std 0.05)
    for term in ("air_time", "foot_slip", "foot_clearance", "foot_swing_height"):
        if term in cfg.rewards:
            cfg.rewards[term].weight = 0.0      # two-footed gait terms
    for term in ("head_pose_tracking", "head_pose_bias"):
        if term in cfg.rewards:
            w = cfg.rewards[term].weight
            # Only ever loosen: a term the recipe already zeroed stays zero.
            cfg.rewards[term].weight = math.copysign(min(abs(w), HEAD_FREE_WEIGHT), w or 1.0)
    # ...and stop the recipe from pinning it again mid-run: its curriculum
    # ramps head_pose_bias 0 -> 1 -> 2 -> 3 at iterations 600/1000/1500, which
    # would clamp the head to its command exactly when the policy starts using
    # it to balance. The weight set at build time does not survive that.
    cfg.curriculum.pop("head_pose_bias_weight", None)
    cfg.rewards["upright"].weight = UPRIGHT_WEIGHT
    cfg.rewards["upright"].params["std"] = UPRIGHT_STD
    cfg.rewards["track_linear_velocity"].weight = TRACK_LINEAR_WEIGHT
    cfg.rewards["track_angular_velocity"].weight = TRACK_ANGULAR_WEIGHT
    cfg.rewards["body_ang_vel"].weight = BODY_ANG_VEL_WEIGHT
    cfg.rewards["angular_momentum"].weight = ANGULAR_MOMENTUM_WEIGHT
    # The support hip_roll rests ON its limit in this stance; the stock
    # limit term would push it off and throw away the lean the stance needs.
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*hip_roll).*",)
    )

    cfg.events["push_robot"].params["velocity_range"] = PUSH_STAGES[0]["velocity_range"]
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={"event_name": "push_robot", "push_stages": PUSH_STAGES},
    )

    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.0, HOP_SPEED_STAGES[0]["lin_vel_range"])
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-HOP_SPEED_STAGES[0]["ang_vel_range"],
                                HOP_SPEED_STAGES[0]["ang_vel_range"])

    if not hop:
        return cfg

    # ── Hop (after STAND_PHASE_ITERS) ───────────────────────────────────────
    cfg.rewards["hop_flight"] = RewardTermCfg(
        func=microduck_mdp.hop_flight_reward,
        weight=HOP_FLIGHT_STAGES[0]["weight"],
        params={"support_sensor": "support_contact", "swing_sensor": "swing_contact",
                "min_s": HOP_FLIGHT_MIN_S, "max_s": HOP_FLIGHT_MAX_S},
    )
    cfg.rewards["track_linear_velocity"].weight = HOP_TRACK_LINEAR_WEIGHT
    cfg.curriculum["hop_flight_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "hop_flight", "weight_stages": HOP_FLIGHT_STAGES},
    )
    cfg.curriculum["stance_pose_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "stance_pose", "weight_stages": STANCE_POSE_HOP_STAGES},
    )
    cfg.curriculum["push_magnitude"].params["push_stages"] = HOP_PUSH_STAGES
    cfg.curriculum["hop_speed"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={"command_name": "twist", "velocity_stages": HOP_SPEED_STAGES,
                "update_lin_vel_y": False, "forward_only": True},
    )
    return cfg


# One experiment directory for both, so a hop run can load a stand checkpoint
# with --agent.load-run ".*_stand".
EXPERIMENT_NAME = "onelegged"
MicroduckOneLeggedStandRlCfg = dataclasses.replace(
    MicroduckRlCfg, experiment_name=EXPERIMENT_NAME, run_name="stand")
MicroduckOneLeggedHopRlCfg = dataclasses.replace(
    MicroduckRlCfg, experiment_name=EXPERIMENT_NAME, run_name="hop")
