"""Microduck BallKickSpeed — the ball kick with a COMMANDED strength.

Built on microduck_ball_kick_env_cfg's factory, so DR / obs noise / NaN
guards / BAM wiring stay identical. The only change: the target ball exit
speed is sampled per episode from KICK_SPEED_RANGE and fed to the actor
through the twist vx slot (actor obs index 48), leaving the 61D layout — and
therefore hot-swappability — untouched.

Speed, not distance: the training ball has no rolling resistance and never
stops, so a landing distance is undefined here, while exit speed is what the
kick actually controls and survives a change of floor. The floor-specific
speed->distance map lives on the deployment side (rayuela board_geometry).

Measured with the trained policy in the loop (escena_rayuela.xml, from HOME):

    command  exit speed  off-axis angle   travel
      0.30      0.67       -20 deg         0.46 m
      0.75      1.01       -12 deg         1.09 m
      1.10      1.22        -3 deg         1.73 m
      1.50      1.81       +12 deg         3.47 m

Two findings shaped this cfg. (1) The soft half of the range did not exist —
0.30 commanded came out at 0.67 — because the linear reward is a plateau above
the target, hence the Gaussian. (2) The kick does not go straight and the angle
TURNS WITH THE STRENGTH, so no fixed yaw offset can fix it at deployment, hence
the aim term on both feet. Exit angle matches the angle to where the ball stops
within 3 deg, so aiming the exit velocity is the whole job (no spin term).

Registered as two task IDs (see tasks/__init__.py), one per foot.
"""

import dataclasses
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    STAND_Z,
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
)

# Widest range the command is ever sampled from, LOG-uniformly so every octave
# gets the same experience (the low end is the hard part, and the one casilla 1
# needs). The ceiling was cut from 2.6 after a run where the exit speed
# saturated around 1.76: unfillable commands taught the policy to throw itself
# at them, falling during the kick at every command >= 0.80.
KICK_SPEED_RANGE = (0.25, 1.8)

# Opened in stages: the policy first learns the strengths it can already make,
# then the extremes. Measured starting point: the previous policy leaves the
# foot at 0.5-0.7 m/s for its softest kick, so stage 0 sits around that.
KICK_SPEED_RANGE_STAGES = [
    {"step": 0,         "range": (0.50, 1.20)},
    {"step": 300 * 24,  "range": (0.40, 1.50)},
    {"step": 600 * 24,  "range": (0.30, 1.65)},
    {"step": 900 * 24,  "range": KICK_SPEED_RANGE},
]

# Speed accuracy comes in two terms that trade places. The LINEAR one is the
# bootstrap: it pays from the first touch so the kick gets discovered, but it
# is a plateau above the target and nothing pulls the speed back down. The
# GAUSSIAN peaks only AT the command and is what buys a usable wide range, but
# has no gradient before the kick exists (a still ball scores exp(-8)).
KICK_REWARD_WEIGHT = 12.0
KICK_OVERSHOOT_WEIGHT = -4.0
KICK_GAUSSIAN_WEIGHT = 12.0
KICK_GAUSSIAN_STD = 0.25  # 25% speed error still scores ~0.6

KICK_LINEAR_STAGES = [
    {"step": 0,         "weight": KICK_REWARD_WEIGHT},
    {"step": 400 * 24,  "weight": 8.0},
    {"step": 700 * 24,  "weight": 4.0},
    {"step": 1000 * 24, "weight": 2.0},   # kept small: it is the safety net
]                                          # that keeps "kick at all" paying
KICK_GAUSSIAN_STAGES = [
    {"step": 0,         "weight": 0.0},
    {"step": 400 * 24,  "weight": 4.0},
    {"step": 700 * 24,  "weight": 8.0},
    {"step": 1000 * 24, "weight": KICK_GAUSSIAN_WEIGHT},
]

# Aiming, on BOTH feet, priced as |tan(off-axis angle)| rather than lateral
# speed over the command: the error turns with the commanded strength (-20 deg
# at 0.30, +12 deg at 1.50), so an angle is what makes the pressure identical
# at every strength. Ramped in, per AGENTS.md — an attempt-tax live while a
# hard skill is still being explored makes "do nothing" the argmax. At full
# weight 20 deg costs tan(20 deg) * 8 = 2.9 against the speed term's 12.
# Standing again once the kick is over — the state the deployment hands back
# in. Before this term the policy ended EVERY kick at 41-45 deg of tilt and
# stayed there, because nothing priced the end state (fell_over only fires at
# 70 deg, and a per-step upright average over 5 s still pays for a lean).
# Multiplicative, and gated to the late episode: the swing stays free, the
# aftermath does not. Weight 6.0 matches the additive stand stack it has to
# outvote (upright 2 + legs 2 + neck 1 + height 1) while staying under the
# speed stack, so "kick at all" keeps winning.
KICK_SETTLE_WEIGHT = 6.0
KICK_SETTLE_AFTER_S = 1.5   # the kick swing is over well before this
KICK_SETTLE_TILT_STD = 0.5    # rad; the 41 deg it parks at scores 0.13 —
                              # visible but poor. 0.25 would score 3e-4, i.e.
                              # no gradient at all out of today's behaviour.
KICK_SETTLE_HEIGHT_STD = 0.03
KICK_SETTLE_POSE_STD = 0.4
# Ramped in like the others: a settle requirement live while the kick is still
# being discovered just taxes every attempt.
KICK_SETTLE_STAGES = [
    {"step": 0,         "weight": 0.0},
    {"step": 300 * 24,  "weight": 2.0},
    {"step": 600 * 24,  "weight": 4.0},
    {"step": 900 * 24,  "weight": KICK_SETTLE_WEIGHT},
]

# A lean this far over is a failed kick. The stock 70 deg let the policy park
# at 41 deg for free; 50 deg keeps the swing's transient affordable while
# making that parked lean one nudge from ending the episode.
KICK_FELL_OVER_ANGLE_DEG = 50.0

KICK_AIM_WEIGHT = -8.0
KICK_AIM_STAGES = [
    {"step": 0,         "weight": 0.0},
    {"step": 400 * 24,  "weight": -2.0},
    {"step": 600 * 24,  "weight": -4.0},
    {"step": 800 * 24,  "weight": KICK_AIM_WEIGHT},
]


def make_microduck_ball_kick_speed_env_cfg(
    play: bool = False,
    kick_foot: str = "right",
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_ball_kick_env_cfg(play=play, kick_foot=kick_foot)

    # The twist slot now carries the target speed instead of shape-parity
    # noise. Drawn once per episode: resample on reset only.
    command = cfg.commands["twist"]
    command.resampling_time_range = (1.0e6, 1.0e6)
    # The base kick cfg wraps twist in VelocityCommandCommandOnlyCfg, which
    # adds fields (rel_turn_in_place_envs) KickSpeedCommandCfg doesn't have.
    accepted = {f.name for f in dataclasses.fields(microduck_mdp.KickSpeedCommandCfg)}
    cfg.commands["twist"] = microduck_mdp.KickSpeedCommandCfg(
        **{k: v for k, v in vars(command).items() if k in accepted},
        speed_range=KICK_SPEED_RANGE_STAGES[0]["range"],
    )

    cfg.rewards["ball_forward_velocity"] = RewardTermCfg(
        func=microduck_mdp.ball_forward_velocity_to_command,
        weight=KICK_LINEAR_STAGES[0]["weight"],
        params={"command_name": "twist", "asset_name": "ball"},
    )
    cfg.rewards["ball_speed_gaussian"] = RewardTermCfg(
        func=microduck_mdp.ball_speed_gaussian_to_command,
        weight=KICK_GAUSSIAN_STAGES[0]["weight"],
        params={"command_name": "twist", "asset_name": "ball",
                "std": KICK_GAUSSIAN_STD},
    )
    cfg.rewards["ball_speed_overshoot"] = RewardTermCfg(
        func=microduck_mdp.ball_speed_overshoot_to_command,
        weight=KICK_OVERSHOOT_WEIGHT,
        params={"command_name": "twist", "asset_name": "ball"},
    )
    cfg.rewards["ball_aim"] = RewardTermCfg(
        func=microduck_mdp.ball_kick_aim_error,
        weight=KICK_AIM_STAGES[0]["weight"],
        params={"asset_name": "ball"},
    )

    cfg.rewards["kick_settled_stand"] = RewardTermCfg(
        func=microduck_mdp.kick_settled_stand,
        weight=KICK_SETTLE_STAGES[0]["weight"],
        params={
            "after_s": KICK_SETTLE_AFTER_S,
            "target_height": STAND_Z,
            "height_std": KICK_SETTLE_HEIGHT_STD,
            "tilt_std": KICK_SETTLE_TILT_STD,
            "pose_std": KICK_SETTLE_POSE_STD,
        },
    )
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(
        KICK_FELL_OVER_ANGLE_DEG
    )

    cfg.curriculum["kick_speed_range"] = CurriculumTermCfg(
        func=microduck_mdp.kick_speed_range_curriculum,
        params={"command_name": "twist", "range_stages": KICK_SPEED_RANGE_STAGES},
    )
    for name, stages in (
        ("ball_forward_velocity", KICK_LINEAR_STAGES),
        ("ball_speed_gaussian", KICK_GAUSSIAN_STAGES),
        ("ball_aim", KICK_AIM_STAGES),
        ("kick_settled_stand", KICK_SETTLE_STAGES),
    ):
        cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": name, "weight_stages": stages},
        )
    return cfg


def _runner_cfg(kick_foot: str):
    name = f"ball_kick_speed_{kick_foot}"
    return dataclasses.replace(MicroduckBallKickRlCfg, experiment_name=name, run_name=name)


MicroduckBallKickSpeedRightRlCfg = _runner_cfg("right")
MicroduckBallKickSpeedLeftRlCfg = _runner_cfg("left")
