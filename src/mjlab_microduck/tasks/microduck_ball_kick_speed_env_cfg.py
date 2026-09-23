"""Microduck BallKickSpeed — the ball kick with a COMMANDED strength.

Same task as microduck_ball_kick_env_cfg (built on its factory, so DR / obs
noise / NaN guards / BAM wiring stay identical), except the target ball exit
speed is no longer the constant BALL_TARGET_SPEED: it is sampled per episode
from KICK_SPEED_RANGE and fed to the actor through the twist vx command slot
(actor obs index 48). The 61D obs layout is unchanged, so the policy stays
hot-swappable with the rest of the family.

Why a speed command and not a distance: the training ball (ball.xml, default
condim 3) has no rolling resistance and never stops, so "how far it goes" is
undefined here, while exit speed is well defined, is what the kick actually
controls, and survives a change of floor. The floor-specific speed->distance
map lives on the deployment side (see rayuela board_geometry).

The range and the two accuracy problems were measured with the trained
policy in the loop (escena_rayuela.xml, ball launched from HOME):

    command  exit speed  off-axis angle   travel
      0.30      0.67       -20 deg         0.46 m
      0.75      1.01       -12 deg         1.09 m
      1.10      1.22        -3 deg         1.73 m
      1.50      1.81       +12 deg         3.47 m

Two things that shaped this cfg. (1) The soft half of the range does not
exist: 0.30 commanded comes out at 0.67, because the linear reward is a
plateau above the target -> hence the Gaussian. (2) The kick does not go
straight, and the angle TURNS WITH THE STRENGTH, so it cannot be corrected
with a fixed yaw offset at deployment -> hence the aim term, on both feet.
The angle of the ball at exit matches the angle to where it stops within
3 deg, so aiming the exit velocity is the whole job (no spin term needed).

Train both feet — a right-foot ball travels along y~-0.04 and can only land on
1,3,4,6,7,9,10; the left foot covers 1,2,4,5,7,8,10. Registered as two task IDs
(see tasks/__init__.py) rather than flipping the KICK_FOOT module flag.
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
)

# Widest range the command is ever sampled from. Reaching 2.6 m/s is for a
# board LONGER than today's rayuela (the current one needs ~1.8); the low end
# is the hard part and the one that matters for casilla 1. Sampled
# LOG-uniformly (see KickSpeedCommand._resample_command) so every octave gets
# the same amount of experience.
KICK_SPEED_RANGE = (0.25, 2.6)

# Opened in stages: the policy first learns the strengths it can already make,
# then the extremes. Measured starting point: the previous policy leaves the
# foot at 0.5-0.7 m/s for its softest kick, so stage 0 sits around that.
KICK_SPEED_RANGE_STAGES = [
    {"step": 0,         "range": (0.50, 1.50)},
    {"step": 300 * 24,  "range": (0.40, 2.00)},
    {"step": 600 * 24,  "range": (0.30, 2.40)},
    {"step": 900 * 24,  "range": KICK_SPEED_RANGE},
]

# Speed accuracy comes in two terms that trade places.
#
# LINEAR (ball_forward_velocity): min(fwd, tgt)/tgt. It is the bootstrap — it
# pays from the very first touch, so the kick gets discovered — but it is a
# PLATEAU: at or above target it pays full and nothing pulls the speed back
# down. That plateau is why the trained policy answers a 0.30 command with a
# 0.67 m/s ball.
#
# GAUSSIAN (ball_speed_gaussian): peaks only AT the command, with a std that
# is a fraction of it. It is what actually buys a usable wide range, but on
# its own it has no gradient before the kick exists (a still ball scores
# exp(-8)). So: start on the linear term, hand over to the Gaussian.
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

# Aiming, now on BOTH feet. The term is |tan(off-axis angle)| (see
# ball_kick_aim_error), not lateral speed over the command: measured, the
# error turns with the commanded strength (-20 deg at 0.30, +12 deg at 1.50),
# so it is an angle problem and pricing it as an angle is what makes the
# pressure identical at every strength.
#
# Ramped in rather than live from step 0: per AGENTS.md an attempt-tax active
# while a hard skill is still being explored makes "do nothing" the argmax.
# At full weight a 20 deg kick costs tan(20 deg) * 8 = 2.9 against the 12 of
# the speed term — visible, but it can never beat not kicking.
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

    cfg.curriculum["kick_speed_range"] = CurriculumTermCfg(
        func=microduck_mdp.kick_speed_range_curriculum,
        params={"command_name": "twist", "range_stages": KICK_SPEED_RANGE_STAGES},
    )
    for name, stages in (
        ("ball_forward_velocity", KICK_LINEAR_STAGES),
        ("ball_speed_gaussian", KICK_GAUSSIAN_STAGES),
        ("ball_aim", KICK_AIM_STAGES),
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
