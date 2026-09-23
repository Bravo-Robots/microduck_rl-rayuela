"""Kick commands on /rayuela/behavior_cmd, shared by sim_worker, sim_node and
vision_node (stdlib only — imported from both the venv and ROS's Python).

Grammar:
    kick_right / kick_left      fixed-speed kick (the original policies)
    kick_right:1.25             kick with a target ball exit speed, m/s
    kick_casilla:7              pick foot + speed to land on casilla 7
                                (board_geometry.kick_for_casilla)
"""

import os

from rayuela import board_geometry

KICK_FEET = ("kick_left", "kick_right")


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


def is_kick_command(cmd: str) -> bool:
    name = cmd.strip().split(":", 1)[0]
    return name in KICK_FEET or name == "kick_casilla"


def parse_kick_command(cmd: str, duck_x: float) -> tuple[str, float | None]:
    """Resolve a kick command to (behavior name, target speed or None).

    ``duck_x``: current trunk x, for kick_casilla's distance (the duck is
    assumed to face down the board, as it does after going home).
    Raises ValueError on a malformed command.
    """
    name, _, arg = cmd.strip().partition(":")
    if name == "kick_casilla":
        return board_geometry.kick_for_casilla(int(arg), duck_x=duck_x)
    if name in KICK_FEET:
        return name, (float(arg) if arg else None)
    raise ValueError(f"not a kick command: {cmd!r}")
