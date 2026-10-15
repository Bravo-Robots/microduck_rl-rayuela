"""Kick commands on /rayuela/behavior_cmd, shared by sim_setup (sim_node) and
vision_node. Stdlib only, so it can be tested without ROS or MuJoCo.

Grammar:
    kick_right / kick_left      fixed-speed kick (the original policies)
    kick_right:1.25             kick with a target ball exit speed, m/s
    kick_casilla:7              pick foot + speed to land on casilla 7
                                (board_geometry.kick_for_casilla)
"""

from rayuela import board_geometry

KICK_FEET = ("kick_left", "kick_right")


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
