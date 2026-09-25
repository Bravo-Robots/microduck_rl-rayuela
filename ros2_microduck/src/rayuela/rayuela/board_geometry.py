"""Shared, ROS-free geometry for the rayuela board and the angled camera.

Read off ``tablero_rayuela.xml``: six 0.3 m columns from x = 0.3 to 2.1.
Singles (1, 4, 7) span y = +/-0.15; doubles (2/3, 5/6, 8/9) span y = +/-0.3
split at y = 0. Past x = 2.1 is the cielo (10), a half-disk of radius 0.4.
Facing down the board (+x), a pair's LEFT square (+y) is the lower number.

Nothing here reads the XML live: if the board is regenerated, update
``CASILLAS``, ``CIELO_*`` and ``CAMERA_*`` to match.
"""

from __future__ import annotations

import math

# (casilla_id, x_min, x_max, y_min, y_max), world frame, meters.
# The singles are HALF the width of the doubles — an earlier version had them
# spanning y in [-0.3, 0.3], which counted a ball resting beside square 1/4/7
# (off the drawn board) as a hit on that square.
CASILLAS: list[tuple[int, float, float, float, float]] = [
    (1, 0.3, 0.6, -0.15, 0.15),
    (2, 0.6, 0.9, 0.0, 0.3),
    (3, 0.6, 0.9, -0.3, 0.0),
    (4, 0.9, 1.2, -0.15, 0.15),
    (5, 1.2, 1.5, 0.0, 0.3),
    (6, 1.2, 1.5, -0.3, 0.0),
    (7, 1.5, 1.8, -0.15, 0.15),
    (8, 1.8, 2.1, 0.0, 0.3),
    (9, 1.8, 2.1, -0.3, 0.0),
]

# The cielo: a half-disk, not a rectangle, so it is handled separately.
CIELO_ID = 10
CIELO_CENTER: tuple[float, float] = (2.1, 0.0)
CIELO_RADIUS = 0.4
# Its "centre" is the half-disk's centroid, 4r/(3*pi) out from the flat edge
# — the middle of the shape, not the midpoint of its diameter (which lies ON
# the boundary line with square 8/9).
CIELO_CENTROID: tuple[float, float] = (
    CIELO_CENTER[0] + 4.0 * CIELO_RADIUS / (3.0 * math.pi),
    CIELO_CENTER[1],
)

# angled_cam pos/fovy/resolution, from tablero_rayuela.xml.
CAMERA_NAME = "angled_cam"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOVY_DEG = 58.0
CAMERA_POSITION = (-0.15, -0.9, 1.5)

# The homography maps the GROUND plane, but the ball's visible centroid sits
# one radius above it, so its apparent position is pushed outward along the
# camera->ball ray by H/(H-r). Measured error: 25 mm near the camera to 58 mm
# at the far end, matching this model to 1-2 mm. ball_ground_from_apparent()
# inverts it. Fiducials lie flat and need no correction (0.0 mm measured).
BALL_RADIUS_M = 0.035

# Calibration fiducials: distinct-colour cylinders at known world XY.
# vision_node matches each by nearest colour and uses the 4 correspondences to
# rectify the oblique view. This is the sim2real-portable path — a real
# deployment uses physical markers the same way, whereas segmenting the board
# plate's shape (the previous approach) has no hardware analogue.
FIDUCIALS: dict[str, tuple[float, float, tuple[int, int, int]]] = {
    # name: (world_x, world_y, rgb_0_255)
    "near_right": (-0.15, -0.4, (255, 0, 0)),
    "near_left": (-0.15, 0.4, (0, 255, 0)),
    "far_left": (2.5, 0.4, (0, 102, 255)),
    "far_right": (2.5, -0.4, (255, 0, 255)),
}

# start_spot geom: light-purple/orchid marker at the world origin. The duck
# always starts here and must return here after a casilla trip.
HOME_POSITION: tuple[float, float] = (0.0, 0.0)

# Virtual top-down canvas that vision_node rectifies the angled_cam view
# into (via the fiducial homography): a fixed world-meters-per-pixel crop
# covering the board + all 4 fiducials with margin. canvas_to_world /
# world_to_canvas below are the (trivial, since the canvas is undistorted)
# conversions used once the image itself is rectified.
TOPDOWN_WORLD_X_RANGE: tuple[float, float] = (-0.4, 2.7)
TOPDOWN_WORLD_Y_RANGE: tuple[float, float] = (-0.5, 0.5)
TOPDOWN_PIXELS_PER_METER: float = 400.0

# Sim-only ground-truth top-down view: MuJoCo (this version) has no
# per-fixed-camera orthographic mode, so it's rendered with a free MjvCamera
# in orthographic mode aimed straight down (elevation=-90 => looks along
# world -z, azimuth=90/lookat centered on the fiducial rectangle). Purely a
# debug aid to sanity-check the vision_node rectification above — a real
# deployment has no ceiling camera, so this path doesn't port to hardware.
TOPDOWN_DEBUG_CAM_LOOKAT: tuple[float, float, float] = (1.175, 0.0, 0.0)
TOPDOWN_DEBUG_CAM_DISTANCE: float = 2.2
TOPDOWN_DEBUG_CAM_AZIMUTH: float = 90.0
TOPDOWN_DEBUG_CAM_ELEVATION: float = -90.0


def topdown_canvas_size() -> tuple[int, int]:
    """(width, height) in pixels of the rectified top-down canvas."""
    x_min, x_max = TOPDOWN_WORLD_X_RANGE
    y_min, y_max = TOPDOWN_WORLD_Y_RANGE
    width = round((x_max - x_min) * TOPDOWN_PIXELS_PER_METER)
    height = round((y_max - y_min) * TOPDOWN_PIXELS_PER_METER)
    return width, height


def world_to_canvas(x: float, y: float) -> tuple[float, float]:
    x_min, _ = TOPDOWN_WORLD_X_RANGE
    _, y_max = TOPDOWN_WORLD_Y_RANGE
    px = (x - x_min) * TOPDOWN_PIXELS_PER_METER
    py = (y_max - y) * TOPDOWN_PIXELS_PER_METER  # image rows grow downward
    return px, py


def canvas_to_world(px: float, py: float) -> tuple[float, float]:
    x_min, _ = TOPDOWN_WORLD_X_RANGE
    _, y_max = TOPDOWN_WORLD_Y_RANGE
    x = px / TOPDOWN_PIXELS_PER_METER + x_min
    y = y_max - py / TOPDOWN_PIXELS_PER_METER
    return x, y


def ball_ground_from_apparent(x: float, y: float) -> tuple[float, float]:
    """Undo the height parallax on a ball position read off the ground-plane
    homography (see BALL_RADIUS_M). Returns where the ball actually is.

    The camera, the ball's centre and the apparent ground point are colinear,
    so scaling the apparent offset from the camera by (H - r)/H walks back
    down that ray to the ball's true XY.
    """
    cam_x, cam_y, cam_z = CAMERA_POSITION
    scale = (cam_z - BALL_RADIUS_M) / cam_z
    return cam_x + (x - cam_x) * scale, cam_y + (y - cam_y) * scale


def casilla_for_point(x: float, y: float) -> int | None:
    """Return the casilla id (1-10) containing world point (x, y), or None."""
    for casilla_id, x_min, x_max, y_min, y_max in CASILLAS:
        if x_min <= x < x_max and y_min <= y < y_max:
            return casilla_id
    cx, cy = CIELO_CENTER
    if x >= cx and (x - cx) ** 2 + (y - cy) ** 2 <= CIELO_RADIUS ** 2:
        return CIELO_ID
    return None


def casilla_center(casilla_id: int) -> tuple[float, float]:
    if casilla_id == CIELO_ID:
        return CIELO_CENTROID
    for cid, x_min, x_max, y_min, y_max in CASILLAS:
        if cid == casilla_id:
            return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    raise ValueError(f"unknown casilla_id {casilla_id}")


# ── Kick strength (BallKickSpeed policies) ───────────────────────────────────
# The kick takes a speed command; how far the ball then carries depends on the
# policy AND the floor, so the map lives here rather than in the policy.
#
KICK_BALL_OFFSET_X = 0.09  # ball starts this far ahead of the trunk (infer_policy BALL_OFFSET_X)
# MEASURED with the trained policies in the loop, one sweep per foot:
# `uv run python scripts/kick_sweep.py --foot right`. Each row is
#     (command, travel of the ball, angle to where it stopped)
# The command rides in the twist vx slot; it is NOT the exit speed, which comes
# out 1.0-1.2x higher. One table PER FOOT — they are not mirror images, and a
# shared table always left one foot miscalibrated. Re-measure after ANY
# retraining and on the real floor: this is a property of the policy and the
# surface, not of the board.
KICK_COMMAND_TO_TRAVEL_RIGHT: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.324, -15.5), (0.35, 0.661, -6.8), (0.45, 0.823, -4.1),
    (0.55, 1.020,  -1.7), (0.70, 1.211,  0.2), (0.85, 1.528,  1.5),
    (1.00, 1.697,   0.0), (1.20, 2.166, -1.7), (1.45, 2.682, -5.3),
    (1.70, 3.575,  -0.2),
)
# The LEFT foot is the RIGHT foot's policy MIRRORED (scripts/mirror_policy.py,
# policies-v1/ball_kick_speed_left_mirrored.onnx). The separately trained left
# policy drifted +15 to +20 deg mid-range, putting casillas 5 and 8 off the
# board; mirrored it drifts +0.2 to +4.7 deg and the whole board is reachable.
KICK_COMMAND_TO_TRAVEL_LEFT: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.381, 13.7), (0.35, 0.685, 7.0), (0.45, 0.839, 4.9),
    (0.55, 1.052,  3.2), (0.70, 1.251, 1.5), (0.85, 1.603, 0.2),
    (1.00, 1.734,  2.0), (1.20, 2.179, 4.7), (1.45, 2.712, 7.5),
    (1.70, 3.536,  3.6),
)
KICK_TABLES = {
    "kick_right": KICK_COMMAND_TO_TRAVEL_RIGHT,
    "kick_left": KICK_COMMAND_TO_TRAVEL_LEFT,
}
KICK_FEET = tuple(KICK_TABLES)
# Ball spawn offset at the kicking foot (infer_policy BALL_OFFSET_ABS_Y).
KICK_BALL_OFFSET_Y = 0.042
# Softest kick each foot can make. Below it the command clamps and the ball
# overshoots, so a casilla nearer than this is only reachable by the OTHER
# foot — which is why casilla 1 (0.36 m of travel) needs one of them to be
# able to tap: the right foot manages 0.324 m, the left 0.454 m.
KICK_MIN_REACHABLE_TRAVEL = {f: t[0][1] for f, t in KICK_TABLES.items()}


def _interp(table: tuple, want: float, col_in: int, col_out: int) -> float:
    """Piecewise-linear lookup between two columns of a kick table, clamped.
    Assumes both columns increase with the row (checked by the sweep)."""
    if want <= table[0][col_in]:
        return table[0][col_out]
    for a, b in zip(table, table[1:]):
        if want <= b[col_in]:
            f = (want - a[col_in]) / (b[col_in] - a[col_in])
            return a[col_out] + f * (b[col_out] - a[col_out])
    return table[-1][col_out]


def kick_speed_for_travel(travel_m: float, foot: str = "kick_right") -> float:
    """Speed COMMAND that carries the ball ``travel_m`` with ``foot``."""
    return _interp(KICK_TABLES[foot], travel_m, 1, 0)


def predict_kick(foot: str, travel_m: float) -> tuple[float, float, float]:
    """(command, x where the ball stops, y where it stops) for ``foot``,
    kicking from HOME. Asking for less than the foot's minimum does not make
    it kick softer — the command clamps — so the predicted landing is what the
    ball ACTUALLY does, not what was asked for.
    """
    table = KICK_TABLES[foot]
    command = kick_speed_for_travel(travel_m, foot)
    travel = _interp(table, command, 0, 1)
    drift = math.radians(_interp(table, command, 0, 2))
    y0 = KICK_BALL_OFFSET_Y if foot == "kick_left" else -KICK_BALL_OFFSET_Y
    return command, HOME_POSITION[0] + KICK_BALL_OFFSET_X + travel, \
        y0 + travel * math.tan(drift)


def kick_for_casilla(casilla_id: int, duck_x: float | None = None) -> tuple[str, float]:
    """(kick behaviour name, speed command) to land the ball on ``casilla_id``.

    The foot is chosen by running BOTH kicks through the measured tables and
    keeping the one that lands on the square — not by the sign of the
    casilla's y, since neither kick goes straight enough for that to hold.

    Measured reach from HOME, all ten covered:
        1  right 0.26 | 2  left  0.34 | 3  right 0.35 | 4  right 0.52
        5  left  0.70 | 6  right 0.72 | 7  right 0.88 | 8  left  1.06
        9  right 1.07 | 10 right 1.21
    ``kick_lands_on_casilla`` re-checks that prediction for a given board.
    """
    cx, cy = casilla_center(casilla_id)
    start_x = (HOME_POSITION[0] if duck_x is None else duck_x) + KICK_BALL_OFFSET_X
    want = cx - start_x

    best = None
    for foot in KICK_FEET:
        command, x, y = predict_kick(foot, want)
        on_target = casilla_for_point(x, y) == casilla_id
        # Prefer a foot that actually lands on the square; otherwise the one
        # that gets closest to its centre.
        score = (0 if on_target else 1, math.hypot(x - cx, y - cy))
        if best is None or score < best[0]:
            best = (score, foot, command)
    return best[1], best[2]


def kick_lands_on_casilla(casilla_id: int) -> bool:
    """Whether the chosen kick is predicted to land ON ``casilla_id`` (False
    for 5 and 8 with the current policies — see kick_for_casilla)."""
    foot, _ = kick_for_casilla(casilla_id)
    cx, cy = casilla_center(casilla_id)
    _, x, y = predict_kick(foot, cx - HOME_POSITION[0] - KICK_BALL_OFFSET_X)
    return casilla_for_point(x, y) == casilla_id


def camera_intrinsics(
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    fovy_deg: float = CAMERA_FOVY_DEG,
) -> tuple[float, float, float, float]:
    """Pinhole (fx, fy, cx, cy) in pixels for the angled camera."""
    fy = height / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    fx = fy  # square pixels
    return fx, fy, width / 2.0, height / 2.0
