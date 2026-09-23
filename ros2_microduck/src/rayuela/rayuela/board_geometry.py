"""Shared, ROS-free geometry for the rayuela board and the angled camera.

Cell boundaries below are read off ``tablero_rayuela.xml``: the
``board_hline_*`` geoms are column dividers at x = 0.3, 0.6, ..., 2.1 (six
0.3 m-wide columns). The SINGLE squares (1, 4, 7) are bounded by
``board_edge{1,4,7}_left/right`` at y = +/-0.15; the DOUBLE columns (2/3, 5/6,
8/9) are bounded by ``board_edge{2..9}`` at y = +/-0.3 and split in two by
``board_vline_{23,56,89}`` at y = 0. Beyond x = 2.1 sits the cielo (10): a
half-disk of radius 0.4 centred on (2.1, 0), drawn by ``board_cielo_arc_*``
with ``board_hline_6`` (y in [-0.4, 0.4]) as its diameter.

Numbering, facing down the board (+x): a pair's LEFT square (+y) is the
lower number. The labels drawn on the board (``casilla_label_*`` geoms in
tablero_rayuela.xml) follow the same numbering.

If the board XML is regenerated with different spacing, update ``CASILLAS``,
``CIELO_*`` and ``CAMERA_*`` to match — nothing here reads the XML live.
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

# The ball's visible centroid sits at its CENTRE, one radius above the
# ground — but the fiducial homography maps the ground PLANE. So the ball's
# apparent ground position is pushed away from the camera along the
# camera->ball ray, by a factor H/(H-r). Measured against ground truth: the
# error runs 25mm near the camera to 58mm at the far end of the board, always
# outward, and matches this model to 1-2mm. ball_ground_from_apparent()
# inverts it. Fiducials need no correction (they are flat on the ground) and
# indeed come back with 0.0mm error.
BALL_RADIUS_M = 0.035

# Calibration fiducials: distinct-color cylinders at known world XY (geoms
# in tablero_rayuela.xml). vision_node detects each by nearest-color match
# in the angled_cam view and uses the 4 correspondences to compute a
# pixel->pixel homography that rectifies the oblique view into a top-down
# canvas — this is the sim2real-portable calibration path (a real deployment
# would use physical markers the same way; segmenting the board plate's
# shape, the previous approach, has no real-hardware analog and is fragile).
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
# CAVEAT worth knowing before trusting a casilla choice: the right-foot kick
# does not go straight. Measured ball rest positions are y = -0.26 to -0.38 m
# (the foot offset is only 0.042), i.e. roughly 20 deg off-axis, which is
# wide enough to miss every SINGLE square (|y| <= 0.15) and, at the softer
# commands, to leave the board sideways altogether. Picking the speed only
# controls the distance; aiming needs either a yaw offset before the kick or
# a lateral-accuracy term in the kick training.
KICK_BALL_OFFSET_X = 0.09  # ball starts this far ahead of the trunk (infer_policy BALL_OFFSET_X)
# MEASURED WITH THE TRAINED POLICY IN THE LOOP (ball_kick_speed_right.onnx,
# escena_rayuela.xml): commanded speed -> distance the ball actually travels
# before stopping. This is NOT the ball's exit speed: the policy kicks harder
# than commanded, by ~1.7x at the bottom of the range (0.30 commanded leaves
# the foot at 0.52 m/s) converging to ~1x near 1.3. An earlier version of this
# table mapped exit speed -> travel from launching the ball directly, with no
# robot, which made every casilla command roughly 30-40% too strong.
#
# Re-measure after retraining, and on the real floor, with the same
# policy-in-the-loop sweep (scripts are ad hoc; see the session notes).
KICK_COMMAND_TO_TRAVEL: tuple[tuple[float, float], ...] = (
    (0.30, 0.57), (0.40, 0.71), (0.50, 0.83), (0.60, 0.97),
    (0.75, 1.17), (0.90, 1.35), (1.10, 1.71), (1.30, 2.26),
)
# The softest kick the policy can produce still carries the ball 0.57 m, so
# casilla 1 (0.36 m of travel from the kick spot) is OUT OF REACH: it clamps
# to the floor command and the ball lands around casilla 3.
KICK_MIN_REACHABLE_TRAVEL = KICK_COMMAND_TO_TRAVEL[0][1]


def kick_speed_for_travel(travel_m: float) -> float:
    """Speed COMMAND that lands the ball ``travel_m`` away, by linear
    interpolation of KICK_COMMAND_TO_TRAVEL (clamped at both ends)."""
    table = KICK_COMMAND_TO_TRAVEL
    if travel_m <= table[0][1]:
        return table[0][0]
    for (v0, d0), (v1, d1) in zip(table, table[1:]):
        if travel_m <= d1:
            return v0 + (v1 - v0) * (travel_m - d0) / (d1 - d0)
    return table[-1][0]


def kick_for_casilla(casilla_id: int, duck_x: float | None = None) -> tuple[str, float]:
    """(kick behaviour name, exit speed) to land the ball on ``casilla_id``,
    kicking from ``duck_x`` (default HOME) while facing down the board.

    A kicked ball travels along its foot's side (y ~ -0.04 right, +0.04 left),
    so the +y squares of a pair (2, 5, 8) need the LEFT foot and the -y ones
    (3, 6, 9) the RIGHT; singles and the cielo take either (right here).
    """
    cx, cy = casilla_center(casilla_id)
    foot = "kick_left" if cy > 0 else "kick_right"
    start_x = (HOME_POSITION[0] if duck_x is None else duck_x) + KICK_BALL_OFFSET_X
    return foot, kick_speed_for_travel(cx - start_x)


def camera_intrinsics(
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    fovy_deg: float = CAMERA_FOVY_DEG,
) -> tuple[float, float, float, float]:
    """Pinhole (fx, fy, cx, cy) in pixels for the angled camera."""
    fy = height / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    fx = fy  # square pixels
    return fx, fy, width / 2.0, height / 2.0
