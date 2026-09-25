"""Rectifies the angled_cam feed to a top-down view, finds the ball, and
reports which casilla it lands on.

The 4 known-colour fiducial cylinders give a one-off homography from the
oblique frame to a metric top-down canvas, where pixel->world is a fixed
affine (board_geometry.canvas_to_world) and the ball is an undistorted blob
of known size. Detection runs on that canvas, never on the raw view.

Landing detection is armed by a kick on /rayuela/behavior_cmd and resolves
when the ball actually STOPS (centroid inside SETTLE_PIXEL_TOL for
SETTLE_WINDOW_S), so a hard kick and a soft one each report on arrival.
"""

import array
import math
from collections import deque

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from rayuela import board_geometry, kick_command
from rayuela_msgs.msg import TargetCasilla

# Ball rgba = (1, 0.55, 0, 1) -> orange. HSV bounds with generous margin for
# MuJoCo's shading gradient across the sphere.
BALL_HSV_LOW = np.array([5, 120, 120])
BALL_HSV_HIGH = np.array([25, 255, 255])

# The duck is orange too and its geoms land INSIDE this window (measured hues
# 9.6 and 20.1 against the ball's 16.5), so "the biggest orange blob" is not
# the ball. Shape separates them; measured on the canvas, ball on casilla 1 —
# the only square the duck's body can reach:
#                          area    circ   SOLIDITY
#     ball, near to far    1.19x   0.88     0.98
#                          1.85x   0.68     0.96   (stretched by the homography)
#     duck, best blob      0.44x   0.43     0.75
#     duck FUSED with ball 2.01x   0.27     0.69
# Solidity (area / convex-hull area) is the test: the far-end ball is a ~2:1
# ellipse, which ruins circularity (0.68, one hundredth above a 0.65 gate —
# that silently lost a ball on the cielo) but not convexity. The area window
# has to be wide enough for that stretch, which lets the fused blob through on
# size; solidity is what rejects it.
BALL_EXPECTED_AREA_PX = math.pi * (
    board_geometry.BALL_RADIUS_M * board_geometry.TOPDOWN_PIXELS_PER_METER
) ** 2
# Measured shape of every orange blob on the canvas (ball on casilla 1 /
# casilla 6 / the cielo, duck standing and face-down):
#                      area    circ   SOLIDITY   axis ratio
#     ball casilla 1   1.19x   0.88     0.98        0.80
#     ball casilla 6   1.41x   0.72     0.96        0.65
#     ball cielo 2.50  1.85x   0.68     0.96        0.49
#     duck, best blob  0.44x   0.43     0.75        0.54
#     duck FUSED with
#       the ball       2.01x   0.27     0.69        0.39
#
# The ball is NOT round at the far end: the homography stretches it into a
# roughly 2:1 ellipse, and it grows to 1.85x its nominal area. Circularity
# therefore drops to 0.68 out there, one hundredth above a 0.65 threshold —
# that is what silently lost a ball resting on the cielo. An ellipse is still
# CONVEX, though, so solidity (area / convex-hull area) does not care about the
# stretch: worst ball 0.95, best non-ball 0.75. That gap is the test.
#
# The area window has to be wide enough for the far-end stretch (hence 2.5),
# which also lets the fused blob through on size — solidity is what rejects it.
BALL_AREA_RANGE = (0.5, 2.5)
BALL_MIN_SOLIDITY = 0.85

# FALLBACK for the fused case only. On the metric canvas the ball's radius is
# known (14 px), which is what makes Hough usable without parameter fiddling.
# Benchmarked against the contour rule over 4 duck poses x 4 ball positions:
# as a REPLACEMENT it is worse — on grayscale it locks onto the near_left
# fiducial (a circle of the same radius) even with no ball in view, and on the
# mask it finds nothing on the far half of the board, whose edges the
# rectification leaves too soft. As a FALLBACK it earns its place: on the one
# case contours cannot do — duck lying on the ball, blobs fused — it recovers
# the ball to 16 mm. Costs 5.2 ms vs 0.5, and only on frames that found
# nothing. It keeps one false positive of its own, a circle inside the
# standing duck's own orange at world (0.02, 0.14), so the result is only
# accepted ON the board.
BALL_HOUGH_PARAM1 = 120
BALL_HOUGH_PARAM2 = 15
BALL_HOUGH_RADIUS_SCALE = (0.8, 1.3)
BALL_ON_BOARD_MIN_X = 0.30        # casilla 1's near edge
BALL_ON_BOARD_MAX_ABS_Y = 0.45

# Fiducial detection: any saturated colorful blob, then classified by
# nearest RGB match against board_geometry.FIDUCIALS. The generic mask
# excludes the board's near-white/gray lines and plate (low saturation).
FIDUCIAL_HSV_LOW = np.array([0, 100, 100])
FIDUCIAL_HSV_HIGH = np.array([180, 255, 255])
FIDUCIAL_MIN_AREA = 20
# Reject ambiguous color matches (e.g. the orange ball or the light-purple
# home marker) — the 4 fiducial colors are far apart in RGB, so a real
# match is always well under this.
FIDUCIAL_MATCH_MAX_DIST = 80.0

# A kick arms landing detection; the report goes out when the ball actually
# STOPS, not after a fixed wait. "Stopped" = its centroid stayed inside
# SETTLE_PIXEL_TOL for SETTLE_WINDOW_S of continuous tracking. Measured on
# the rectified canvas, which is a fixed board_geometry.TOPDOWN_PIXELS_PER_METER
# affine, so the tolerance converts directly: at 400 px/m, 3 px = 7.5 mm.
SETTLE_WINDOW_S = 1.0
SETTLE_PIXEL_TOL = 3.0
SETTLE_MIN_SAMPLES = 5     # don't call it settled off one or two lucky frames
# Give up if the ball never settles (kicked off the board, lost behind the
# duck, never detected at all) instead of waiting forever.
LANDING_TIMEOUT_S = 20.0
# For the first moments after a kick the ball is still AT THE FOOT, fused with
# the duck, so "no ball-shaped blob" is the expected answer and not worth a
# warning. Measured: the ball is clear well inside a second and even the
# softest kick has it stopped by ~2.2 s.
BALL_WARN_GRACE_S = 2.0

# How the rectified canvas is SHOWN and PUBLISHED: "horizontal" as built, or
# "vertical" rotated 90 deg counter-clockwise (duck at the bottom, cielo at the
# top). Purely a view — detection and every canvas<->world conversion run on
# the unrotated canvas, so this cannot move a landing point.
TOPDOWN_ORIENTATION = "horizontal"

class RayuelaVisionNode(Node):
    def __init__(self):
        super().__init__("rayuela_vision_node")
        self._homography = None  # angled-pixel -> canvas-pixel, set once calibrated
        self._canvas_size = board_geometry.topdown_canvas_size()

        # Landing detection, armed by any kick command (kick_command.is_kick_command).
        self._awaiting_landing = False
        self._kick_stamp = None                 # when the kick armed us
        self._ball_track: deque = deque()       # (t, px, py) over SETTLE_WINDOW_S
        self._last_ball_world: tuple[float, float] | None = None

        self.ball_pub = self.create_publisher(PointStamped, "/rayuela/ball/position", 10)
        self.target_pub = self.create_publisher(TargetCasilla, "/rayuela/target_casilla", 10)
        self.topdown_pub = self.create_publisher(Image, "/rayuela/topdown_camera/image_raw", 10)
        self.create_subscription(String, "/rayuela/behavior_cmd", self._on_behavior_cmd, 10)

        self.vis_rectify_img = True   # debug window with the rectified canvas
        self.vis_raw_img = False      # debug window with the raw angled feed
        
        self._vertical_view = TOPDOWN_ORIENTATION == "vertical"
        self.get_logger().info(f"Top-down view: {TOPDOWN_ORIENTATION}")
        self.create_subscription(Image, "/rayuela/angled_camera/image_raw", self._on_image, 10)

        self.get_logger().info("rayuela_vision_node ready")

    def _find_fiducials(self, frame_bgr: np.ndarray) -> dict[str, tuple[float, float]]:
        """Nearest-color-match centroids for each of the 4 known fiducials."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, FIDUCIAL_HSV_LOW, FIDUCIAL_HSV_HIGH)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: dict[str, tuple[tuple[float, float], float]] = {}
        for c in contours:
            if cv2.contourArea(c) < FIDUCIAL_MIN_AREA:
                continue
            blob_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            mean_b, mean_g, mean_r, _ = cv2.mean(frame_bgr, mask=blob_mask)
            name, dist = self._closest_fiducial((mean_r, mean_g, mean_b))
            if name is None:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])
            if name not in best or dist < best[name][1]:
                best[name] = (centroid, dist)
        return {name: centroid for name, (centroid, _) in best.items()}

    @staticmethod
    def _closest_fiducial(rgb: tuple[float, float, float]) -> tuple[str | None, float]:
        best_name, best_dist = None, FIDUCIAL_MATCH_MAX_DIST
        for name, (_, _, ref_rgb) in board_geometry.FIDUCIALS.items():
            dist = math.dist(rgb, ref_rgb)
            if dist < best_dist:
                best_name, best_dist = name, dist
        return best_name, best_dist
        
    def _calibrate_homography(self, frame_bgr: np.ndarray) -> bool:
        fiducials_px = self._find_fiducials(frame_bgr)
        names = list(board_geometry.FIDUCIALS.keys())
        if not all(name in fiducials_px for name in names):
            return False

        pixel_pts = np.array([fiducials_px[name] for name in names], dtype=np.float32)
        canvas_pts = np.array(
            [board_geometry.world_to_canvas(*board_geometry.FIDUCIALS[name][:2]) for name in names],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(pixel_pts, canvas_pts)
        if H is None:
            return False
        self._homography = H
        self.get_logger().info(f"Top-down homography calibrated from fiducials: {names}")
        return True

    def _find_ball_pixel(self, frame_bgr: np.ndarray) -> tuple[float, float] | None:
        """Centroid of the ball on the rectified canvas, or None.

        Picks the most BALL-SHAPED orange blob, not the biggest one — see
        BALL_MIN_SOLIDITY for why the biggest is often the duck. Returning None
        when the duck is lying on the ball is the intended outcome: a fused
        blob's centroid is not the ball's.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_solidity, rejected = None, 0.0, 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 10:
                continue
            rejected += 1
            ratio = area / BALL_EXPECTED_AREA_PX
            if not (BALL_AREA_RANGE[0] <= ratio <= BALL_AREA_RANGE[1]):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < BALL_MIN_SOLIDITY or solidity <= best_solidity:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            best_solidity = solidity
            best = (m["m10"] / m["m00"], m["m01"] / m["m00"])
            rejected -= 1

        if best is not None:
            return best

        if rejected and self._awaiting_landing:
            hough = self._find_ball_hough(mask)
            if hough is not None:
                return hough
            if self._since_kick() > BALL_WARN_GRACE_S:
                self.get_logger().warn(
                    f"{rejected} orange blob(s) in view, none ball-shaped and no "
                    f"circle recoverable (the duck is orange too, and may be "
                    f"lying on the ball)",
                    throttle_duration_sec=2.0,
                )
        return None

    def _since_kick(self) -> float:
        """Seconds since the kick that armed landing detection (inf if none)."""
        if self._kick_stamp is None:
            return float("inf")
        return (self.get_clock().now() - self._kick_stamp).nanoseconds * 1e-9

    def _find_ball_hough(self, mask: np.ndarray) -> tuple[float, float] | None:
        """Last resort when the ball's blob is fused with the duck's.

        Hough finds the ball's circular EDGE even where the two silhouettes
        overlap, which no contour rule can do. Only trusted on the board — off
        it lives Hough's own false positive.
        """
        radius = board_geometry.BALL_RADIUS_M * board_geometry.TOPDOWN_PIXELS_PER_METER
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(mask, (5, 5), 1.5),
            cv2.HOUGH_GRADIENT, dp=1, minDist=int(radius * 1.5),
            param1=BALL_HOUGH_PARAM1, param2=BALL_HOUGH_PARAM2,
            minRadius=int(radius * BALL_HOUGH_RADIUS_SCALE[0]),
            maxRadius=int(radius * BALL_HOUGH_RADIUS_SCALE[1]),
        )
        if circles is None:
            return None
        for cx, cy, _r in circles[0]:
            wx, wy = board_geometry.canvas_to_world(float(cx), float(cy))
            if wx >= BALL_ON_BOARD_MIN_X and abs(wy) <= BALL_ON_BOARD_MAX_ABS_Y:
                if self._since_kick() > BALL_WARN_GRACE_S:
                    self.get_logger().info(
                        f"Ball recovered by Hough at ({wx:.2f}, {wy:.2f}) — its "
                        f"blob is fused with the duck's",
                        throttle_duration_sec=2.0,
                    )
                return float(cx), float(cy)
        return None

    def _oriented(self, img: np.ndarray) -> np.ndarray:
        """The canvas as it should be SHOWN (see TOPDOWN_ORIENTATION)."""
        if not self._vertical_view:
            return img
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _oriented_point(self, x: float, y: float, width: int) -> tuple[float, float]:
        """A canvas point in the oriented image's coordinates.

        Verified against cv2.rotate on a marked pixel: a 90 deg
        counter-clockwise rotation of a ``width``-wide image sends (x, y) to
        (y, width - 1 - x).
        """
        if not self._vertical_view:
            return x, y
        return y, width - 1 - x

    def _show_rectified(self, canvas_bgr: np.ndarray, ball_px) -> None:
        """Debug window: rectified canvas with the detected ball centroid."""
        # Rotate FIRST, then draw: the overlay text has to read upright in the
        # vertical view, and the marker is placed through the same transform.
        vis = self._oriented(canvas_bgr)  # copies when rotating
        if vis is canvas_bgr:
            vis = canvas_bgr.copy()       # no tocar el canvas original
        if ball_px is not None:
            ox, oy = self._oriented_point(*ball_px, canvas_bgr.shape[1])
            cx, cy = int(round(ox)), int(round(oy))
            # World coordinates come from the UNROTATED point: the rotation is
            # a view, and must never leak into the geometry.
            wx, wy = board_geometry.canvas_to_world(*ball_px)
            cv2.drawMarker(vis, (cx, cy), (0, 255, 0),
                           cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)         # crosshair
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)                 # centre
            cv2.putText(vis, f"px ({cx}, {cy})  world ({wx:.2f}, {wy:.2f})",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (127,7,111), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "ball: not detected", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
        try:
            cv2.imshow("Rayuela - Top-down (rectified)", vis)
            cv2.waitKey(1)
        except cv2.error:
            pass  # headless mode, no GUI available

    def _on_image(self, msg: Image) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().warn(f"Unsupported image encoding '{msg.encoding}', expected rgb8")
            return
        frame_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if self.vis_raw_img:
            try:
                cv2.imshow("Rayuela - Angled Camera", frame_rgb)
                cv2.waitKey(1)
            except cv2.error:
                pass  # headless mode, no GUI available


        if self._homography is None:
            if not self._calibrate_homography(frame_bgr):
                return

        canvas_bgr = cv2.warpPerspective(frame_bgr, self._homography, self._canvas_size)
        self._publish_topdown(canvas_bgr, msg.header)

        ball_px = self._find_ball_pixel(canvas_bgr)
        if self.vis_rectify_img:
            self._show_rectified(canvas_bgr, ball_px)

        if ball_px is None:
            # Check the give-up clock here too, not only in the tracker: a ball
            # that is never detected (kicked off the board, hidden behind the
            # duck) would otherwise leave us armed forever.
            self._check_landing_timeout()
            return

        # Correct the ball's height parallax — the homography maps the ground
        # PLANE, but the ball's centroid is one radius above it (measured:
        # 25-58mm of outward error without this).
        world_x, world_y = board_geometry.ball_ground_from_apparent(
            *board_geometry.canvas_to_world(*ball_px)
        )
        self._last_ball_world = (world_x, world_y)

        ball_msg = PointStamped()
        ball_msg.header = msg.header
        ball_msg.header.frame_id = "world"
        ball_msg.point = Point(x=world_x, y=world_y, z=0.0)
        self.ball_pub.publish(ball_msg)

        if self._awaiting_landing:
            self._track_until_stopped(*ball_px)

    def _on_behavior_cmd(self, msg: String) -> None:
        # Any form: kick_right, kick_right:1.25, kick_casilla:7.
        if not kick_command.is_kick_command(msg.data):
            return
        self._awaiting_landing = True
        self._kick_stamp = self.get_clock().now()
        self._ball_track.clear()
        self.get_logger().info(
            f"'{msg.data}' received — tracking the ball, will report where it stops "
            f"(centroid within {SETTLE_PIXEL_TOL:.0f}px for {SETTLE_WINDOW_S:.0f}s)"
        )

    def _check_landing_timeout(self) -> bool:
        """Disarm if the ball never settled. Returns True if it gave up."""
        if not self._awaiting_landing:
            return False
        if (self.get_clock().now() - self._kick_stamp).nanoseconds * 1e-9 <= LANDING_TIMEOUT_S:
            return False
        self._awaiting_landing = False
        self.get_logger().warn(
            f"Ball never settled within {LANDING_TIMEOUT_S:.0f}s — giving up on this kick"
        )
        return True

    def _track_until_stopped(self, px: float, py: float) -> None:
        """Called per frame while a kick is in flight. Declares the landing as
        soon as the ball's centroid stops moving, instead of waiting out a
        fixed delay — a hard kick and a soft one no longer take the same time
        to report."""
        if self._check_landing_timeout():
            return
        now = self.get_clock().now()
        elapsed = (now - self._kick_stamp).nanoseconds * 1e-9

        self._ball_track.append((now, px, py))
        while self._ball_track and (now - self._ball_track[0][0]).nanoseconds * 1e-9 > SETTLE_WINDOW_S:
            self._ball_track.popleft()

        # Need a full window of samples before the spread means anything —
        # otherwise the first frame or two always look "stopped".
        if len(self._ball_track) < SETTLE_MIN_SAMPLES:
            return
        span = (self._ball_track[-1][0] - self._ball_track[0][0]).nanoseconds * 1e-9
        if span < SETTLE_WINDOW_S * 0.9:
            return

        xs = [p[1] for p in self._ball_track]
        ys = [p[2] for p in self._ball_track]
        if (max(xs) - min(xs)) > SETTLE_PIXEL_TOL or (max(ys) - min(ys)) > SETTLE_PIXEL_TOL:
            return  # still rolling

        self._awaiting_landing = False
        landing_x, landing_y = board_geometry.ball_ground_from_apparent(
            *board_geometry.canvas_to_world(float(np.mean(xs)), float(np.mean(ys)))
        )
        casilla_id = board_geometry.casilla_for_point(landing_x, landing_y)
        if casilla_id is None:
            self.get_logger().warn(
                f"Ball stopped at ({landing_x:.2f}, {landing_y:.2f}) after {elapsed:.1f}s "
                f"— outside the board, nothing to report"
            )
            return

        center_x, center_y = board_geometry.casilla_center(casilla_id)
        target_msg = TargetCasilla()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = "world"
        target_msg.casilla_id = casilla_id
        target_msg.center = Point(x=center_x, y=center_y, z=0.0)
        target_msg.landing_point = Point(x=landing_x, y=landing_y, z=0.0)
        self.target_pub.publish(target_msg)
        self.get_logger().info(
            f"Ball landed on casilla {casilla_id} at ({landing_x:.2f}, {landing_y:.2f})"
        )

    def _publish_topdown(self, canvas_bgr: np.ndarray, header) -> None:
        canvas_rgb = cv2.cvtColor(self._oriented(canvas_bgr), cv2.COLOR_BGR2RGB)
        height, width, _ = canvas_rgb.shape
        img_msg = Image()
        img_msg.header = header
        img_msg.header.frame_id = "topdown_canvas"
        img_msg.height = height
        img_msg.width = width
        img_msg.encoding = "rgb8"
        img_msg.is_bigendian = 0
        img_msg.step = width * 3
        # array.array('B'), not bytes — see bridge_node._publish_image.
        img_msg.data = array.array('B', np.ascontiguousarray(canvas_rgb).tobytes())
        self.topdown_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RayuelaVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
