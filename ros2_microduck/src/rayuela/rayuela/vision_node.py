"""Watches the angled_cam feed, rectifies it to a top-down view via the 4
color fiducials, finds the ball there, and reports which casilla it lands on.

Pipeline: detect the 4 known-color fiducial cylinders in the raw oblique
frame (nearest-color match, not fragile per-color HSV tuning) and use their
known world XY to compute a pixel(angled)->pixel(top-down canvas)
homography once; warp the whole frame into that canvas (this is the "vista
virtual ortogonal" — segmenting a rectified top-down image gives an
undistorted, circular ball blob and an accurate centroid, unlike segmenting
the raw oblique view and only transforming the resulting point). Ball
detection then runs on the rectified canvas, where pixel->world is a fixed
affine (board_geometry.canvas_to_world).

Landing detection is armed by a kick on /rayuela/behavior_cmd and resolves
when the ball actually STOPS — its canvas centroid staying inside
SETTLE_PIXEL_TOL for SETTLE_WINDOW_S — rather than after a fixed delay, so a
hard kick and a soft one each report as soon as their ball comes to rest.
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

        self.vis_rectify_img = True  # show the rectified top-down canvas in a window for debugging
        self.vis_raw_img = True  # show the raw angled camera feed in a window for debugging    
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
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 10:
            return None
        m = cv2.moments(largest)
        if m["m00"] == 0:
            return None
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

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

        if self.vis_rectify_img:
            try:
                cv2.imshow("Rayuela - Top-down (rectified)", canvas_bgr)
                cv2.waitKey(1)
            except cv2.error:
                pass  # headless mode, no GUI available

        ball_px = self._find_ball_pixel(canvas_bgr)
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
        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
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
