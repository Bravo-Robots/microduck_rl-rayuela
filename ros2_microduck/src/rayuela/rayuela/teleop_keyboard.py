"""Keyboard teleop over ROS2: publishes /rayuela/cmd_vel and
/rayuela/behavior_cmd instead of driving PolicyInference directly, so a
human can steer the duck (and manually trigger kicks) through the same
topics rayuela_control_node uses. Terminal reader lifted from run_env.py's
TerminalInput.
"""

import os
import queue
import select
import sys
import termios
import threading
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String

VEL_STEP_X = 0.1
VEL_STEP_Y = 0.1
# MEASURED on the walking policy (escena_rayuela.xml, command held 8-12 s,
# net motion after the first 2 s). These are not preferences — below each
# floor the duck simply does not move, and a command that does nothing looks
# exactly like a frozen teleop.
#
# FORWARD: 0.20 -> 0.000 m/s, 0.25 -> 0.078, 0.30 -> 0.107, 0.50 -> 0.215.
VEL_MAX_X = 0.3
VEL_MAX_Y = 0.2
# TURN IN PLACE: 0.60 -> 0.002 rad/s, 0.80 -> 0.002, 1.00 -> 0.008 (ten
# degrees in eight seconds — this was the old value, and it is why turning
# looked stuck), then a cliff: 1.20 -> 0.527, 1.50 -> 0.739, 2.00 -> 1.112.
# 1.5 is past the trained command range (the velocity task samples ang_vel_z
# in +/-1.0), so this is extrapolation — but it is the same 1.5 the control
# node already uses for every in-place turn, for the same measured reason.
# Turning WHILE walking is a different regime and works from ~0.4.
VEL_MAX_ANG = 1.5
# BACKWARD is NOT a threshold problem, it is a gap in the policy: -0.10 to
# -0.30 all give exactly 0.000 m/s, and only at -0.40 (the edge of the trained
# range, lin_vel_x in +/-0.4) does it start to move — at 0.08 m/s while veering
# 1.5 m sideways over 10 s, i.e. it curves more than it reverses. Commanding
# the trained limit at least makes the key do something; walking backwards
# properly needs training, not a bigger number.
VEL_MIN_X = -0.4


class TerminalInput:
    """Single-keypress reader on stdin (cbreak mode, background thread).

    Arrow keys arrive as ESC [ A/B/C/D escape sequences and are translated to
    symbolic names ("up"/"down"/"left"/"right"); letters are lowercased.
    cbreak (not raw) mode keeps ISIG enabled, so Ctrl+C still works.
    """

    _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left", "Q": "quit", "X": "stop"}

    def __init__(self):
        self._queue = queue.Queue()
        self.enabled = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self.enabled else -1
        self._old_attrs = None
        self._stop = threading.Event()

    def __enter__(self):
        if not self.enabled:
            print("WARNING: stdin is not a TTY — keyboard control disabled")
            return self
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def _read1(self, timeout):
        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return None
        data = os.read(self._fd, 1)
        return data.decode(errors="ignore") if data else None

    def _reader(self):
        while not self._stop.is_set():
            ch = self._read1(0.1)
            if not ch:
                continue
            if ch == "\x1b":
                if self._read1(0.05) == "[":
                    final = self._read1(0.05)
                    name = self._ARROWS.get(final) if final else None
                    if name:
                        self._queue.put(name)
                continue
            self._queue.put(ch.lower() if ch.isalpha() else ch)

    def get_keys(self):
        keys = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                return keys


class RayuelaTeleopKeyboard(Node):
    def __init__(self):
        super().__init__("rayuela_teleop_keyboard")
        self.cmd_pub = self.create_publisher(Twist, "/rayuela/cmd_vel", 10)
        self.behavior_pub = self.create_publisher(String, "/rayuela/behavior_cmd", 10)
        self.go_home_pub = self.create_publisher(Bool, "/rayuela/go_home", 10)
        self.vel_cmd = [0.0, 0.0, 0.0]
        print(
            "\nrayuela teleop — arrows: walk, space: stop, "
            "k/l: kick left/right, r: roulade, g: ground_pick, y: toggle_sit, "
            "h: home, x: stop, 1-9/0: kick to casilla 1-9/10, q: quit\n"
        )
    
    def handle_key(self, key: str) -> bool:
        """Returns False on quit request."""
        if key == "up":
            self.vel_cmd[0] = VEL_MAX_X
        elif key == "down":
            self.vel_cmd[0] = VEL_MIN_X
        elif key == "left":
            self.vel_cmd[2] = VEL_MAX_ANG
        elif key == "right":
            self.vel_cmd[2] = -VEL_MAX_ANG
        elif key == " ":
            self.vel_cmd = [0.0, 0.0, 0.0]
        elif key in ("k", "l", "r", "g", "y", "x"):
            name = {
                "k": "kick_left",
                "l": "kick_right",
                "r": "roulade",
                "g": "ground_pick",
                "y": "toggle_sit",
                "x": "stop"
            }[key]
            if key == "x":
                # This handler publishes the twist below after the behavior;
                # with the old velocity still in it, that twist would arrive
                # right behind "stop" and set the duck walking again.
                self.vel_cmd = [0.0, 0.0, 0.0]
            self.behavior_pub.publish(String(data=name))
            print(f"Triggered behavior: {name}")
        elif key == "h":
            self.go_home_pub.publish(Bool(data=True))
            print("Go home requested")
        elif key.isdigit():
            casilla = 10 if key == "0" else int(key)
            self.behavior_pub.publish(String(data=f"kick_casilla:{casilla}"))
            print(f"Kick to casilla {casilla}")
        elif key == "q":
            return False
        else:
            return True

        twist = Twist()
        twist.linear.x = self.vel_cmd[0]
        twist.linear.y = self.vel_cmd[1]
        twist.angular.z = self.vel_cmd[2]
        self.cmd_pub.publish(twist)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = RayuelaTeleopKeyboard()
    try:
        with TerminalInput() as term:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                for key in term.get_keys():
                    if not node.handle_key(key):
                        rclpy.shutdown()
                        return
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
