"""Wire protocol between sim_worker.py (runs under the project's venv —
Python 3.12, has mujoco/onnxruntime/bam) and bridge_node.py (a normal rclpy
node — rclpy is only importable under the system's Python 3.10 on this
machine). Stdlib-only on purpose: this module gets imported by both
interpreters, and neither ROS nor project-venv packages can be assumed
available on the other side.

Framing: [4-byte big-endian length][1-byte msg type][type-specific body],
over a single Unix domain socket. One frame = one message; length covers
type byte + body.
"""

import os
import socket
import struct

SOCKET_PATH = os.environ.get("RAYUELA_BRIDGE_SOCKET", "/tmp/rayuela_sim_bridge.sock")

MSG_POSE = 1          # worker -> bridge: duck ground-truth pose
MSG_IMAGE = 2         # worker -> bridge: one rendered camera frame (rgb8)
MSG_CMD_VEL = 3       # bridge -> worker: velocity command
MSG_BEHAVIOR = 4      # bridge -> worker: behavior_cmd string
MSG_POLICY_STATE = 5  # worker -> bridge: PolicyInference.current_policy string

CAM_ANGLED = 0
CAM_TOPDOWN_DEBUG = 1

_CMD_VEL_FMT = "<3f"  # vx, vy, wz


def send_frame(sock: socket.socket, msg_type: int, body: bytes) -> None:
    payload = bytes([msg_type]) + body
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None  # peer closed
        buf.extend(chunk)
    return bytes(buf)


def read_frame(sock: socket.socket) -> tuple[int, bytes] | None:
    """Blocking read of one frame. Returns (msg_type, body) or None on EOF."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return payload[0], payload[1:]


def pack_pose(stamp_sec: float, x: float, y: float, z: float,
              qx: float, qy: float, qz: float, qw: float) -> bytes:
    return struct.pack("<8f", stamp_sec, x, y, z, qx, qy, qz, qw)


def unpack_pose(body: bytes) -> tuple[float, float, float, float, float, float, float, float]:
    return struct.unpack("<8f", body)


def pack_cmd_vel(vx: float, vy: float, wz: float) -> bytes:
    return struct.pack(_CMD_VEL_FMT, vx, vy, wz)


def unpack_cmd_vel(body: bytes) -> tuple[float, float, float]:
    return struct.unpack(_CMD_VEL_FMT, body)


def pack_string(s: str) -> bytes:
    return s.encode("utf-8")


def unpack_string(body: bytes) -> str:
    return body.decode("utf-8")


# behavior_cmd and policy_state are both plain strings on the wire.
pack_behavior = pack_string
unpack_behavior = unpack_string
pack_policy_state = pack_string
unpack_policy_state = unpack_string


def pack_image(cam_id: int, stamp_sec: float, width: int, height: int, rgb_bytes: bytes) -> bytes:
    header = struct.pack("<Bf HH", cam_id, stamp_sec, width, height)
    return header + rgb_bytes


def unpack_image(body: bytes) -> tuple[int, float, int, int, bytes]:
    header_size = struct.calcsize("<Bf HH")
    cam_id, stamp_sec, width, height = struct.unpack("<Bf HH", body[:header_size])
    return cam_id, stamp_sec, width, height, body[header_size:]
