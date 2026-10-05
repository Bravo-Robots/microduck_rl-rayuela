"""Where the microduck_rl checkout lives, found at runtime — no hardcoded paths.

Order: the MICRODUCK_RL_ROOT environment variable if set, else walk up from
this file (symlinks resolved, so `colcon build --symlink-install` points back
into the source tree) until a directory holding both pyproject.toml and
src/mjlab_microduck/ is found. (Not policies-v1/: it is git-ignored, so a
fresh clone does not have it until the ONNX files are copied in.)

Stdlib only: imported both by the ROS 2 nodes (system Python 3.10) and by
sim_worker.py (the project venv, Python 3.12). The constants are Path objects;
wrap them in str() where an API wants a string (ROS parameters, launch args).
"""

import os
from pathlib import Path


def _is_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "mjlab_microduck").is_dir()


def repo_root() -> Path:
    env = os.environ.get("MICRODUCK_RL_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not _is_repo_root(root):
            raise RuntimeError(
                f"MICRODUCK_RL_ROOT={env!r} is not a microduck_rl checkout "
                "(expected pyproject.toml and src/mjlab_microduck/ in it)."
            )
        return root
    for parent in Path(__file__).resolve().parents:
        if _is_repo_root(parent):
            return parent
    raise RuntimeError(
        "Cannot find the microduck_rl checkout. Build with "
        "`colcon build --symlink-install`, or export MICRODUCK_RL_ROOT=/path/to/microduck_rl"
    )


ROOT = repo_root()
VENV_PYTHON = ROOT / ".venv" / "bin" / "python3"
SCRIPTS_DIR = ROOT / "scripts"
POLICIES_DIR = ROOT / "policies-v1"
SCENE_XML = ROOT / "src" / "mjlab_microduck" / "robot" / "microduck" / "escena_rayuela.xml"
SIM_WORKER_SCRIPT = ROOT / "ros2_microduck" / "src" / "rayuela" / "rayuela" / "sim_worker.py"
