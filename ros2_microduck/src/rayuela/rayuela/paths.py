"""Where the microduck_rl checkout lives, found at runtime — no hardcoded paths.

Order: the MICRODUCK_RL_ROOT environment variable if set, else walk up from
this file (symlinks resolved, so `colcon build --symlink-install` points back
into the source tree) until a directory holding both pyproject.toml and
policies-v1/ is found.
"""

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("MICRODUCK_RL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "policies-v1").is_dir():
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