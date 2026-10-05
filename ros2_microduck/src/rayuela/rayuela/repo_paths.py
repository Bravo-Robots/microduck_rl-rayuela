"""Where the microduck_rl checkout lives, for every rayuela script.

The rayuela files used to hardcode ``/home/robot/microduck_rl``, so the game
only ran on the machine it was written on. Everything that needs a file from
the checkout (the MuJoCo scene, the ONNX policies, ``scripts/infer_policy.py``,
the project venv) now asks this module instead.

Stdlib only on purpose: imported both by the ROS 2 nodes (system Python 3.10)
and by sim_worker.py (the project venv, Python 3.12).

Resolution order for the checkout root:

1. ``$MICRODUCK_RL_ROOT``, when set (for a workspace built outside the repo).
2. Walking up from this file's real path. This covers running from ``src/``,
   ``colcon build --symlink-install`` (symlinks resolve back to ``src/``) and a
   plain ``colcon build`` inside ``ros2_microduck/`` (``install/`` still sits
   inside the checkout).
3. Walking up from the current directory.
"""

import os

ROOT_ENV_VAR = "MICRODUCK_RL_ROOT"

_SCENE_RELPATH = os.path.join(
    "src", "mjlab_microduck", "robot", "microduck", "escena_rayuela.xml"
)
_SIM_WORKER_RELPATH = os.path.join(
    "ros2_microduck", "src", "rayuela", "rayuela", "sim_worker.py"
)


def _is_repo_root(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "pyproject.toml")) and os.path.isdir(
        os.path.join(path, "src", "mjlab_microduck")
    )


def _search_up(start: str) -> str | None:
    path = os.path.abspath(start)
    while True:
        if _is_repo_root(path):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def repo_root() -> str:
    """Absolute path of the microduck_rl checkout (see module docstring)."""
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        if not _is_repo_root(env):
            raise FileNotFoundError(
                f"${ROOT_ENV_VAR}={env!r} is not a microduck_rl checkout "
                "(expected pyproject.toml and src/mjlab_microduck/ in it)."
            )
        return os.path.abspath(env)

    for start in (os.path.dirname(os.path.realpath(__file__)), os.getcwd()):
        found = _search_up(start)
        if found is not None:
            return found

    raise FileNotFoundError(
        "Could not find the microduck_rl checkout from "
        f"{os.path.realpath(__file__)!r} or the current directory. "
        f"Build ros2_microduck/ inside the repo, or export {ROOT_ENV_VAR}=/path/to/microduck_rl."
    )


def scripts_dir() -> str:
    """``scripts/`` (home of infer_policy.py's PolicyInference)."""
    return os.path.join(repo_root(), "scripts")


def scene_xml() -> str:
    """The rayuela MuJoCo scene: board, ball, camera and duck."""
    return os.path.join(repo_root(), _SCENE_RELPATH)


def policies_dir() -> str:
    """Where the exported ONNX policies are looked up (git-ignored)."""
    return os.path.join(repo_root(), "policies-v1")


def venv_python() -> str:
    """The project venv's interpreter (created by ``uv sync``)."""
    return os.path.join(repo_root(), ".venv", "bin", "python3")


def sim_worker_script() -> str:
    """sim_worker.py in the source tree, run by the venv interpreter."""
    return os.path.join(repo_root(), _SIM_WORKER_RELPATH)
