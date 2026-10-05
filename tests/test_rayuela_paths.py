"""rayuela finds the checkout wherever it was cloned (no /home/robot paths).

rayuela/paths.py is stdlib-only, so it is loaded straight from its file: these
tests need neither ROS 2 nor the rayuela package installed. It resolves the
root at import time, so each case loads a fresh copy.
"""

import importlib.util
import pathlib
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RAYUELA_PKG = REPO / "ros2_microduck" / "src" / "rayuela"
PATHS = RAYUELA_PKG / "rayuela" / "paths.py"


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"rayuela_paths_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("MICRODUCK_RL_ROOT", raising=False)


def test_resolves_the_checkout_from_the_source_tree():
    p = _load(PATHS)
    assert p.ROOT == REPO
    assert p.SCENE_XML.is_file()
    assert (p.SCRIPTS_DIR / "infer_policy.py").is_file()
    assert p.SIM_WORKER_SCRIPT.is_file()
    assert p.VENV_PYTHON == REPO / ".venv" / "bin" / "python3"


def test_does_not_need_the_git_ignored_policies_dir(tmp_path):
    # A fresh clone has no policies-v1/ until the ONNX files are copied in.
    fake_repo = tmp_path / "microduck_rl"
    (fake_repo / "src" / "mjlab_microduck").mkdir(parents=True)
    (fake_repo / "pyproject.toml").write_text("")
    copy = fake_repo / "ros2_microduck" / "src" / "rayuela" / "rayuela" / "paths.py"
    copy.parent.mkdir(parents=True)
    shutil.copy(PATHS, copy)
    assert _load(copy).ROOT == fake_repo


def test_symlink_install_resolves_back_to_the_checkout(tmp_path):
    # colcon build --symlink-install: the imported file is a symlink outside src/.
    link = tmp_path / "build" / "rayuela" / "rayuela" / "paths.py"
    link.parent.mkdir(parents=True)
    link.symlink_to(PATHS)
    assert _load(link).ROOT == REPO


def test_copied_install_inside_the_checkout_walks_up_to_it(tmp_path):
    # Plain colcon build: a COPY under ros2_microduck/install/... still sits
    # inside the checkout.
    fake_repo = tmp_path / "microduck_rl"
    (fake_repo / "src" / "mjlab_microduck").mkdir(parents=True)
    (fake_repo / "pyproject.toml").write_text("")
    copy = fake_repo / "ros2_microduck" / "install" / "rayuela" / "paths.py"
    copy.parent.mkdir(parents=True)
    shutil.copy(PATHS, copy)
    assert _load(copy).ROOT == fake_repo


def test_env_var_overrides_and_is_validated(tmp_path, monkeypatch):
    copy = tmp_path / "elsewhere" / "paths.py"
    copy.parent.mkdir()
    shutil.copy(PATHS, copy)

    with pytest.raises(RuntimeError, match="MICRODUCK_RL_ROOT"):
        _load(copy)

    monkeypatch.setenv("MICRODUCK_RL_ROOT", str(REPO))
    assert _load(copy).ROOT == REPO

    monkeypatch.setenv("MICRODUCK_RL_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="not a microduck_rl checkout"):
        _load(copy)


def test_no_machine_specific_paths_left_in_the_rayuela_package():
    offenders = []
    for path in RAYUELA_PKG.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "/home/" in line:
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "hardcoded home paths:\n" + "\n".join(offenders)
