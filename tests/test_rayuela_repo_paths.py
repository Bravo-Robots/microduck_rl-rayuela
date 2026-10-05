"""rayuela finds the checkout wherever it was cloned (no /home/robot paths).

repo_paths.py is stdlib-only, so it is loaded straight from its file: these
tests need neither ROS 2 nor the rayuela package installed.
"""

import importlib.util
import os
import pathlib
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RAYUELA_PKG = REPO / "ros2_microduck" / "src" / "rayuela"
REPO_PATHS = RAYUELA_PKG / "rayuela" / "repo_paths.py"


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"repo_paths_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("MICRODUCK_RL_ROOT", raising=False)


def test_resolves_the_checkout_from_the_source_tree():
    rp = _load(REPO_PATHS)
    assert pathlib.Path(rp.repo_root()) == REPO
    assert pathlib.Path(rp.scene_xml()).is_file()
    assert (pathlib.Path(rp.scripts_dir()) / "infer_policy.py").is_file()
    assert pathlib.Path(rp.sim_worker_script()).is_file()


def test_symlink_install_resolves_back_to_the_checkout(tmp_path, monkeypatch):
    # colcon build --symlink-install: the imported file is a symlink outside src/.
    link = tmp_path / "build" / "rayuela" / "rayuela" / "repo_paths.py"
    link.parent.mkdir(parents=True)
    link.symlink_to(REPO_PATHS)
    monkeypatch.chdir(tmp_path)
    assert pathlib.Path(_load(link).repo_root()) == REPO


def test_copied_install_inside_the_checkout_walks_up_to_it(tmp_path, monkeypatch):
    # Plain colcon build: a COPY under ros2_microduck/install/... still sits
    # inside the checkout. Simulated with a throwaway checkout skeleton.
    fake_repo = tmp_path / "microduck_rl"
    (fake_repo / "src" / "mjlab_microduck").mkdir(parents=True)
    (fake_repo / "pyproject.toml").write_text("")
    copy = fake_repo / "ros2_microduck" / "install" / "rayuela" / "repo_paths.py"
    copy.parent.mkdir(parents=True)
    shutil.copy(REPO_PATHS, copy)
    monkeypatch.chdir(tmp_path)
    assert pathlib.Path(_load(copy).repo_root()) == fake_repo


def test_falls_back_to_the_current_directory(tmp_path, monkeypatch):
    copy = tmp_path / "elsewhere" / "repo_paths.py"
    copy.parent.mkdir()
    shutil.copy(REPO_PATHS, copy)
    rp = _load(copy)

    monkeypatch.chdir(REPO / "ros2_microduck")
    assert pathlib.Path(rp.repo_root()) == REPO

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="MICRODUCK_RL_ROOT"):
        rp.repo_root()


def test_env_var_overrides_and_is_validated(tmp_path, monkeypatch):
    copy = tmp_path / "elsewhere" / "repo_paths.py"
    copy.parent.mkdir()
    shutil.copy(REPO_PATHS, copy)
    rp = _load(copy)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("MICRODUCK_RL_ROOT", str(REPO))
    assert pathlib.Path(rp.repo_root()) == REPO
    assert pathlib.Path(rp.venv_python()) == REPO / ".venv" / "bin" / "python3"

    monkeypatch.setenv("MICRODUCK_RL_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not a microduck_rl checkout"):
        rp.repo_root()


def test_no_machine_specific_paths_left_in_the_rayuela_package():
    offenders = []
    for path in RAYUELA_PKG.rglob("*.py"):
        if path == REPO_PATHS:
            continue  # its docstring names the old path on purpose
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "/home/" in line:
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "hardcoded home paths:\n" + "\n".join(offenders)
