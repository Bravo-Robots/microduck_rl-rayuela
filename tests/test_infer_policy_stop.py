"""infer_policy.py: the X key aborts whatever is running and drops into the
standing policy. Measured in sim: from walking, sitting, mid-kick and
upside-down mid-roulade, alpha_standing ends standing (z ~116 mm) every time."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def PolicyInference():
    spec = importlib.util.spec_from_file_location(
        "infer_policy", REPO / "scripts" / "infer_policy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PolicyInference


STANDING, WALKING, OTHER = object(), object(), object()


def _busy(**overrides):
    """A duck in the middle of something, with every mode switched on."""
    state = dict(
        standing_session=STANDING, walking_session=WALKING, ort_session=OTHER,
        current_policy="roulade", behavior_mode="roulade", behavior_time_left=1.2,
        ground_pick_mode=True, ground_pick_phase=0.4,
        sit_mode=True, _sit_fall_s=0.3, slope_mode=True,
        vel_cmd=np.array([0.3, 0.1, 0.5], dtype=np.float32),
        head_offset=np.full(4, 0.2, dtype=np.float32),
        body_cmd=np.full(6, 0.01, dtype=np.float32),
        updated=0,
    )
    state.update(overrides)
    ns = SimpleNamespace(**state)
    ns._update_command = lambda: setattr(ns, "updated", ns.updated + 1)
    return ns


def test_stop_ends_every_mode_and_hands_over_to_standing(PolicyInference):
    duck = _busy()
    PolicyInference.stop_all(duck)
    assert duck.behavior_mode is None and duck.behavior_time_left == 0.0
    assert not duck.ground_pick_mode and duck.ground_pick_phase == 0.0
    assert not duck.sit_mode and duck._sit_fall_s == 0.0
    assert not duck.slope_mode
    assert duck.current_policy == "standing"
    assert duck.ort_session is STANDING


def test_stop_zeroes_every_command(PolicyInference):
    duck = _busy()
    PolicyInference.stop_all(duck)
    assert not duck.vel_cmd.any()
    assert not duck.head_offset.any()
    assert not duck.body_cmd.any()
    # The observation's command block has to be rebuilt, or the standing
    # policy would still see the aborted action's command.
    assert duck.updated == 1


def test_stop_without_a_standing_policy_changes_nothing(PolicyInference):
    duck = _busy(standing_session=None)
    PolicyInference.stop_all(duck)
    assert duck.behavior_mode == "roulade" and duck.ort_session is OTHER
    assert duck.updated == 0


def test_x_is_bound_and_documented():
    src = (REPO / "scripts" / "infer_policy.py").read_text()
    assert 'elif key == "x":' in src and "policy.stop_all()" in src
    assert "X:" in src and "STOP" in src
