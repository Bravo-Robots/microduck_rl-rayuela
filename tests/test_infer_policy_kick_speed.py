"""infer_policy.py feeds a speed-commanded kick (BallKickSpeed) its target
ball exit speed through the twist vx slot — and ONLY that kind of policy.
The fixed-speed kick policies were trained on an all-zero command, so the
opt-in must default off and leave them untouched."""

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


def _state(behavior_mode, commanded, kick_speed=1.25, current_policy="standing"):
    return SimpleNamespace(
        new_cmd_obs=True,
        behavior_mode=behavior_mode,
        kick_speed_commanded=commanded,
        kick_speed=kick_speed,
        current_policy=current_policy,
        vel_cmd=np.array([0.3, 0.0, 0.5], dtype=np.float32),
        is_sitstand=False,
        sit_mode=False,
        head_offset=np.zeros(4, dtype=np.float32),
        body_cmd=np.zeros(6, dtype=np.float32),
        command=None,
    )


def test_opt_in_defaults_off(PolicyInference):
    import inspect
    params = inspect.signature(PolicyInference.__init__).parameters
    assert params["kick_speed_commanded"].default is False


@pytest.mark.parametrize("kick", ["kick_left", "kick_right"])
def test_fixed_speed_kick_still_gets_all_zero_command(PolicyInference, kick):
    s = _state(kick, commanded=False)
    PolicyInference._update_command(s)
    assert np.all(s.command == 0.0)


@pytest.mark.parametrize("kick", ["kick_left", "kick_right"])
def test_speed_commanded_kick_gets_speed_in_vx_only(PolicyInference, kick):
    s = _state(kick, commanded=True, kick_speed=1.25)
    PolicyInference._update_command(s)
    assert s.command.shape == (13,)
    assert s.command[0] == pytest.approx(1.25)
    assert np.all(s.command[1:] == 0.0)


def test_non_kick_behavior_stays_zero_even_when_opted_in(PolicyInference):
    s = _state("roulade", commanded=True)
    PolicyInference._update_command(s)
    assert np.all(s.command == 0.0)


def test_walking_command_unaffected_by_opt_in(PolicyInference):
    s = _state(None, commanded=True, current_policy="walking")
    PolicyInference._update_command(s)
    assert np.allclose(s.command[:3], [0.3, 0.0, 0.5])


def test_trigger_behavior_records_requested_speed(PolicyInference):
    # Unknown behaviour -> returns early, but the requested speed is stored
    # first, so a later _update_command writes it.
    s = SimpleNamespace(kick_speed=1.0, behavior_sessions={})
    PolicyInference.trigger_behavior(s, "kick_right", kick_speed=0.6)
    assert s.kick_speed == pytest.approx(0.6)
