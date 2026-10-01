"""OneLeggedStand / OneLeggedHop: balance on one foot, then hop on it."""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_onelegged_env_cfg import (
    EXPERIMENT_NAME,
    HEAD_FREE_WEIGHT,
    HOP_FLIGHT_MAX_S,
    HOP_FLIGHT_MIN_S,
    HOP_FLIGHT_STAGES,
    HOP_PUSH_STAGES,
    HOP_SPEED_STAGES,
    PUSH_STAGES,
    STANCE_BASE_ROLL,
    STANCE_POSE_HOP_STAGES,
    STANCE_RIGHT,
    STANCE_SPAWN_STAGES,
    STAND_PHASE_ITERS,
    SWING_GRACE_S,
    MicroduckOneLeggedHopRlCfg,
    MicroduckOneLeggedStandRlCfg,
    make_microduck_onelegged_env_cfg,
    mirror_stance,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


@pytest.fixture(scope="module")
def stand():
    return make_microduck_onelegged_env_cfg()


@pytest.fixture(scope="module")
def hop():
    return make_microduck_onelegged_env_cfg(hop=True)


@pytest.fixture(scope="module")
def walk():
    return make_microduck_velocity_env_cfg()


def _sensors(support_air, swing_found, t_s=5.0):
    """Fake contact sensors for the two feet, plus an episode clock."""
    def sensor(air, found):
        n = len(air)
        return SimpleNamespace(data=SimpleNamespace(
            current_air_time=torch.tensor(air).reshape(n, 1),
            found=torch.tensor(found).reshape(n, 1).float(),
        ))
    n = len(support_air)
    return SimpleNamespace(
        num_envs=n, device="cpu", step_dt=0.02,
        episode_length_buf=torch.full((n,), int(t_s / 0.02)),
        scene=SimpleNamespace(sensors={
            "support_contact": sensor(support_air, [0.0] * n),
            "swing_contact": sensor([0.0] * n, swing_found),
        }),
    )


# ── the measured stance ─────────────────────────────────────────────────────

def test_stance_respects_the_joint_limits():
    """The first stance used 28 deg of hip roll against a +/-22 deg limit."""
    assert abs(math.degrees(STANCE_RIGHT[1])) <= 22.0 + 1e-6
    assert abs(math.degrees(STANCE_RIGHT[10])) <= 22.0 + 1e-6
    assert abs(math.degrees(STANCE_RIGHT[8])) <= 25.0 + 1e-6      # head_roll


def test_stance_turns_the_head():
    """The head is what widens the CoM margin from 1.2 to 5.8 mm."""
    assert abs(math.degrees(STANCE_RIGHT[7])) >= 45.0               # head_yaw


def test_left_stance_is_the_mirror_of_the_right():
    left = mirror_stance(STANCE_RIGHT)
    assert mirror_stance(left) == pytest.approx(STANCE_RIGHT)
    # The tucked leg swaps sides: the right leg folds in the left stance.
    assert left[11] == pytest.approx(-STANCE_RIGHT[2])
    assert left[1] == pytest.approx(-STANCE_RIGHT[10])


@pytest.mark.parametrize("foot, other, sign", [("right", "left", 1), ("left", "right", -1)])
def test_sensors_and_spawn_follow_the_support_foot(foot, other, sign):
    cfg = make_microduck_onelegged_env_cfg(support_foot=foot)
    by_name = {s.name: s for s in cfg.scene.sensors}
    assert foot in by_name["support_contact"].primary.pattern
    assert other in by_name["swing_contact"].primary.pattern
    assert by_name["support_contact"].track_air_time
    spawn = cfg.events["reset_one_leg_stance"].params
    assert spawn["base_roll"] == pytest.approx(sign * STANCE_BASE_ROLL)


# ── spawn ───────────────────────────────────────────────────────────────────

def test_spawn_runs_after_the_default_resets(stand):
    """It overwrites what reset_base / reset_robot_joints wrote (dict order)."""
    names = list(stand.events)
    i = names.index("reset_one_leg_stance")
    assert i > names.index("reset_base") and i > names.index("reset_robot_joints")
    assert stand.events["reset_one_leg_stance"].mode == "reset"


def test_two_foot_starts_grow_once_balance_exists():
    """Deployment always starts on two feet, so the lift must be trained."""
    probs = [s["params"]["prob"] for s in STANCE_SPAWN_STAGES]
    assert probs == sorted(probs, reverse=True)
    assert 0.0 < probs[-1] < 1.0


# ── rewards: what was wrong the first time ──────────────────────────────────

def test_the_hip_roll_pinning_pose_term_is_gone(stand):
    """std 0.05 on hip_roll made the whole one-leg region a flat zero."""
    assert stand.rewards["pose"].weight == 0.0
    target = stand.rewards["stance_pose"].params["target_overrides"]
    assert target[10] == pytest.approx(STANCE_RIGHT[10])
    assert stand.rewards["stance_pose"].params["std"] >= 0.4


def test_upright_does_not_crush_the_required_lean(stand):
    """27 deg of roll paid ~0.01 under the stock std; it must stay visible."""
    std = stand.rewards["upright"].params["std"]
    assert math.exp(-(STANCE_BASE_ROLL / std) ** 2) > 0.3


def test_standing_still_no_longer_dominates(stand, walk):
    """Velocity tracking at 2.0 + 2.0 paid half the stack for not moving."""
    for term in ("track_linear_velocity", "track_angular_velocity"):
        assert stand.rewards[term].weight < walk.rewards[term].weight


def test_the_head_is_released_and_stays_released(stand, walk):
    for term in ("head_pose_tracking", "head_pose_bias"):
        assert abs(stand.rewards[term].weight) <= HEAD_FREE_WEIGHT
        assert abs(stand.rewards[term].weight) <= abs(walk.rewards[term].weight)
    # The recipe's curriculum ramps head_pose_bias to 3.0 mid-run, which would
    # re-pin the head exactly when balancing starts to use it.
    assert "head_pose_bias_weight" in walk.curriculum
    assert "head_pose_bias_weight" not in stand.curriculum


def test_hip_roll_is_allowed_to_rest_on_its_limit(stand):
    pattern = stand.rewards["dof_pos_limits"].params["asset_cfg"].joint_names[0]
    import re
    assert not re.match(pattern, "right_hip_roll")
    assert re.match(pattern, "right_knee")


def test_two_footed_gait_terms_are_off(stand):
    for term in ("air_time", "foot_slip", "foot_clearance", "foot_swing_height"):
        if term in stand.rewards:
            assert stand.rewards[term].weight == 0.0


def test_motion_blockers_stay_low(stand, walk):
    for term in ("body_ang_vel", "angular_momentum"):
        assert stand.rewards[term].weight < 0
        assert abs(stand.rewards[term].weight) < abs(walk.rewards[term].weight)


# ── swing foot: termination with a grace window, and a dense signal ─────────

def test_swing_touchdown_terminates_after_the_grace_only():
    down = _sensors([0.0], [1.0], t_s=SWING_GRACE_S / 2)
    assert not mdp.swing_foot_touchdown(down, "swing_contact", grace_s=SWING_GRACE_S).any()
    down = _sensors([0.0], [1.0], t_s=SWING_GRACE_S * 2)
    assert mdp.swing_foot_touchdown(down, "swing_contact", grace_s=SWING_GRACE_S).all()


def test_swing_foot_has_both_a_termination_and_a_dense_cost(stand):
    term = stand.terminations["swing_foot_down"]
    assert term.func is mdp.swing_foot_touchdown and not term.time_out
    assert stand.rewards["swing_foot_lifted"].weight < 0


# ── hop phase ───────────────────────────────────────────────────────────────

def test_stand_task_has_no_hop_terms(stand):
    assert "hop_flight" not in stand.rewards
    assert "hop_speed" not in stand.curriculum


def test_flight_starts_exactly_where_the_stand_phase_ends():
    """So a hop run warm-started from a stand checkpoint lines up."""
    first_on = next(s["step"] for s in HOP_FLIGHT_STAGES if s["weight"] > 0)
    assert first_on == STAND_PHASE_ITERS * 24
    assert HOP_FLIGHT_STAGES[0]["weight"] == 0.0


def test_the_support_leg_is_released_to_jump():
    weights = [s["weight"] for s in STANCE_POSE_HOP_STAGES]
    assert weights == sorted(weights, reverse=True) and weights[-1] > 0


def test_speed_opens_only_after_hopping_in_place():
    first_open = HOP_SPEED_STAGES[1]["step"]
    flight_full = HOP_FLIGHT_STAGES[-1]["step"]
    assert first_open >= flight_full
    assert HOP_SPEED_STAGES[0]["lin_vel_range"] <= 0.05


def test_flight_pays_only_inside_the_band_and_with_the_swing_foot_up():
    env = _sensors([0.0, HOP_FLIGHT_MIN_S / 2, 0.15, HOP_FLIGHT_MAX_S * 2, 0.15],
                   [0.0, 0.0, 0.0, 0.0, 1.0])
    out = mdp.hop_flight_reward(env, "support_contact", "swing_contact",
                                HOP_FLIGHT_MIN_S, HOP_FLIGHT_MAX_S)
    assert list(out) == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_missing_sensors_are_survivable():
    env = SimpleNamespace(num_envs=3, device="cpu", scene=SimpleNamespace(sensors={}))
    assert torch.equal(mdp.hop_flight_reward(env, "a", "b"), torch.zeros(3))
    assert not mdp.swing_foot_touchdown(env, "a").any()


# ── pushes: sized for one leg, not for walking ──────────────────────────────

def _mag(stage):
    return max(abs(v) for rng in stage["velocity_range"].values() for v in rng)


def test_stand_pushes_stay_in_the_recoverable_range():
    """Measured on the first Stand run: 49% of episodes survive 0.10 m/s, 15%
    survive 0.15, 0% survive the 0.30 of the walking recipe."""
    mags = [_mag(s) for s in PUSH_STAGES]
    assert mags == sorted(mags)
    assert mags[0] <= 0.05 and mags[-1] <= 0.10


@pytest.mark.parametrize("hop_task", [False, True])
def test_the_walking_push_is_replaced(hop_task):
    cfg = make_microduck_onelegged_env_cfg(hop=hop_task)
    assert cfg.events["push_robot"].params["velocity_range"] == PUSH_STAGES[0]["velocity_range"]
    term = cfg.curriculum["push_magnitude"]
    assert term.func is mdp.push_curriculum
    assert term.params["push_stages"] == (HOP_PUSH_STAGES if hop_task else PUSH_STAGES)


def test_larger_pushes_only_once_hopping_exists():
    extra = HOP_PUSH_STAGES[len(PUSH_STAGES):]
    assert extra and all(_mag(s) > _mag(PUSH_STAGES[-1]) for s in extra)
    assert all(s["step"] >= HOP_FLIGHT_STAGES[-1]["step"] for s in extra)


def test_the_stand_trains_through_every_push_stage_before_the_hop_starts():
    """A hop warm-started from a stand checkpoint resumes at that checkpoint's
    iteration; starting the hop before the stand has seen its last push stage
    would skip it."""
    assert STAND_PHASE_ITERS * 24 > PUSH_STAGES[-1]["step"]


# ── runners ─────────────────────────────────────────────────────────────────

def test_both_tasks_share_an_experiment_so_hop_can_load_a_stand_checkpoint():
    assert MicroduckOneLeggedStandRlCfg.experiment_name == EXPERIMENT_NAME
    assert MicroduckOneLeggedHopRlCfg.experiment_name == EXPERIMENT_NAME
    assert MicroduckOneLeggedStandRlCfg.run_name != MicroduckOneLeggedHopRlCfg.run_name
