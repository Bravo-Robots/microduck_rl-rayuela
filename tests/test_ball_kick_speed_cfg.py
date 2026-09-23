"""BallKickSpeed: commanded kick strength (target ball exit speed in twist vx)."""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_ball_kick_speed_env_cfg import (
    KICK_AIM_STAGES,
    KICK_GAUSSIAN_STAGES,
    KICK_GAUSSIAN_STD,
    KICK_LINEAR_STAGES,
    KICK_SPEED_RANGE,
    KICK_SPEED_RANGE_STAGES,
    MicroduckBallKickSpeedLeftRlCfg,
    MicroduckBallKickSpeedRightRlCfg,
    make_microduck_ball_kick_speed_env_cfg,
)


def _fake_env(ball_vel_xy, target_speed, kick_dir=(1.0, 0.0)):
    """Just enough of an env for the ball-speed rewards."""
    vel = torch.tensor(ball_vel_xy, dtype=torch.float32)
    n = vel.shape[0]
    vel3 = torch.cat([vel, torch.zeros(n, 1)], dim=1)
    cmd = torch.zeros(n, 3)
    cmd[:, 0] = torch.tensor(target_speed, dtype=torch.float32)
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        scene={"ball": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=vel3))},
        command_manager=SimpleNamespace(get_command=lambda name: cmd),
    )
    env._ball_kick_dir_w = torch.tensor([kick_dir] * n, dtype=torch.float32)
    return env


# ── cfg ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("foot, sign", [("right", -1.0), ("left", 1.0)])
def test_cfg_builds_per_foot_with_speed_command(foot, sign):
    cfg = make_microduck_ball_kick_speed_env_cfg(kick_foot=foot)
    twist = cfg.commands["twist"]
    assert isinstance(twist, mdp.KickSpeedCommandCfg)
    assert tuple(twist.speed_range) == KICK_SPEED_RANGE_STAGES[0]["range"]
    # Drawn once per episode, never mid-episode (episodes are 5 s).
    assert min(twist.resampling_time_range) > 100.0
    # Ball spawns at the kicking foot's side.
    assert cfg.events["reset_ball"].params["offset"][1] * sign > 0


def test_speed_range_is_positive_and_spans_the_board():
    lo, hi = KICK_SPEED_RANGE
    # Measured in escena_rayuela.xml: 0.3 m/s -> casilla 1, ~1.75 m/s -> cielo,
    # and headroom above that for a longer board.
    assert 0.0 < lo <= 0.3 and hi >= 2.5


def test_kick_rewards_are_command_relative_with_correct_signs():
    cfg = make_microduck_ball_kick_speed_env_cfg(kick_foot="right")
    fwd = cfg.rewards["ball_forward_velocity"]
    over = cfg.rewards["ball_speed_overshoot"]
    assert fwd.func is mdp.ball_forward_velocity_to_command
    assert over.func is mdp.ball_speed_overshoot_to_command
    # Positive reward on a [0,1] term; the overshoot COST (>= 0) needs a
    # negative weight — a positive one would pay the policy to overshoot.
    assert fwd.weight > 0 and over.weight < 0
    assert fwd.params["command_name"] == over.params["command_name"] == "twist"


def test_runner_cfgs_have_distinct_experiments():
    assert MicroduckBallKickSpeedRightRlCfg.experiment_name == "ball_kick_speed_right"
    assert MicroduckBallKickSpeedLeftRlCfg.experiment_name == "ball_kick_speed_left"


# ── rewards ──────────────────────────────────────────────────────────────────

def test_forward_reward_is_one_exactly_at_target_for_any_target():
    targets = [0.3, 0.75, 1.0, 1.8]
    env = _fake_env([[t, 0.0] for t in targets], targets)
    r = mdp.ball_forward_velocity_to_command(env)
    assert torch.allclose(r, torch.ones(len(targets)))


def test_forward_reward_is_linear_below_target_and_capped_above():
    env = _fake_env([[0.5, 0.0], [2.0, 0.0]], [1.0, 1.0])
    r = mdp.ball_forward_velocity_to_command(env)
    assert torch.allclose(r, torch.tensor([0.5, 1.0]))


def test_forward_reward_ignores_still_backward_and_sideways_balls():
    env = _fake_env([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.5]], [1.0, 1.0, 1.0])
    r = mdp.ball_forward_velocity_to_command(env)
    assert torch.allclose(r, torch.zeros(3))


def test_forward_reward_follows_the_kick_direction():
    # Robot faced +y at reset: a ball rolling along +y is "forward".
    env = _fake_env([[0.0, 0.6]], [0.6], kick_dir=(0.0, 1.0))
    assert torch.allclose(mdp.ball_forward_velocity_to_command(env), torch.ones(1))


def test_overshoot_is_zero_up_to_target_and_normalised_above():
    env = _fake_env([[0.3, 0.0], [0.45, 0.0], [1.5, 0.0]], [0.3, 0.3, 1.0])
    o = mdp.ball_speed_overshoot_to_command(env)
    # 0 at target; 0.15 over a 0.3 target = 0.5; 0.5 over a 1.0 target = 0.5.
    assert torch.allclose(o, torch.tensor([0.0, 0.5, 0.5]), atol=1e-6)


def test_matches_fixed_target_kick_at_one_meter_per_second():
    # At 1.0 m/s the normalised terms equal the original raw ones, so the
    # weights 12 / -4 reproduce the config that trained the working policy.
    speeds = [[0.4, 0.0], [1.0, 0.0], [1.7, 0.0]]
    env = _fake_env(speeds, [1.0, 1.0, 1.0])
    assert torch.allclose(
        mdp.ball_forward_velocity_to_command(env),
        mdp.ball_forward_velocity(env, max_speed=1.0),
    )
    assert torch.allclose(
        mdp.ball_speed_overshoot_to_command(env),
        mdp.ball_speed_overshoot_penalty(env, target_speed=1.0),
    )


# ── command sampling ─────────────────────────────────────────────────────────

def test_command_samples_inside_range_per_env_and_zeroes_other_slots():
    torch.manual_seed(0)
    n = 4096
    fake = SimpleNamespace(
        vel_command_b=torch.full((n, 3), 9.0),
        device="cpu",
        _speed_lo=KICK_SPEED_RANGE[0],
        _speed_hi=KICK_SPEED_RANGE[1],
        _log_uniform=True,
    )
    mdp.KickSpeedCommand._resample_command(fake, torch.arange(n))
    speed = fake.vel_command_b[:, 0]
    assert speed.min() >= KICK_SPEED_RANGE[0] and speed.max() <= KICK_SPEED_RANGE[1]
    assert speed.std() > 0.3  # genuinely spread across the range, per env
    assert torch.all(fake.vel_command_b[:, 1:] == 0.0)


def test_command_resample_only_touches_given_envs():
    fake = SimpleNamespace(
        vel_command_b=torch.full((4, 3), 7.0),
        device="cpu",
        _speed_lo=0.3,
        _speed_hi=1.8,
        _log_uniform=True,
    )
    mdp.KickSpeedCommand._resample_command(fake, torch.tensor([1, 3]))
    assert torch.all(fake.vel_command_b[[0, 2]] == 7.0)


# ── aiming, both feet ────────────────────────────────────────────────────────

@pytest.mark.parametrize("foot", ["left", "right"])
def test_both_feet_get_the_aim_term_and_its_ramp(foot):
    cfg = make_microduck_ball_kick_speed_env_cfg(kick_foot=foot)
    assert cfg.rewards["ball_aim"].func is mdp.ball_kick_aim_error
    assert "ball_aim_weight" in cfg.curriculum
    # Live from step 0 it would tax the kick before the kick exists.
    assert cfg.rewards["ball_aim"].weight == 0.0
    steps = [st["step"] for st in KICK_AIM_STAGES]
    weights = [st["weight"] for st in KICK_AIM_STAGES]
    assert steps == sorted(steps) and steps[0] == 0
    assert weights[-1] < 0 and weights == sorted(weights, reverse=True)


def test_aim_cost_is_zero_for_a_straight_kick():
    env = _fake_env([[1.0, 0.0], [0.4, 0.0]], [1.0, 0.4])
    assert torch.allclose(mdp.ball_kick_aim_error(env), torch.zeros(2))


def test_aim_cost_is_the_tangent_of_the_angle_at_any_speed():
    """The point of the ratio: 20 deg costs the same however hard the kick."""
    ang = math.radians(20)
    env = _fake_env(
        [[v * math.cos(ang), v * math.sin(ang)] for v in (0.3, 1.0, 2.5)],
        [0.3, 1.0, 2.5],
    )
    out = mdp.ball_kick_aim_error(env)
    assert torch.allclose(out, torch.full((3,), math.tan(ang)), atol=1e-5)


def test_aim_cost_is_symmetric_and_follows_the_kick_direction():
    env = _fake_env([[1.0, 0.5], [1.0, -0.5]], [1.0, 1.0])
    assert torch.allclose(mdp.ball_kick_aim_error(env), torch.full((2,), 0.5), atol=1e-6)
    # Robot faced +y: "sideways" is now the x axis.
    turned = _fake_env([[0.5, 1.0]], [1.0], kick_dir=(0.0, 1.0))
    assert torch.allclose(mdp.ball_kick_aim_error(turned), torch.tensor([0.5]), atol=1e-6)


def test_aim_cost_is_capped_and_muted_on_a_still_ball():
    # Sideways/backward kicks would send the tangent to infinity; a still ball
    # has no meaningful direction at all.
    env = _fake_env([[0.0, 1.0], [-1.0, 0.3], [0.0, 0.0]], [1.0, 1.0, 1.0])
    out = mdp.ball_kick_aim_error(env)
    assert out[0] == out[1] == 1.0  # max_penalty
    assert out[2] == 0.0


def test_a_20deg_kick_costs_more_than_it_loses_in_projection():
    """The whole point: the forward term alone barely notices a 20deg drift."""
    v, ang = 1.0, math.radians(20)
    env = _fake_env([[v * math.cos(ang), v * math.sin(ang)]], [1.0])
    fwd_loss = 1.0 - float(mdp.ball_forward_velocity_to_command(env)[0])
    aim_cost = float(mdp.ball_kick_aim_error(env)[0])
    assert fwd_loss < 0.07               # cos(20deg) -> only ~6% lost
    assert aim_cost > 5 * fwd_loss       # the new term is what actually bites


# ── speed accuracy: the Gaussian takes over from the linear plateau ──────────

def test_gaussian_peaks_only_at_the_command():
    env = _fake_env([[1.0, 0.0], [1.5, 0.0], [0.5, 0.0]], [1.0, 1.0, 1.0])
    g = mdp.ball_speed_gaussian_to_command(env, std=KICK_GAUSSIAN_STD)
    assert abs(float(g[0]) - 1.0) < 1e-6
    # Overshooting by 50% is punished as hard as undershooting by 50% — which
    # the linear term does NOT do: it pays full for both 1.0 and 1.5.
    assert abs(float(g[1]) - float(g[2])) < 1e-6
    assert float(g[1]) < 0.2
    lin = mdp.ball_forward_velocity_to_command(env)
    assert float(lin[0]) == float(lin[1]) == 1.0


def test_gaussian_demands_the_same_RELATIVE_precision_at_every_command():
    # 20% too fast, at three very different commanded strengths.
    targets = [0.3, 1.0, 2.5]
    env = _fake_env([[t * 1.2, 0.0] for t in targets], targets)
    g = mdp.ball_speed_gaussian_to_command(env, std=KICK_GAUSSIAN_STD)
    assert torch.allclose(g, g[0].expand(3), atol=1e-6)


def test_gaussian_is_flat_on_a_still_ball_so_it_cannot_bootstrap():
    """Why the linear term stays in the stack instead of being deleted."""
    env = _fake_env([[0.0, 0.0]], [1.0])
    assert float(mdp.ball_speed_gaussian_to_command(env, std=KICK_GAUSSIAN_STD)) < 0.01


def test_linear_hands_over_to_the_gaussian_without_a_gap():
    lin = {st["step"]: st["weight"] for st in KICK_LINEAR_STAGES}
    gau = {st["step"]: st["weight"] for st in KICK_GAUSSIAN_STAGES}
    assert lin.keys() == gau.keys()
    assert lin[0] > 0 and gau[0] == 0.0        # discovery first
    # The speed payoff never collapses mid-handover, and the linear term keeps
    # a residual so "kick at all" always pays.
    for step in sorted(lin):
        assert lin[step] + gau[step] >= 10.0
    assert min(lin.values()) > 0
    assert gau[max(gau)] > lin[max(lin)]       # accuracy ends up dominant


# ── range curriculum ─────────────────────────────────────────────────────────

def test_range_opens_from_what_the_policy_can_already_do_to_the_full_range():
    los = [st["range"][0] for st in KICK_SPEED_RANGE_STAGES]
    his = [st["range"][1] for st in KICK_SPEED_RANGE_STAGES]
    assert los == sorted(los, reverse=True) and his == sorted(his)
    assert KICK_SPEED_RANGE_STAGES[-1]["range"] == KICK_SPEED_RANGE
    # Stage 0 brackets the ~0.5-0.7 m/s the previous policy actually produced.
    assert los[0] <= 0.55 <= his[0]


@pytest.mark.parametrize("foot", ["left", "right"])
def test_cfg_wires_the_range_curriculum(foot):
    cfg = make_microduck_ball_kick_speed_env_cfg(kick_foot=foot)
    term = cfg.curriculum["kick_speed_range"]
    assert term.func is mdp.kick_speed_range_curriculum
    assert term.params["range_stages"] == KICK_SPEED_RANGE_STAGES


def test_range_curriculum_sets_the_live_command_not_the_cfg():
    applied = {}
    term = SimpleNamespace(set_speed_range=lambda lo, hi: applied.update(r=(lo, hi)))
    env = SimpleNamespace(
        common_step_counter=650 * 24,
        command_manager=SimpleNamespace(get_term=lambda name: term),
    )
    mdp.kick_speed_range_curriculum(env, None, "twist", KICK_SPEED_RANGE_STAGES)
    assert applied["r"] == KICK_SPEED_RANGE_STAGES[2]["range"]


def test_sampling_is_log_uniform_so_every_octave_gets_data():
    torch.manual_seed(0)
    n = 20000
    fake = SimpleNamespace(
        vel_command_b=torch.zeros(n, 3), device="cpu",
        _speed_lo=KICK_SPEED_RANGE[0], _speed_hi=KICK_SPEED_RANGE[1],
        _log_uniform=True,
    )
    mdp.KickSpeedCommand._resample_command(fake, torch.arange(n))
    speed = fake.vel_command_b[:, 0]
    assert speed.min() >= KICK_SPEED_RANGE[0] - 1e-6
    assert speed.max() <= KICK_SPEED_RANGE[1] + 1e-6
    # Uniform sampling of 0.25-2.6 would put only ~11% below 0.5; log-uniform
    # gives the soft end — the hard, useful half — about a third of the data.
    soft = float((speed < 0.5).float().mean())
    assert 0.25 < soft < 0.40
    assert torch.all(fake.vel_command_b[:, 1:] == 0.0)
