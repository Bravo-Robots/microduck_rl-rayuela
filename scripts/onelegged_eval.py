"""Headless evaluation of a OneLeggedStand / OneLeggedHop checkpoint.

The acceptance test for the one-legged tasks: runs the real training env
(same DR, same terminations) with the checkpoint's policy, and reports per
spawn type how long episodes last, what ends them, and — when the swing foot
comes down — which way the duck was falling.

It is what showed that the first Stand run HAD learned to stand: the pushes
it trained with were knocking it over. Always compare against --push 0 before
concluding that a policy "cannot balance".

    uv run python scripts/onelegged_eval.py --checkpoint <run>/model_1499.pt
    uv run python scripts/onelegged_eval.py --checkpoint <...> --push 0.1
    uv run python scripts/onelegged_eval.py --checkpoint <...> --push 0 --no-com-dr

--iteration sets the curriculum step counter (defaults to the checkpoint's
number), so the DR and spawn mix match the end of training.
"""

import argparse
import math
import re
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab_microduck.tasks.microduck_onelegged_env_cfg import STANCE_BASE_ROLL

TASKS = {"stand": "Mjlab-OneLeggedStand-Flat-MicroDuck",
         "hop": "Mjlab-OneLeggedHop-Flat-MicroDuck"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", choices=TASKS, default="stand")
    ap.add_argument("--iteration", type=int, default=None,
                    help="curriculum step counter, in iterations")
    ap.add_argument("--push", type=float, default=None,
                    help="fixed push magnitude in m/s (0 disables pushes); "
                         "default: whatever the curriculum gives")
    ap.add_argument("--no-com-dr", action="store_true",
                    help="disable the CoM randomization")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    task = TASKS[args.task]
    iteration = args.iteration
    if iteration is None:
        m = re.search(r"model_(\d+)\.pt$", args.checkpoint)
        iteration = int(m.group(1)) if m else 0

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = args.num_envs
    if args.push is not None:
        cfg.curriculum.pop("push_magnitude", None)
        if args.push == 0:
            cfg.events.pop("push_robot", None)
        else:
            v = args.push
            cfg.events["push_robot"].params["velocity_range"] = {"x": (-v, v), "y": (-v, v)}
    if args.no_com_dr:
        for e in ("base_com", "randomize_com", "randomize_head_com"):
            cfg.events.pop(e, None)
        for c in ("com_range", "head_com_range"):
            cfg.curriculum.pop(c, None)

    agent = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.common_step_counter = iteration * 24
    env.reset()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
        wrapped, asdict(agent), device="cuda:0")
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    robot = env.scene["robot"]
    dt = env.step_dt

    def roll_pitch():
        q = robot.data.root_link_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
        return torch.rad2deg(roll), torch.rad2deg(pitch)

    n = args.num_envs
    roll, pitch = roll_pitch()
    one_leg = roll.abs() > 15          # spawn type of each env's current episode
    t = torch.zeros(n, device=roll.device)
    episodes = []
    names = env.termination_manager.active_terms
    obs = wrapped.get_observations()
    for _ in range(int(args.seconds / dt)):
        with torch.no_grad():
            obs, _, dones, _ = wrapped.step(policy(obs))
        t += 1
        causes = {k: env.termination_manager.get_term(k).clone() for k in names}
        for i in torch.nonzero(dones).flatten().tolist():
            episodes.append(dict(
                one_leg=bool(one_leg[i]), dur=float(t[i]) * dt,
                cause=next((k for k in names if bool(causes[k][i])), "?"),
                roll=float(roll[i]), pitch=float(pitch[i])))
        roll_now, pitch_now = roll_pitch()
        d = dones.bool()
        one_leg[d] = roll_now[d].abs() > 15
        t[d] = 0
        roll, pitch = roll_now, pitch_now

    nominal = math.degrees(STANCE_BASE_ROLL)
    push = "curriculum" if args.push is None else f"{args.push} m/s"
    print(f"\n{task}  iteration {iteration}  push {push}  "
          f"CoM DR {'off' if args.no_com_dr else 'on'}  "
          f"({len(episodes)} episodes, {args.seconds:.0f} s x {n} envs)")
    for label, sel in (("one-leg spawn", True), ("two-foot spawn", False)):
        eps = [e for e in episodes if e["one_leg"] == sel]
        if not eps:
            continue
        dur = sorted(e["dur"] for e in eps)
        counts = {}
        for e in eps:
            counts[e["cause"]] = counts.get(e["cause"], 0) + 1
        print(f"  {label:15} n={len(eps):4d}  median {dur[len(dur) // 2]:5.2f} s  "
              f"mean {sum(dur) / len(dur):5.2f} s   " + "  ".join(
                  f"{c}:{v / len(eps):.0%}"
                  for c, v in sorted(counts.items(), key=lambda x: -x[1])))
        down = [e for e in eps if e["cause"] == "swing_foot_down"]
        if down:
            inward = sum(e["roll"] < nominal - 5 for e in down) / len(down)
            print(f"  {'':15} swing foot down: mean roll {sum(e['roll'] for e in down) / len(down):+.1f}"
                  f" deg (stance {nominal:+.0f}), falling inward {inward:.0%}")


if __name__ == "__main__":
    main()
