"""Sweep commanded kick speed -> what the ball actually does, policy in the loop.

Acceptance test for a BallKickSpeed policy, and the source of the
speed->distance tables the rayuela deployment uses (board_geometry).

Runs escena_rayuela.xml headless (BAM actuators, 50 Hz, exactly as
infer_policy), settles the duck at HOME, triggers one kick per commanded speed
and waits until the ball stops (|v| < 1 cm/s for 0.5 s).

Columns: exit = the ball's peak speed. ang_out = its direction at that moment
relative to the kick direction; ang_rest = the direction to where it stopped
(the two agreeing means the roll is straight and the error is all in the
kick). tilt@2s / t_fall / tilt_end = whether the duck survived its own kick.

A kick is good when |ang_out| < 5 deg across the range and exit is within
~15% of the command.

    uv run python scripts/kick_sweep.py --foot right
    uv run python scripts/kick_sweep.py --foot left --mirror --speeds 0.3 1.2
"""
import argparse, math, os, sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_policy as ip

SCENE = "src/mjlab_microduck/robot/microduck/escena_rayuela.xml"
WALKING = "policies-v1/alpha_walking.onnx"
STANDING = "policies-v1/alpha_standing.onnx"
KICK_LEFT = os.environ.get("KICK_LEFT_ONNX",
                          "policies-v1/ball_kick_speed_left.onnx")
KICK_RIGHT = "policies-v1/ball_kick_speed_right.onnx"


def run_one(foot, speed, settle_s=2.0, max_s=25.0, kick_duration=2.0,
            verbose=False, mirror=False):
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
    model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(SCENE, bam_model, 0.005, 0.1, ip.BAM_VIN_MIN)
    policy = ip.PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path=WALKING,
        standing_onnx_path=STANDING,
        use_projected_gravity=True, new_cmd_obs=True,
        kick_left_onnx_path=KICK_LEFT if os.path.exists(KICK_LEFT) else None,
        kick_right_onnx_path=KICK_RIGHT if os.path.exists(KICK_RIGHT) else None,
        kick_duration=kick_duration,
        kick_speed_commanded=True, kick_speed=speed,
        switch_threshold=0.15,
    )
    if mirror:
        from mirror_policy import MirroredSession
        other = KICK_RIGHT if foot == "left" else KICK_LEFT
        policy.behavior_sessions[f"kick_{foot}"] = MirroredSession(other)
    policy.set_vel_cmd(0.0, 0.0, 0.0)

    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    data.qpos[adr:adr + 7] = [0, 0, 0.125, 1, 0, 0, 0]
    for i, qi in enumerate(policy.joint_qpos_indices):
        data.qpos[qi] = policy.default_pose[i]
    bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    dec, dt = 4, model.opt.timestep
    cdt = dec * dt
    bq, bv = policy.ball_qpos_adr, policy.ball_qvel_adr

    def step():
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(dec):
            if bam_ctrl is not None:
                bam_ctrl.update()
            mujoco.mj_step(model, data)
        policy.update_behavior(cdt)

    for _ in range(int(settle_s / cdt)):
        step()

    policy.trigger_behavior("kick_%s" % foot, kick_speed=speed)
    x0, y0 = float(data.qpos[bq]), float(data.qpos[bq + 1])

    exit_speed, still, t = 0.0, 0.0, 0.0
    tilt_handover = None     # tilt when the kick policy hands back (t = kick_duration)
    t_fall = None            # when the trunk first passed 60 deg
    exit_ang = 0.0
    spin_fwd = 0.0   # spin about the x (forward) axis -> curves the roll in y
    while t < max_s:
        step()
        t += cdt
        _qw, _qx, _qy, _qz = (float(v) for v in data.qpos[adr + 3:adr + 7])
        _tilt = math.degrees(math.acos(max(-1.0, min(1.0,
            1.0 - 2.0 * (_qx * _qx + _qy * _qy)))))
        if tilt_handover is None and t >= kick_duration:
            tilt_handover = _tilt
        if t_fall is None and _tilt > 60.0:
            t_fall = t
        v = data.qvel[bv:bv + 3]
        sp = float(math.hypot(v[0], v[1]))
        if sp > exit_speed:
            exit_speed = sp
            exit_ang = math.degrees(math.atan2(float(v[1]), float(v[0])))
            spin_fwd = float(data.qvel[bv + 3])
        # Stopped = under 1 cm/s for half a second, once it has actually moved.
        still = still + cdt if sp < 0.01 else 0.0
        if still > 0.5 and exit_speed > 0.05:
            break
    x1, y1 = float(data.qpos[bq]), float(data.qpos[bq + 1])
    # Did the duck survive its own kick? Measured once the ball has stopped,
    # by which point the kick policy has long handed back to standing.
    z_end = float(data.qpos[adr + 2])
    qw, qx, qy, qz = (float(v) for v in data.qpos[adr + 3:adr + 7])
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    fell = z_end < 0.07
    rest_ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return dict(foot=foot, cmd=speed, exit=exit_speed, x0=x0, y0=y0, x=x1, y=y1,
                travel=x1 - x0, lat=y1 - y0, t=t, fell=fell,
                exit_ang=exit_ang, rest_ang=rest_ang, spin=spin_fwd,
                z_end=z_end, tilt=tilt,
                tilt_handover=tilt_handover, t_fall=t_fall)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--foot", default="right", choices=("left", "right"))
    ap.add_argument("--mirror", action="store_true",
                    help="Drive the chosen foot with the OTHER foot's policy, "
                         "mirrored left<->right (see scripts/mirror_policy.py)")
    ap.add_argument("--kick-duration", type=float, default=2.0,
                    help="Seconds the kick policy drives before handing back to "
                         "standing (the deployment's kick_duration)")
    ap.add_argument("--speeds", type=float, nargs="+",
                    default=[0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.10, 1.30, 1.50, 1.80])
    args = ap.parse_args()
    print(f"{'foot':6} {'cmd':>5} {'exit':>6} {'ang_out':>8} {'ang_rest':>9} {'travel':>7} {'tilt@2s':>8} {'t_fall':>7} {'tilt_end':>8}")
    for s in args.speeds:
        r = run_one(args.foot, s, kick_duration=args.kick_duration,
                    mirror=args.mirror)
        print(f"{r['foot']:6} {r['cmd']:5.2f} {r['exit']:6.2f} {r['exit_ang']:8.1f} "
              f"{r['rest_ang']:9.1f} {r['travel']:7.3f} "
              f"{(r['tilt_handover'] if r['tilt_handover'] is not None else float('nan')):8.1f} "
              f"{(r['t_fall'] if r['t_fall'] is not None else float('nan')):7.2f} "
              f"{r['tilt']:8.1f}",
              flush=True)
