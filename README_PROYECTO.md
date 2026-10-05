# Microduck RL

<img width="2215" height="884" alt="image" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />


RL training environments for [Microduck](https://github.com/pollen-robotics/microduck) —
a ~800 g, ~25 cm tall bipedal robot — built on
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) with PPO.
Policies are trained here at 50 Hz, exported to ONNX, and deployed on the real
robot by the runtime in [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck).

<!-- HERO VIDEO — real robot montage: walking, standup, roulade, roller skating.
     Keep it short (~30 s) and real-robot-first: this is the "why should I care" shot. -->

https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

The repo encodes the full sim2real recipe: [BAM](https://github.com/Rhoban/bam)
actuator physics, domain randomization, backlash simulation, and the
reward-design lessons that made it work
(see [AGENTS.md](AGENTS.md) for the distilled playbook).

Installation (the uv venv and an isolated ROS 2 Humble for the rayuela game)
is in [README.md](README.md), in Spanish. This file describes what is in the
repo and how to use it once installed.

## Train, watch, export

Training needs a CUDA GPU (it runs through MuJoCo Warp); everything after the
export runs on CPU.

```bash
# train the walking policy (uses your GPU; ~1-2 h for a usable gait at 4096 envs)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# watch a trained policy in the viewer
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# export to ONNX for deployment
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
uv run publish --onnx output.onnx --repo <user>/microduck-<name> --kind episodic --duration-s 4.0   # share it (see "Publishing a policy")

# drive the exported policy in CPU MuJoCo with the keyboard
uv run scripts/infer_policy.py --walking output.onnx
```

Resume from a checkpoint:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

No GPU? Add `--hf-jobs` to any train command to run it on Hugging Face Jobs
instead of locally (see [scripts/hf/README.md](scripts/hf/README.md)).

## Tasks

`uv run list-envs` prints the live registry. Flat/Rough variants exist where noted.

<!-- SHOWCASE GRID — one short GIF per task family (sim or real), 3 per row.
     Priority order if you only record a few: Velocity, VelStand (fall+recover),
     Roulade, SitStand, Rollers/Swizzle, BallKick. -->

| Task id | Terrain | Description |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | flat/rough | **The main task**: walking with velocity commands + head-pose commands |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | flat/rough | Walking + fall recovery in one policy |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | flat/rough | Stand up from face-down/face-up/sitting, then hold the stand + body-pose control |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | flat/rough | Commanded sit ↔ stand in one policy, gently, head commandable |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | flat/rough | Crouch and touch the ground with the mouth tip, return to stand |
| `Mjlab-BallKick-Flat-MicroDuck` | flat | Kick a 70 mm / 15 g ball forward (actor is ball-blind) |
| `Mjlab-BallKickSpeed-{Left,Right}-Flat-MicroDuck` | flat | The same kick with a **commanded strength**: target ball exit speed rides in the twist vx slot |
| `Mjlab-Roulade-Flat-MicroDuck` | flat | Forward roll over the head, land back on the feet |
| `Mjlab-OneLeggedStand-Flat-MicroDuck` | flat | Balance on one foot, the other tucked up, head free to counterbalance |
| `Mjlab-OneLeggedHop-Flat-MicroDuck` | flat | The same stand, then hopping on it (warm-starts from a Stand checkpoint) |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | flat | Roller-skate velocity tracking (passive wheels under the feet) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | flat | Classic symmetric swizzle skating |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | flat | Crouch while gliding on rollers |
| `Mjlab-RollerSlope-Flat-MicroDuck` | slope | Glide down slopes on rollers |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | flat | Stand up from the ground onto the wheels |
| `Mjlab-Spin-Flat-MicroDuck` | flat | Fast spin in place on rollers |

At deployment the runtime hot-swaps these policies (walk / recover / trick)
behind a shared 61-dimensional observation contract, so any of them can take
over the robot at any moment. `scripts/infer_policy.py` rehearses exactly that:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

Keyboard-driven (velocity commands, `G` ground pick, `Y` sit/stand, `R` roulade,
`K`/`L` kicks); `--debug`, `--save-csv`, `--record` support sim2real comparisons.
The servos are simulated with the same BAM M6 XL330 model the policies are
trained against (voltage control + load-dependent friction, via
`bam.mujoco.MujocoController`); `--vin` / `--vin-drop-gain` / `--kp-fw` pin the
training DR ranges to one value, `--no-bam` falls back to the XML PD actuators.

### Backlash variants

Every main task has a **Backlash** twin that trains on a model with ±1° of gear
play (2° total) in series with each of the 14 servo joints: insert `-Backlash`
before `MicroDuck` in the task id, e.g. `Mjlab-Velocity-Flat-Backlash-MicroDuck`.

The backlash is modeled properly for sim2real: each servo gets an unactuated
`passive_<joint>_backlash` hinge, and because the real encoder sits on the
output side of the play, both the firmware PD emulation
(`BacklashEncoderBamActuator`) and the `joint_pos`/`joint_vel` observations
read *through* the backlash (`qpos[servo] + qpos[backlash]`). Observation and
action dims are unchanged, so ONNX export and the runtime need no changes.
See `src/mjlab_microduck/tasks/backlash.py`.

## Rayuela — a hopscotch game

<!-- VIDEO — one run: kick, ball lands on a square, duck walks/rolls to it. -->

`ros2_microduck/` is a ROS 2 (Humble) application built on top of these
policies: the duck kicks a ball onto a hopscotch board, a camera works out
which square it landed on, and the duck walks — or rolls — to that square.
It is the end-to-end integration test for the whole policy family. Building
and launching it is covered in [README.md](README.md).

The nodes talk on `/rayuela/*` topics:

| Node | Does |
|---|---|
| `sim_node` | MuJoCo + the ONNX policies in one rclpy process, with plain XML position actuators; publishes the camera, the duck pose and `current_policy` |
| `bridge_node` + `sim_worker.py` | The same simulation with the BAM actuators the policies were trained on (`use_bam_bridge:=true`), split across two interpreters (below) |
| `vision_node` | Rectifies the angled camera to a metric top-down view, finds the ball, reports the casilla |
| `control_node` | Drives the duck to the reported casilla and back to HOME |
| `teleop_keyboard` | Manual driving, tricks, and "kick to casilla N" (digits 1-9, 0 = cielo) |

Launch arguments: `use_bam_bridge`, `use_viewer`, `enable_vision`,
`enable_control`, `enable_teleop`, and `venv_python` (the interpreter that runs
`sim_worker.py`, `<repo>/.venv/bin/python3` by default).

**Python split.** `rclpy` lives in the system Python 3.10 while mujoco /
onnxruntime / bam live in the project's 3.12 venv. So `sim_worker.py` runs the
simulation in the venv, and `bridge_node` republishes its pose and camera
frames as ROS topics and forwards `cmd_vel` / `behavior_cmd` back down a Unix
socket. The wire format is in `rayuela_ipc.py`; it and `kick_command.py`
(the `kick_right:1.25` / `kick_casilla:7` command grammar) are stdlib-only
because both interpreters import them. `paths.py` finds the checkout at
runtime, so nothing hardcodes a home directory.

### Policies the game loads

The simulation reads them from `policies-v1/` (git-ignored; unzip
`policies-v1.zip` there):

| File | Trained by |
|---|---|
| `alpha_walking.onnx` | `Mjlab-Velocity-Flat-MicroDuck` |
| `alpha_standing.onnx` | `Mjlab-StandUp-Flat-MicroDuck` |
| `alpha_sitstand.onnx` | `Mjlab-SitStand-Flat-MicroDuck` |
| `alpha_ground_pick.onnx` | `Mjlab-GroundPick-Flat-MicroDuck` |
| `roulade.onnx` | `Mjlab-Roulade-Flat-MicroDuck` |
| `ball_kick_speed_right.onnx`, `ball_kick_speed_left.onnx` | `Mjlab-BallKickSpeed-{Right,Left}-Flat-MicroDuck` |

Without the two BallKickSpeed files the game falls back to
`ball_kick_right.onnx` / `ball_kick_left.onnx` (fixed-strength kicks). To
replace any of them, export your own run with `scripts/export.py` (see
[Train, watch, export](#train-watch-export)) under the same file name.

**Vision.** Four coloured fiducials give a one-off homography into a metric
top-down canvas, where the ball is a blob of known size. It is picked by
*solidity*, not size or circularity: the duck is orange too, and at the far end
of the board the homography stretches the ball into a 2:1 ellipse that ruins
circularity but not convexity. When the duck lies on the ball and the two blobs
fuse, a Hough fallback recovers the circle.

**Control.** Every command is bang-bang, because the gait has measured dead
zones: no net motion below 0.25 m/s commanded, and no in-place rotation below
~1.5 rad/s. Tapering a command to zero looks exactly like a frozen controller.

**Kick calibration.** `scripts/kick_sweep.py` measures, policy in the loop,
what each commanded speed actually does — travel, exit speed, off-axis angle,
and whether the duck stayed on its feet. Those sweeps are the source of the
per-foot tables in `board_geometry.py` that map a casilla to a kick command,
and the acceptance test for a retrained kick.

**Mirrored policies.** The two kick tasks are exact mirror images and the robot
is bilaterally symmetric, so `scripts/mirror_policy.py` turns the right-foot
policy into a left-foot one — permuting and sign-flipping the 61-D observation
in and the 14-D action out — and can bake that into a standalone `.onnx`. Here
it beat the separately trained left policy (+1.5 deg of drift mid-range against
+19.6) and is what makes all ten casillas reachable.

## Actuator model

All tasks use the [BAM](https://github.com/Rhoban/bam) M6 actuator model for
the Dynamixel XL330 (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent
friction), with per-env domain randomization on battery voltage, voltage sag
under load, command delay, and friction magnitude
(`FrictionDRBamActuator` in `src/mjlab_microduck/actuator/`).

At this scale — tiny servos driving a ~800 g biped — actuator fidelity is most
of the sim2real gap, which is why the actuator is modeled down to its voltage
control law instead of an ideal PD.

## Robot models

MJCF models live in `src/mjlab_microduck/robot/microduck/` and are exported
from Onshape with [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot),
one `config_mjcf_*.json` per model:

| XML | Used by |
|---|---|
| `robot_walk.xml` | Velocity (stripped trunk/head contacts — falling is cheap) |
| `robot_groundcontact.xml` | VelStand, StandUp, SitStand, GroundPick, BallKick, Roulade (curated collision set for the parts that touch the floor — body can physically lie on the ground; formerly `robot_allcollisions.xml`) |
| `robot_groundcontact_rollers.xml` | Roller tasks (passive wheels) |
| `robot_allcollisions.xml` | True full-collision model — every part has a collision geom. No task uses it yet |
| `robot_*_backlash.xml` | Backlash task variants (generated by `add_backlash.py`) |

`scene*.xml` files wrap the robots with a floor + keyframes (STAND/SIT/FOLD)
for quick viewing and for `infer_policy.py`.

<!-- IMAGE — side-by-side render: walk model vs rollers model (or a collision-geom
     visualization). One image here makes the model-variant story instant. -->

## Project structure

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF exports, export configs, scenes, add_backlash.py
│   └── microduck_constants.py        # robot cfgs, HOME frame, BAM actuator cfg
├── actuator/friction_dr_bam.py       # BAM + friction DR + backlash encoder feedback
├── tasks/
│   ├── __init__.py                   # task registration (base + backlash variants)
│   ├── mdp.py                        # rewards, events, observations, custom classes
│   ├── backlash.py                   # make_backlash_variant() env-cfg wrapper
│   ├── symmetry.py                   # 61-D left/right mirror tables (mirror loss, mirror_policy.py)
│   ├── slope_terrain.py              # flat + ramp terrain for RollerSlope
│   ├── testbench_env_cfg.py          # single-XL330 test bench env (testbench_sim2real.py)
│   └── microduck_*_env_cfg.py        # one cfg module per task family
├── export.py                         # checkpoint -> ONNX with the obs normalizer baked in
├── publish/                          # `uv run publish`: manifest + checks + Hub upload
├── sim/                              # `uv run duck-body`: a simulated body (+ camera, ToF) served to the real robotd
├── train_cli.py                      # `train` script (identical to mjlab's)
├── train_hook.py                     # intercepts `train ... --hf-jobs`
└── hf_jobs.py                        # Hugging Face Jobs submission

scripts/                              # standalone tools, see "Scripts" below
tests/                                # CPU-only regression tests
docs/                                 # design notes and plans
policies-v1.zip                       # the ONNX policies the rayuela game loads

ros2_microduck/src/                   # the rayuela game (ROS 2 Humble)
├── rayuela_msgs/                     # TargetCasilla message
└── rayuela/
    ├── launch/rayuela.launch.py
    └── rayuela/
        ├── sim_node.py               # MuJoCo + policies in one rclpy process (XML actuators)
        ├── sim_worker.py             # the same with BAM actuators, in the venv
        ├── bridge_node.py            # ROS side of sim_worker
        ├── rayuela_ipc.py            # Unix-socket wire format between the two
        ├── kick_command.py           # behavior_cmd kick grammar
        ├── paths.py                  # finds the checkout at runtime
        ├── vision_node.py            # rectify, find the ball, report the casilla
        ├── control_node.py           # drive to the casilla and back to HOME
        ├── board_geometry.py         # board layout, camera, measured kick tables
        └── teleop_keyboard.py        # manual driving and tricks
```

Conventions worth knowing:

- The observation layout is shared across every policy (61-dim actor obs:
  48 proprioception + commands `[twist(3), head_pose(4), body_pose(6)]`), which
  is what makes runtime policy hot-swapping possible. Envs that don't use a
  command slot zero-pad it rather than dropping it.
- Unactuated joints are all named `passive_*` (roller wheels, backlash
  hinges); actuators, joint observations and pose rewards select servo joints
  with `^(?!passive_).*`.
- Domain-randomization toggles are `ENABLE_*` booleans at the top of each
  env cfg file.
- Joint layout (14 servos): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll),
  9–13 right leg.
- The exporter bakes the observation normalizer into the ONNX graph — always
  deploy ONNX produced by `scripts/export.py`, never a hand-converted
  checkpoint, or the policy sees unnormalized observations at runtime.

[AGENTS.md](AGENTS.md) documents the env-building workflow and the reward-design
rules learned across the project (also aimed at AI coding agents working in
this repo).

## Scripts

Run them from the repo root with `uv run python scripts/<name>.py --help` for
the full options.

| Script | What it is for |
|---|---|
| **Policies** | |
| `export.py` | Checkpoint (local or wandb) to ONNX with the observation normalizer baked in. The only safe export path |
| `infer_policy.py` | CPU MuJoCo rehearsal of a deployment: hot-swaps walk / stand / tricks / kicks from the keyboard, BAM actuators as in training (`--no-bam` for XML PD). `--save-csv` and `--record` log runs for sim2real |
| `mirror_policy.py` | Turns a one-footed policy into its mirror twin (right kick to left kick); `--export SRC DST` bakes the mirror into a standalone `.onnx` |
| `play_latest.py` | Finds a user's latest run in the `pollen-robotics/mjlab_microduck` wandb project (optionally only `--crouch`, `--roller`, `--swizzle` or `--slope` runs) and launches `uv run play` on it. Helpers in `wandb_utils.py` |
| **Evaluation** | |
| `kick_sweep.py` | Acceptance test for BallKickSpeed: commanded speed against what the ball actually does on the rayuela board. Source of the kick tables in `board_geometry.py` |
| `onelegged_eval.py` | Headless eval of a OneLeggedStand / OneLeggedHop checkpoint in the real training env: episode length and end cause per spawn type, with `--push` and `--no-com-dr` to separate balance from DR |
| **Building envs** | |
| `crouch_pose_editor.py` | Viewer sliders to compose the roller crouch pose; prints the `CROUCH_POSE` dict to paste into the cfg |
| `view_slope_terrain.py` | Opens the RollerSlope ramp terrain in the viewer (no policy needed) to check its geometry |
| **Several ducks** | |
| `build_multi_duck_scene.py` | Builds an MJCF scene with N copies of the duck from `scene.xml` (`--layout line/race/track`) |
| `multi_duck_controller.py` | Runs the same walking ONNX on every duck of that scene |
| **Sim2real / actuator** | |
| `testbench_sim2real.py` | One XL330 on a test bench: runs a policy in sim (BAM) or on the real servo (rustypot), then plots sim against real |
| `validate_bam_testbench.py` | Replays real test-bench recordings in MuJoCo with the BAM M6 model and compares the traces (expects a BAM checkout with its data in `~/Rhoban/bam`) |
| `plot_observations_comparison_plotly.py` | Plots real against simulated observation logs (pickles such as the ones `infer_policy.py --record` writes) |
| **Hugging Face Jobs** | |
| `hf/` | Remote training on HF GPUs: `train_hf.py` (old entry point; prefer `uv run train ... --hf-jobs`) and `uploader.py` (checkpoint uploader inside the job). See [scripts/hf/README.md](scripts/hf/README.md) |

Two entry points live in the package instead of `scripts/`: `uv run publish`
(next section) and `uv run duck-body`, which serves a simulated duck (body,
camera, ToF sensor) to the real `robotd` daemon over TCP, so the full onboard
stack can run against MuJoCo (see `src/mjlab_microduck/sim/body_server.py`).

## Publishing a policy

`uv run publish` puts a policy on the Hugging Face Hub in the shape the robot's
daemon loads: one `policy.onnx` with the observation normalizer baked in, a
`manifest.json` following schema 2 of the
[microduck policy manifest](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md),
and a README saying how to run it. Anyone with a microduck can then install it
with one command, no daemon release needed.

```bash
# From a wandb run — exports through the one safe path, then uploads
uv run publish --task Mjlab-PoliteBow-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000 \
    --repo <user>/microduck-polite-bow --kind episodic --duration-s 4.0 \
    --description "Bows from a two-foot stand and comes back up."

# From an ONNX you already exported (validated, not re-exported)
uv run publish --onnx output.onnx --repo <user>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5 --twist-help "[flag, side, 0]"

# A new gait for a slot
uv run publish --onnx output.onnx --repo <user>/microduck-my-walk --kind perpetual --slot walk

# See what would be uploaded without touching the Hub
uv run publish --onnx output.onnx --repo <user>/microduck-bow --kind episodic --duration-s 4.0 --dry-run
```

Then on a robot:

```bash
sudo robotctl policy add polite-bow <user>/microduck-polite-bow   # episodic: length comes from the manifest
sudo robotctl policy add flamingo <user>/microduck-flamingo --hold 5   # held pose: you pick how long
sudo robotctl policy load walk <user>/microduck-my-walk                # gait: into the walk slot
robotctl robot do polite-bow
```

What `--kind` means, and what each needs:

- **episodic** — runs for `--duration-s` and returns itself to a standing pose
  (kicks, roulade, a bow). Add `--chain` if holding the button should repeat it.
- **perpetual** — runs until told otherwise. Two shapes:
  - a **gait** (a new walk or stand): add `--slot walk` (or `stand`) and
    nothing else; the owner installs it with `robotctl policy load walk <repo>`.
  - a **held pose** (the flamingo): give `--unwind-s`, how long the daemon
    drives the idle twist (`--idle`, zeros by default) before handing back to
    the gait, so the robot is not let go of on one foot. The owner runs it as a
    one-shot with `policy add ... --hold <seconds>`.

Before anything is uploaded, `publish` checks the graph is `[1,61] -> [1,14]`
(a 51-D legacy policy is refused with a message), runs it on plausible inputs
and refuses NaNs or a constant output, fills the `training` block from git and
wandb (task, commit, branch, dirty flag, run, checkpoint), and refuses to
overwrite an existing `.onnx` in the repo without `--force`. Repos are created
private; `--no-private` for public, `--tag v1` to tag the revision.

Only constant-command policies are publishable this way. Phase-driven moves
(the ground pick) and the posture-flag sit↔stand are driven by the daemon
itself and live in the official set, `pollen-robotics/microduck-policies`.

## Tests

```bash
uv run --with pytest pytest tests/
```

CPU-only config-invariant and reward-function regression tests — they lock in
joint-index mappings, reward sign conventions, and NaN guards.

## Related projects

- [microduck](https://github.com/pollen-robotics/microduck) — the Microduck project home, including the onboard runtime that runs the exported policies
- [mjlab](https://github.com/mujocolab/mjlab) — the training framework (MuJoCo Warp + rsl_rl)
- [BAM](https://github.com/Rhoban/bam) — better actuator models, by Rhoban

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
3D model files are licensed under Creative Commons BY-SA-NC.
