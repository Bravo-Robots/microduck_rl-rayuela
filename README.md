# Microduck RL + Rayuela (ROS 2 Humble)

This branch adds **Rayuela**, a hopscotch game for the
[Microduck](https://github.com/pollen-robotics/microduck) robot, as a ROS 2
Humble application (Ubuntu 22.04) on top of the RL policies from
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl).
For Ubuntu 24.04 and ROS 2 Jazzy, use the [rayuela-ros2-jazzy](https://github.com/dasanplaen-cmyk/microduck_rl-rayuela/tree/rayuela-ros2-jazzy) branch instead.



For everything that is not the game (training tasks, the BAM actuator model,
backlash variants, project structure, publishing a policy), see the
[original Pollen Robotics repository](https://github.com/pollen-robotics/microduck_rl).
The reward-design rules and the workflow for creating new tasks are in
[AGENTS.md](AGENTS.md).

**Contents**

- [Rayuela: a hopscotch game](#rayuela-a-hopscotch-game)
  - [Nodes](#nodes) · [Topics](#topics) · [Launch arguments](#launch-arguments) · [Policies](#policies) · [How it works](#how-it-works)
- [Setup](#setup)
  - [0. Requirements](#0-requirements) · [1. Clone](#1-clone) · [2. Project venv](#2-project-venv-terminal-a) · [3. ROS 2 Humble](#3-ros-2-humble-terminal-b) · [4. Build](#4-build-the-ros-2-workspace-terminal-b) · [5. Launch](#5-launch-the-game-terminal-b) · [6. Without ROS 2](#6-without-ros-2-terminal-a)
- [Troubleshooting](#troubleshooting)

## Rayuela: a hopscotch game

<!-- VIDEO: one run. Kick, ball lands on a square, duck walks/rolls to it. -->

The duck kicks a ball onto a hopscotch board, a camera works out which square
(*casilla*) it landed on, and the duck walks, or rolls, to that square. The game
lives in [`ros2_microduck/`](ros2_microduck/) and doubles as the end-to-end
integration test for the whole policy family. Casillas are numbered 1 to 9,
and 10 is the *cielo* (the half-circle at the far end).

https://github.com/user-attachments/assets/3912fb5d-77c5-415f-b958-27979f76b475

### Nodes

| Node | What it does |
|---|---|
| `sim_worker.py` + `bridge_node` | The simulation: MuJoCo and the ONNX policies with the BAM actuators they were trained on, running in the project venv. `bridge_node` connects it to ROS (see [How it works](#how-it-works)) and publishes the camera, the duck pose and `current_policy`. |
| `vision_node` | Rectifies the angled camera to a metric top-down view, finds the ball and reports the casilla it stops on. |
| `control_node` | Drives the duck to the reported casilla, and back to HOME on request. |
| `teleop_keyboard` | Manual driving, tricks, and "kick to casilla N" (digits 1-9, 0 = cielo). |

Source: [`ros2_microduck/src/rayuela/rayuela/`](ros2_microduck/src/rayuela/rayuela/).
Besides one file per node, it holds the shared modules:

| Module | Contents | Imported by |
|---|---|---|
| `sim_setup.py` | Policy loading, walk/stand thresholds, command handling, cameras | `sim_worker` |
| `ros_utils.py` | ROS message builders and the common `main()` | every rclpy node |
| `board_geometry.py` | Board, camera and kick-calibration tables | everything (stdlib only) |
| `kick_command.py` | The `behavior_cmd` kick grammar | `sim_setup`, `vision_node` (stdlib only) |
| `rayuela_ipc.py` | Socket protocol between `sim_worker` and `bridge_node` | both (stdlib only) |
| `paths.py` | Finds the checkout at runtime | everything (stdlib only) |

The custom message `TargetCasilla` is in
[`ros2_microduck/src/rayuela_msgs/`](ros2_microduck/src/rayuela_msgs/).

### Topics

| Topic | Type | Publisher → subscriber |
|---|---|---|
| `/rayuela/cmd_vel` | `geometry_msgs/Twist` | control / teleop → sim |
| `/rayuela/behavior_cmd` | `std_msgs/String` | control / teleop → sim, vision, control |
| `/rayuela/go_home` | `std_msgs/Bool` | teleop → control |
| `/rayuela/duck_pose` | `geometry_msgs/PoseStamped` | sim → control |
| `/rayuela/current_policy` | `std_msgs/String` | sim → control |
| `/rayuela/angled_camera/image_raw` | `sensor_msgs/Image` | sim → vision |
| `/rayuela/angled_camera/camera_info` | `sensor_msgs/CameraInfo` | sim (latched) |
| `/rayuela/topdown_camera/image_raw` | `sensor_msgs/Image` | vision (rectified board) |
| `/rayuela/ball/position` | `geometry_msgs/PointStamped` | vision |
| `/rayuela/target_casilla` | `rayuela_msgs/TargetCasilla` | vision → control |

`behavior_cmd` accepts `kick_left`, `kick_right`, `kick_right:<m/s>` (kick
with a target ball speed), `kick_casilla:<N>` (pick the foot and speed for
casilla N), `roulade`, `ground_pick`, `toggle_sit` and `stop`. For example:

```bash
ros2 topic pub --once /rayuela/behavior_cmd std_msgs/msg/String "{data: 'kick_casilla:7'}"
ros2 topic pub --once /rayuela/go_home std_msgs/msg/Bool "{data: true}"
```

### Launch arguments

| Argument | Default | Effect |
|---|---|---|
| `use_viewer` | `false` | Opens the MuJoCo viewer. |
| `enable_vision` | `true` | Runs `vision_node`. |
| `enable_control` | `true` | Runs `control_node`. |
| `venv_python` | `<repo>/.venv/bin/python3` | Interpreter that runs `sim_worker.py`. |
| `vis_raw` | `false` | Debug window with the angled camera. |
| `vis_rect` | `false` | Debug window with the rectified board and the detected ball. |
| `vertical_view` | `false` | Rotates the rectified board view (and the published `/rayuela/topdown_camera/image_raw`) so the cielo is at the top. |

### Policies

The game loads these ONNX files from [`policies-v1/`](policies-v1/), which is
included in the repo:

| File | Trained by |
|---|---|
| `alpha_walking.onnx` | `Mjlab-Velocity-Flat-MicroDuck` |
| `alpha_standing.onnx` | `Mjlab-StandUp-Flat-MicroDuck` |
| `alpha_sitstand.onnx` | `Mjlab-SitStand-Flat-MicroDuck` |
| `alpha_ground_pick.onnx` | `Mjlab-GroundPick-Flat-MicroDuck` |
| `roulade.onnx` | `Mjlab-Roulade-Flat-MicroDuck` |
| `ball_kick_speed_right.onnx`, `ball_kick_speed_left.onnx` | `Mjlab-BallKickSpeed-{Right,Left}-Flat-MicroDuck` |

If either BallKickSpeed file is missing, the game falls back to
`ball_kick_right.onnx` / `ball_kick_left.onnx` (fixed-strength kicks). To
replace a policy, export your own run with `scripts/export.py` (see the
[Quickstart](https://github.com/pollen-robotics/microduck_rl#quickstart) in
the original repo) and save it under the same file name.

### How it works

**Two Python interpreters.** `rclpy` is only available for Humble's system
Python 3.10, while mujoco, onnxruntime and bam live in the project's Python
3.12 venv. So `sim_worker.py` runs the simulation in the venv, and
`bridge_node` republishes its pose and camera frames as ROS topics and sends
`cmd_vel` and `behavior_cmd` back over a Unix socket. The wire format is in
`rayuela_ipc.py`. That file and `kick_command.py` (the command grammar above)
use only the standard library, because both interpreters import them.
`paths.py` finds the checkout at runtime, so no home directory is hardcoded.

**Vision.** Four coloured fiducials give a one-off homography into a metric
top-down canvas, where the ball is a blob of known size. The ball is picked
by *solidity* rather than size or circularity: the duck is orange too, and at
the far end of the board the homography stretches the ball into a 2:1 ellipse,
which ruins circularity but not convexity. When the duck lies on the ball and
the two blobs merge, a Hough circle search recovers the ball.

**Control.** Every command is bang-bang, because the gait has measured dead
zones: no net motion below 0.25 m/s, and no in-place rotation below about
1.5 rad/s. A command that tapers to zero looks exactly like a frozen
controller.

**Kick calibration.** `scripts/kick_sweep.py` runs the kick policy in the
loop and measures what each commanded speed actually does: travel, exit
speed, off-axis angle, and whether the duck stayed on its feet. These sweeps
produce the per-foot tables in `board_geometry.py` that map a casilla to a
kick command, and they are the acceptance test for a retrained kick.

**Mirrored policies.** The two kick tasks are mirror images and the robot is
symmetric, so `scripts/mirror_policy.py` turns the right-foot policy into a
left-foot one by permuting and sign-flipping the 61-D observation and the 14-D
action, and can bake that into a standalone `.onnx`. The mirrored policy beat
the separately trained left policy (+1.5° of drift mid-range against +19.6°)
and is what makes all ten casillas reachable.

## Setup

The project uses **two separate Python environments**, each in its own
terminal:

| Terminal | Environment | Python | Contains | Used for |
|---|---|---|---|---|
| A | Project venv (`.venv/`, managed by `uv`) | 3.12 | mjlab, MuJoCo Warp, torch, rsl_rl, onnxruntime, BAM | Training, ONNX export, simulation with BAM actuators |
| B | ROS 2 Humble (system Python) | 3.10 | rclpy, messages, colcon, OpenCV | The rayuela nodes (vision, control, teleop, bridge) |

**Keep them apart.** Do not source ROS in terminal A: ROS's `PYTHONPATH`
(Python 3.10 packages) leaks into the 3.12 venv. If your `~/.bashrc` runs
`source /opt/ros/humble/setup.bash`, start terminal A with `unset PYTHONPATH`.
Likewise, do not activate the venv in terminal B (VS Code may do it for you):
`colcon build` then picks up the wrong Python.

### 0. Requirements

- Ubuntu 22.04 (the platform for ROS 2 Humble).
- An NVIDIA GPU, **only for training**. The game and `infer_policy.py` run on CPU.
- About 10 GB of free disk space.

### 1. Clone

```bash
git clone -b rayuela-ros2-humble https://github.com/dasanplaen-cmyk/microduck_rl-rayuela ~/microduck_rl-rayuela 
cd ~/microduck_rl-rayuela
```

The nodes find the repo on their own as long as you build `ros2_microduck/`
inside the clone (step 4). Otherwise, export
`MICRODUCK_RL_ROOT=/path/to/microduck_rl-rayuela`.

### 2. Project venv (terminal A)

```bash
unset PYTHONPATH                                  # only if your ~/.bashrc sources ROS
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
exec $SHELL                                       # reload PATH so `uv` is found
cd ~/microduck_rl-rayuela
uv sync                                           # creates .venv/ with Python 3.12
```

On ARM machines (DGX Spark, Jetson), run `export UV_HTTP_TIMEOUT=600` first.

Check the install:

```bash
uv run list-envs        # should list Mjlab-BallKickSpeed-*
uv run --with pytest pytest tests/test_ball_kick_speed_cfg.py tests/test_rayuela_paths.py
uv run train Mjlab-BallKickSpeed-Right-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5     # smoke test, needs a GPU
```

### 3. ROS 2 Humble (terminal B)

Install Humble by following the official guide:
https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

Then install the extra packages the game needs:

```bash
sudo apt install -y python3-colcon-common-extensions python3-numpy python3-opencv
```

### 4. Build the ROS 2 workspace (terminal B)

```bash
source /opt/ros/humble/setup.bash
cd ~/microduck_rl-rayuela/ros2_microduck
colcon build --symlink-install && source install/setup.bash
ros2 interface show rayuela_msgs/msg/TargetCasilla   # checks the custom message
```

### 5. Launch the game (terminal B)

```bash
ros2 launch rayuela rayuela.launch.py use_viewer:=true
```

The simulation runs with the venv's Python (`.venv/bin/python3`), so step 2
must be done first. If your venv is somewhere else, add
`venv_python:=/path/to/python3`. To see what the vision node detects, add
`vis_rect:=true` (rectified board with the ball), `vis_raw:=true` (angled
camera) and `vertical_view:=true` (board shown vertically, cielo at the top).

Run the keyboard teleop in a third terminal with ROS sourced:

```bash
source /opt/ros/humble/setup.bash
source ~/microduck_rl-rayuela/ros2_microduck/install/setup.bash
ros2 run rayuela teleop_keyboard
```

| Key | Action |
|---|---|
| Arrows | Walk forward / backward, turn left / right |
| Space | Zero the velocity command |
| `1`-`9`, `0` | Kick the ball to casilla 1-9, or to the cielo (10) |
| `k` / `l` | Kick with the left / right foot |
| `r` | Roulade (forward roll) |
| `g` | Ground pick |
| `y` | Sit / stand |
| `h` | Go back to HOME |
| `x` | Stop everything (behaviour, target, go-home) |
| `q` | Quit |

### 6. Without ROS 2 (terminal A)

A simpler simulation to try the trained policies, with no ROS involved:

```bash
cd ~/microduck_rl-rayuela
uv run scripts/infer_policy.py --new-cmd-obs --walking policies-v1/alpha_walking.onnx \
    --standing policies-v1/alpha_standing.onnx --roulade policies-v1/roulade.onnx \
    --sitstand policies-v1/alpha_sitstand.onnx --ground-pick policies-v1/alpha_ground_pick.onnx \
    --kick-right policies-v1/ball_kick_speed_right.onnx --kick-left policies-v1 ball_kick_speed_left.onnx \
    --kick-speed-commanded --kick-speed 0.3
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: rclpy` inside `uv run` | Expected: rclpy is not in the venv | Run ROS commands in terminal B |
| Odd import errors in `uv run` | ROS is sourced in that terminal | `unset PYTHONPATH`, or open a terminal without `source /opt/ros/...` |
| `ModuleNotFoundError: No module named 'em'` during `colcon build` | The project venv is active in terminal B, so CMake uses its Python | `deactivate`, or open a fresh terminal |
| `No module named 'lark'` when pytest starts | With ROS sourced, pytest loads ROS's pytest plugins (Python 3.10) | `pyproject.toml` already blocks them; if another plugin shows up, `unset PYTHONPATH` |
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` in a ROS node | A NumPy 2 in `~/.local` (from some earlier `pip install --user`) shadows apt's NumPy 1, which `python3-opencv` was built against | In terminal B: `python3 -m pip install --user opencv-python` |
| `IndexError` in `select_gpus()` when training on ARM | Torch without CUDA | See the aarch64 section of [AGENTS.md](AGENTS.md) |
| `sim_worker.py` does not start | `<repo>/.venv` does not exist | Run `uv sync` (step 2) or pass `venv_python:=...` |
| `Cannot find the microduck_rl checkout` | The ROS 2 workspace was built outside the clone | Build inside `ros2_microduck/` as in step 4, or `export MICRODUCK_RL_ROOT=/path/to/microduck_rl-rayuela` |
| The duck does not move in ROS | ONNX files missing from `policies-v1/` | Restore them from the repo (`git checkout -- policies-v1/`) or export your own ([Policies](#policies)) |

## License

Apache 2.0, see [LICENSE](LICENSE). Based on
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl).
