from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rayuela import paths

# Resolved from wherever the checkout lives (rayuela/paths.py); override
# with MICRODUCK_RL_ROOT, or venv_python:=... for a venv outside the repo.
SIM_WORKER_SCRIPT = str(paths.SIM_WORKER_SCRIPT)


def generate_launch_description():
    use_viewer = LaunchConfiguration("use_viewer")
    enable_vision = LaunchConfiguration("enable_vision")
    vis_rect = LaunchConfiguration("vis_rect")
    vis_raw = LaunchConfiguration("vis_raw")
    vertical_view = LaunchConfiguration("vertical_view")
    enable_control = LaunchConfiguration("enable_control")
    venv_python = LaunchConfiguration("venv_python")

    return LaunchDescription([
        DeclareLaunchArgument("use_viewer", default_value="false", description=(
            "Open the MuJoCo viewer window."
        )),
        DeclareLaunchArgument("enable_vision", default_value="true", description=(
            "Run vision_node (ball detection and casilla reports)."
        )),
        DeclareLaunchArgument("enable_control", default_value="true", description=(
            "Run control_node (walks the duck to the reported casilla)."
        )),
        DeclareLaunchArgument(
            "venv_python", default_value=str(paths.VENV_PYTHON),
            description="Interpreter of the project venv (uv sync) that runs sim_worker.py.",
        ),
        DeclareLaunchArgument("vis_rect", default_value="false", description=(
            "Visualize the rectified camera image."
        )),
        DeclareLaunchArgument("vis_raw", default_value="false", description=(
            "Visualize the raw camera image."
        )),
        DeclareLaunchArgument("vertical_view", default_value="false", description=(
            "Display the rectified image in vertical orientation."
        )),

        # The simulation: sim_worker.py runs MuJoCo + the policies with the BAM
        # actuators they were trained on, under the project venv (Python 3.12).
        # rclpy only exists for ROS's Python 3.10, so the worker is a plain
        # subprocess, not a ROS node; bridge_node carries its traffic to and
        # from ROS topics over a Unix socket (rayuela_ipc.py).
        Node(
            package="rayuela",
            executable="bridge_node",
            name="rayuela_bridge_node",
            output="screen",
        ),
        ExecuteProcess(
            cmd=[venv_python, SIM_WORKER_SCRIPT, "--use-viewer", use_viewer],
            output="screen",
        ),

        Node(
            package="rayuela",
            executable="vision_node",
            name="rayuela_vision_node",
            output="screen",
            parameters=[{"vis_rect": vis_rect, "vis_raw": vis_raw, "vertical_view": vertical_view}],
            condition=IfCondition(enable_vision),
        ),
        Node(
            package="rayuela",
            executable="control_node",
            name="rayuela_control_node",
            output="screen",
            condition=IfCondition(enable_control),
        ),
    ])
