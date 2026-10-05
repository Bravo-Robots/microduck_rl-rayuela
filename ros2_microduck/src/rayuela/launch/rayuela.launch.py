from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from rayuela import repo_paths

# Resolved from wherever the checkout lives (rayuela/repo_paths.py); override
# with MICRODUCK_RL_ROOT, or venv_python:=... for a venv outside the repo.
SIM_WORKER_SCRIPT = repo_paths.sim_worker_script()


def generate_launch_description():
    use_viewer = LaunchConfiguration("use_viewer")
    enable_vision = LaunchConfiguration("enable_vision")
    enable_control = LaunchConfiguration("enable_control")
    enable_teleop = LaunchConfiguration("enable_teleop")
    use_bam_bridge = LaunchConfiguration("use_bam_bridge")
    venv_python = LaunchConfiguration("venv_python")

    return LaunchDescription([
        DeclareLaunchArgument("use_viewer", default_value="false"),
        DeclareLaunchArgument("enable_vision", default_value="true"),
        DeclareLaunchArgument("enable_control", default_value="true"),
        DeclareLaunchArgument("enable_teleop", default_value="false"),
        DeclareLaunchArgument(
            "venv_python", default_value=repo_paths.venv_python(),
            description="Interpreter of the project venv (uv sync) that runs sim_worker.py.",
        ),
        DeclareLaunchArgument(
            "use_bam_bridge", default_value="false",
            description=(
                "true: real BAM actuators via sim_worker.py (project venv, "
                "Python 3.12) + rayuela_bridge_node over a Unix socket. "
                "false: single-process sim_node with plain XML position "
                "actuators (lower fidelity, no venv/bam dependency)."
            ),
        ),

        # use_bam_bridge=false path: everything in one rclpy process.
        Node(
            package="rayuela",
            executable="sim_node",
            name="rayuela_sim_node",
            output="screen",
            parameters=[{"use_viewer": use_viewer}],
            condition=UnlessCondition(use_bam_bridge),
        ),

        # use_bam_bridge=true path: rclpy-side bridge + the venv-side MuJoCo
        # worker as a plain subprocess (not a ROS node — it has no rclpy).
        Node(
            package="rayuela",
            executable="bridge_node",
            name="rayuela_bridge_node",
            output="screen",
            condition=IfCondition(use_bam_bridge),
        ),
        ExecuteProcess(
            cmd=[venv_python, SIM_WORKER_SCRIPT, "--use-viewer"],
            output="screen",
            condition=IfCondition(PythonExpression(
                ["'", use_bam_bridge, "' == 'true' and '", use_viewer, "' == 'true'"]
            )),
        ),
        ExecuteProcess(
            cmd=[venv_python, SIM_WORKER_SCRIPT],
            output="screen",
            condition=IfCondition(PythonExpression(
                ["'", use_bam_bridge, "' == 'true' and '", use_viewer, "' == 'false'"]
            )),
        ),

        Node(
            package="rayuela",
            executable="vision_node",
            name="rayuela_vision_node",
            output="screen",
            condition=IfCondition(enable_vision),
        ),
        Node(
            package="rayuela",
            executable="control_node",
            name="rayuela_control_node",
            output="screen",
            condition=IfCondition(enable_control),
        ),
        Node(
            package="rayuela",
            executable="teleop_keyboard",
            name="rayuela_teleop_keyboard",
            output="screen",
            condition=IfCondition(enable_teleop),
            emulate_tty=True,
        ),
    ])
