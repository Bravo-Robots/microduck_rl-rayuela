from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

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
            # ROS's PYTHONPATH holds Python 3.10 packages; keep them out of the
            # venv's 3.12. Unbuffered so prints reach the launch log live.
            additional_env={"PYTHONPATH": "", "PYTHONUNBUFFERED": "1"},
            # Without the simulation the rest of the stack has nothing to do.
            on_exit=Shutdown(),
        ),

        # world -> angled_cam, the MuJoCo angled_cam pose in the ROS optical
        # convention (z forward, y down). Recompute if the camera moves in
        # tablero_rayuela.xml. Fixed as the camera is; canvas frames
        # (topdown_canvas, topdown_debug) are image spaces with no TF.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="rayuela_camera_tf",
            output="screen",
            arguments=[
                "--x", "-0.15", "--y", "-0.9", "--z", "1.5",
                "--qx", "0.87449876", "--qy", "-0.34768202",
                "--qz", "0.12494252", "--qw", "-0.31425864",
                "--frame-id", "world", "--child-frame-id", "angled_cam",
            ],
        ),

        Node(
            package="rayuela",
            executable="vision_node",
            name="rayuela_vision_node",
            output="screen",
            parameters=[{
                "vis_rect": ParameterValue(vis_rect, value_type=bool),
                "vis_raw": ParameterValue(vis_raw, value_type=bool),
                "vertical_view": ParameterValue(vertical_view, value_type=bool),
            }],
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
