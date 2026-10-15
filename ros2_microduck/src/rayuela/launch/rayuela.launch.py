from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_viewer = LaunchConfiguration("use_viewer")
    enable_vision = LaunchConfiguration("enable_vision")
    vis_rect = LaunchConfiguration("vis_rect")
    vis_raw = LaunchConfiguration("vis_raw")
    vertical_view = LaunchConfiguration("vertical_view")
    enable_control = LaunchConfiguration("enable_control")

    return LaunchDescription([
        DeclareLaunchArgument("use_viewer", default_value="false", description=(
            "Open the MuJoCo viewer window. Closing it ends the game."
        )),
        DeclareLaunchArgument("enable_vision", default_value="true", description=(
            "Run vision_node (ball detection and casilla reports)."
        )),
        DeclareLaunchArgument("enable_control", default_value="true", description=(
            "Run control_node (walks the duck to the reported casilla)."
        )),
        DeclareLaunchArgument("vis_rect", default_value="false", description=(
            "Visualize the rectified camera image."
        )),
        DeclareLaunchArgument("vis_raw", default_value="false", description=(
            "Visualize the raw camera image."
        )),
        DeclareLaunchArgument("vertical_view", default_value="false", description=(
            "Display the rectified image in vertical orientation."
        )),

        # MuJoCo + the policies, with the BAM actuators they were trained on.
        # Everything else depends on it, so when it exits (Ctrl+C, or the
        # viewer window closed) the whole launch goes down with it.
        Node(
            package="rayuela",
            executable="sim_node",
            name="rayuela_sim_node",
            output="screen",
            parameters=[{"use_viewer": use_viewer}],
            on_exit=Shutdown(),
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
