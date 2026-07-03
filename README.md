# lane-keeping-carla-ros2

    Terminal 1 — Carla:
cd ~/carla
./CarlaUE4.sh -quality-level=Low -ResX=800 -ResY=600



    Terminal 2 — ROS Bridge:
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py timeout:=60



    Terminal 3 — Các node:
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run autonomous_driving perception_node &
ros2 run autonomous_driving planning_node &
ros2 run autonomous_driving control_node

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run autonomous_driving npc_velocity_manager

    build
cd ~/ros2_ws && colcon build --packages-select autonomous_driving && source ~/.bashrc

    cammera sematic
unset GTK_PATH
ros2 run rqt_image_view rqt_image_view
