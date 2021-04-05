# Cloth Region Segmentation for Robust Grasp Selection ROS code

ROS code for running the segmentation model from [cloth_region_seg_training](https://github.com/thomasweng15/cloth-region-seg-training) on a robot.

# Installation
Requirements
* Python 2.7
* Ubuntu 16.04 with ROS Kinetic
* Sawyer Robot
* WSG-32 Gripper
* Azure Kinect DK

Here we provide a list of required catkin packages. See the [Sawyer SDK docs](https://sdk.rethinkrobotics.com/intera/Workstation_Setup), [WSG-32 manual](https://schunk.com/fileadmin/pim/docs/IM0014443.PDF), and [Azure Kinect docs](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) for details on robot, gripper, and sensor setup respectively.
- [intera_common](https://github.com/RethinkRobotics/intera_common)
- [intera_sdk](https://github.com/RethinkRobotics/intera_sdk)
- [sawyer_robot](https://github.com/RethinkRobotics/sawyer_robot)
- [sawyer_simulator](https://github.com/RethinkRobotics/sawyer_simulator)
- [sawyer_moveit](https://github.com:RethinkRobotics/sawyer_moveit)
- [sawyer_wsg_moveit](https://github.com/thomasweng15/sawyer_wsg_moveit)
- [wsg50-ros-pkg](https://github.com/thomasweng15/wsg50-ros-pkg)
- [Azure Kinect ROS Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver)

1. Install the above packages in a catkin workspace. Move this repo into the workspace as well.  
2. Install python packages for this repo (`pip install -r requirements.txt`).
3. Build the workspace.
4. Update `config/config.yaml`
    - Update camera intrinsics and extrinsics
    - Update path to network weights, pre-trained weights can be downloaded [here](https://drive.google.com/file/d/1XHGQjz4tGabmkx_VNef5OCHhqZP2Rk94/view?usp=sharing)- Note that the pre-trained weights may not work well on other setups.
5. Set collision geometries by editing `scripts/init/init_collision_geometries.py` or importing your own `collision.scene` into RViz.

# Execution
Simulation
1. Run gazebo simulation `roslaunch clothgrasp gazebo_electricgripper.launch`
2. Start rviz `roslaunch clothgrasp rviz.launch is_sim:=true transform_gripper:=false`
3. `roslaunch clothgrasp sensors.launch is_sim:=true`
4. `roslaunch clothgrasp grasping.launch is_sim:=true`
5. Execute a sliding grasp on a cloth `rosrun clothgrasp execute_grasp.py`

All commands should be run in terminals that have sourced the Sawyer robot (`source intera.sh`, see the [SDK docs](https://sdk.rethinkrobotics.com/intera/Workstation_Setup)).
1. Start rviz `roslaunch clothgrasp rviz.launch`
2. Start the Azure sensor `roslaunch clothgrasp sensors.launch`
3. Start the grasping scripts `roslaunch clothgrasp grasping.launch`
4. Execute a sliding grasp on a cloth `rosrun clothgrasp execute_grasp.py`
