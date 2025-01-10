cmake_minimum_required(VERSION 2.8.3)
project(sciurus17_examples)

find_package(catkin REQUIRED COMPONENTS
  rospy
  pcl_ros
  roscpp
  sensor_msgs
  pcl_conversions
  visualization_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sciurus17_examples
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(object_detection src/object_detection.cpp)
add_executable(tuantd_object_position src/tuantd_object_position.cpp)
add_executable(tuantd_test_2 src/tuantd_test_2.cpp)
add_executable(basic_shapes src/basic_shapes.cpp)
add_executable(basic_shapes2 src/basic_shapes2.cpp)


target_link_libraries(object_detection
  ${catkin_LIBRARIES}
)

target_link_libraries(tuantd_object_position
  ${catkin_LIBRARIES}
)

target_link_libraries(basic_shapes
  ${catkin_LIBRARIES}
)
target_link_libraries(basic_shapes2
  ${catkin_LIBRARIES}
)
target_link_libraries(tuantd_test_2
  ${catkin_LIBRARIES}
)

#############
## Install ##
#############

install(PROGRAMS
  scripts/pick_and_place_left_arm_demo.py
  scripts/pick_and_place_right_arm_demo.py
  scripts/gripper_action_example.py
  scripts/hand_position_publisher_example.py
  scripts/neck_joint_trajectory_example.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
