#!/usr/bin/env python
# coding: utf-8

import rospy
import math
import sys
import os
import glob
import cv2
import numpy as np
from std_msgs.msg import Int32
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point


# for NeckYawPitch
import actionlib
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTrajectoryControllerState
)
from trajectory_msgs.msg import JointTrajectoryPoint

RGB_DIR = "image"
if not os.path.exists(RGB_DIR):
    os.makedirs(RGB_DIR)

DEPTH_DIR = "depth"
if not os.path.exists(DEPTH_DIR):
    os.makedirs(DEPTH_DIR)

files = glob.glob(os.path.join(DEPTH_DIR, "*"))
for f in files:
    os.remove(f)

class VideoRecorder:
    def __init__(self):
        self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber("/sciurus17/camera/color/image_raw", Image, self._image_callback, queue_size=1)
        self._depth_sub = rospy.Subscriber("/sciurus17/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback, queue_size=1)
        
	self._image_pub = rospy.Publisher("~output_image", Image, queue_size=1)
        self._median_depth_pub = rospy.Publisher("~output_median_depth", Int32, queue_size=1)

	self._video_name_rgb = 'tuantd_rgb.avi'
	self._video_name_depth = 'tuantd_depth.avi'
	self._fourcc = cv2.VideoWriter_fourcc(*'XVID')
	self._frame_size = (640,480)
	#self._frame_size = (1920,1080)
	self._out_rgb = cv2.VideoWriter(self._video_name_rgb, self._fourcc, 30.0, self._frame_size)
	#self._out_depth = cv2.VideoWriter(self._video_name_depth, self._fourcc, 30.0, self._frame_size, 0)

	self._frame_idx = 0

    def _image_callback(self, ros_image):
        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError, e:
            rospy.logerr(e)
	    return

	cv2.imwrite(os.path.join(RGB_DIR, str(self._frame_idx) + '.png'),input_image)
	self._frame_idx += 1
	
	self._out_rgb.write(input_image)

        #self._image_pub.publish(input_image)

    def _depth_callback(self, ros_image):	
	
        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "passthrough")
        except CvBridgeError, e:
            rospy.logerr(e)
            return
            
	#input_image = input_image.astype(np.uint16)
	#input_image = (input_image * 65535.0) / 10000.0
        #input_image = cv2.resize(input_image, self._frame_size)
	#print(input_image.shape)
	#self._out_depth.write(input_image)
	
	cv2.imwrite(os.path.join(DEPTH_DIR, str(self._frame_idx) + '.png'),input_image)
	self._frame_idx += 1


class NeckYawPitch(object):
    def __init__(self):
        self.__client = actionlib.SimpleActionClient("/sciurus17/controller3/neck_controller/follow_joint_trajectory",
                                                     FollowJointTrajectoryAction)
        self.__client.wait_for_server(rospy.Duration(5.0))
        if not self.__client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Action Server Not Found")
            rospy.signal_shutdown("Action Server not found")
            sys.exit(1)

        self._state_sub = rospy.Subscriber("/sciurus17/controller3/neck_controller/state", 
                JointTrajectoryControllerState, self._state_callback, queue_size=1)

        self._state_received = False
        self._current_yaw = 0.0 # Degree
        self._current_pitch = 0.0 # Degree


    def _state_callback(self, state):
        # 首の現在角度を取得

        self._state_received = True
        yaw_radian = state.actual.positions[0]
        pitch_radian = state.actual.positions[1]

        self._current_yaw = math.degrees(yaw_radian)
        self._current_pitch = math.degrees(pitch_radian)


    def state_received(self):
        return self._state_received


    def get_current_yaw(self):
        return self._current_yaw


    def get_current_pitch(self):
        return self._current_pitch


    def set_angle(self, yaw_angle, pitch_angle, goal_secs=1.0e-9):
        # 首を指定角度に動かす
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["neck_yaw_joint", "neck_pitch_joint"]

        yawpoint = JointTrajectoryPoint()
        yawpoint.positions.append(yaw_angle)
        yawpoint.positions.append(pitch_angle)
        yawpoint.time_from_start = rospy.Duration(goal_secs)
        goal.trajectory.points.append(yawpoint)

        self.__client.send_goal(goal)
        self.__client.wait_for_result(rospy.Duration(0.1))
        return self.__client.get_result()

def hook_shutdown():
    video_recorder._out_rgb.release()
    #video_recorder._out_depth.release()
    # shutdown時に0度へ戻る
    neck.set_angle(math.radians(0), math.radians(0), 3.0)

def main():
    r = rospy.Rate(60)
    rospy.on_shutdown(hook_shutdown)

    # 正規化された座標系(px, px)
    THRESH_X = 0.05
    THRESH_Y = 0.05

    # 首の初期角度 Degree
    INITIAL_YAW_ANGLE = 0
    INITIAL_PITCH_ANGLE = 0

    # 首の制御角度リミット値 Degree
    MAX_YAW_ANGLE   = 120
    MIN_YAW_ANGLE   = -120
    MAX_PITCH_ANGLE = 80
    MIN_PITCH_ANGLE = -70

    # 首の制御量
    # 値が大きいほど首を大きく動かす
    OPERATION_GAIN_X = 5.0
    OPERATION_GAIN_Y = 5.0

    # 初期角度に戻る時の制御角度 Degree
    RESET_OPERATION_ANGLE = 3

    # 現在の首角度を取得する
    # ここで現在の首角度を取得することで、ゆっくり初期角度へ戻る
    while not neck.state_received():
        pass
    yaw_angle = 0 #neck.get_current_yaw()
    pitch_angle = 0 #neck.get_current_pitch()

    move_x = -1.5
    move_y = -13
    if math.fabs(move_x) > THRESH_X:
    	yaw_angle += -move_x * OPERATION_GAIN_X

    if math.fabs(move_y) > THRESH_Y:
    	pitch_angle += move_y * OPERATION_GAIN_Y

    # 首の制御角度を制限する
    if yaw_angle > MAX_YAW_ANGLE:
        yaw_angle = MAX_YAW_ANGLE
    if yaw_angle < MIN_YAW_ANGLE:
        yaw_angle = MIN_YAW_ANGLE

    if pitch_angle > MAX_PITCH_ANGLE:
        pitch_angle = MAX_PITCH_ANGLE
    if pitch_angle < MIN_PITCH_ANGLE:
        pitch_angle = MIN_PITCH_ANGLE

    yaw_angle = 0
    pitch_angle = -70

    neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle), 2.0)

    while not rospy.is_shutdown():
        r.sleep()


if __name__ == '__main__':
    rospy.init_node("video_recorder")
    
    neck = NeckYawPitch()
    video_recorder = VideoRecorder()

    main()

