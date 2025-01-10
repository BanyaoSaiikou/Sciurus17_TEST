#! /usr/bin/env python
# coding: utf-8

import rospy
import math
import sys
import pickle
import copy
import socket
import cv2

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped, Pose, Vector3
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# for NeckYawPitch
import actionlib
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTrajectoryControllerState
)
from trajectory_msgs.msg import JointTrajectoryPoint

# for Stacker
import moveit_commander
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

# 首の初期角度 degree
INITIAL_YAW_ANGLE = 0
INITIAL_PITCH_ANGLE = -78

OBJECT_SIZE_MAX = 0.07
OBJECT_SIZE_MIN = 0.03

box_zero_positions = [[0,0,0],[0,0,0],[0,0,0]]
last_box_positions = copy.deepcopy(box_zero_positions)

clicked_x = -1
clicked_y = -1

def mouse_clicked(event,x,y,flags,param):
	global clicked_x
	global clicked_y
	if event == cv2.EVENT_LBUTTONDOWN:
	    print "x", x, "y", y
	    clicked_x = x
	    clicked_y = y

class DepthTo3D(object):
    def __init__(self):
	self._bridge = CvBridge()
        self._image_sub = rospy.Subscriber("/sciurus17/camera/color/image_raw", Image, self._image_callback, queue_size=1)
        self._depth_sub = rospy.Subscriber("/sciurus17/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback, queue_size=1)
        self._info_sub = rospy.Subscriber("/sciurus17/camera/aligned_depth_to_color/camera_info", CameraInfo, self._info_callback, queue_size=1)

    def _info_callback(self, info):
        #print info	
	self.cam_intrinsics = info.K
    
    def _depth_callback(self, ros_image):

        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "passthrough")
        except CvBridgeError, e:
            rospy.logerr(e)
            return
	
        self.depth_img = input_image

    def _image_callback(self, ros_image):
        try:
            input_image = self._bridge.imgmsg_to_cv2(ros_image, "bgr8")
	    
            cv2.circle(input_image, (clicked_x,clicked_y), 2, (255, 255, 0), -1)
	    cv2.namedWindow('image')
	    cv2.setMouseCallback('image', mouse_clicked)
	    cv2.imshow('image', input_image)
            cv2.waitKey(1)
        except CvBridgeError, e:
            rospy.logerr(e)
	    return

class NeckYawPitch(object):
    def __init__(self):
        self._client = actionlib.SimpleActionClient("/sciurus17/controller3/neck_controller/follow_joint_trajectory",
                                                     FollowJointTrajectoryAction)
        self._client.wait_for_server(rospy.Duration(5.0))
        if not self._client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Action Server Not Found")
            rospy.signal_shutdown("Action Server not found")
            sys.exit(1)


    def set_angle(self, yaw_angle, pitch_angle, goal_secs=1.0e-9):
        # 首を指定角度に動かす
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["neck_yaw_joint", "neck_pitch_joint"]

        yawpoint = JointTrajectoryPoint()
        yawpoint.positions.append(yaw_angle)
        yawpoint.positions.append(pitch_angle)
        yawpoint.time_from_start = rospy.Duration(goal_secs)
        goal.trajectory.points.append(yawpoint)

        self._client.send_goal(goal)
        self._client.wait_for_result(rospy.Duration(0.1 if goal_secs < 0.05 else 1.5))
        return self._client.get_result()


class Stacker(object):
    _RIGHT_ARM = 1
    _LEFT_ARM = 2
    # グリッパの開閉度
    _GRIPPER_OPEN = 0.9
    _GRIPPER_CLOSE = 0.34 #0.42

    def __init__(self):
        self._my_object_sub = rospy.Subscriber("/sciurus17/example/my_object",
                Vector3, self._my_object_callback, queue_size=1)

	self._point_pub = rospy.Publisher("/sciurus17/example/my_point", Vector3, queue_size=1)

        self._right_arm = moveit_commander.MoveGroupCommander("r_arm_waist_group")
        self._right_arm.set_max_velocity_scaling_factor(0.1)
        self._right_gripper = actionlib.SimpleActionClient("/sciurus17/controller1/right_hand_controller/gripper_cmd", GripperCommandAction)
        self._right_gripper.wait_for_server()

        self._left_arm = moveit_commander.MoveGroupCommander("l_arm_waist_group")
        self._left_arm.set_max_velocity_scaling_factor(0.1)
        self._left_gripper = actionlib.SimpleActionClient("/sciurus17/controller2/left_hand_controller/gripper_cmd", GripperCommandAction)
        self._left_gripper.wait_for_server()

        self._gripper_goal = GripperCommandGoal()
        self._gripper_goal.command.max_effort = 2.0

        # アームとグリッパー姿勢の初期化
        self.initialize_arms()

        self._current_arm = None

    def _my_object_callback(self, msg):
        self._my_object = msg
	#print self._my_object

	OFFSET_X = 0.204
	OFFSET_Y = 0.0325
	OFFSET_Z = 0.48

	self._my_object.x += OFFSET_X
	self._my_object.y += OFFSET_Y
	self._my_object.z += OFFSET_Z

	self.pick_up(True, self._my_object)

    def get_num_of_markers(self):
        return len(self._markers.markers)

    def _move_arm(self, current_arm, target_pose):
        if current_arm == self._RIGHT_ARM:
            # 手先を下に向ける
            q = quaternion_from_euler(3.14/2.0, 0.0, 0.0)
            target_pose.orientation.x = q[0]
            target_pose.orientation.y = q[1]
            target_pose.orientation.z = q[2]
            target_pose.orientation.w = q[3]
            self._right_arm.set_pose_target(target_pose)
            return self._right_arm.go()
        elif current_arm == self._LEFT_ARM:
            # 手先を下に向ける
            q = quaternion_from_euler(-3.14/2.0, 0.0, 0.0)
            target_pose.orientation.x = q[0]
            target_pose.orientation.y = q[1]
            target_pose.orientation.z = q[2]
            target_pose.orientation.w = q[3]
            self._left_arm.set_pose_target(target_pose)
            return self._left_arm.go()
        else:
            return False


    def _move_arm_to_init_pose(self, current_arm):
        if current_arm == self._RIGHT_ARM:
            self._right_arm.set_named_target("r_arm_waist_init_pose")
            return self._right_arm.go()
        elif current_arm == self._LEFT_ARM:
            self._left_arm.set_named_target("l_arm_waist_init_pose")
            return self._left_arm.go()
        else:
            return False


    def _open_gripper(self, current_arm):
        if current_arm == self._RIGHT_ARM:
            self._gripper_goal.command.position = self._GRIPPER_OPEN
            self._right_gripper.send_goal(self._gripper_goal)
            return self._right_gripper.wait_for_result(rospy.Duration(1.0))
        elif current_arm == self._LEFT_ARM:
            self._gripper_goal.command.position = -self._GRIPPER_OPEN
            self._left_gripper.send_goal(self._gripper_goal)
            return self._left_gripper.wait_for_result(rospy.Duration(1.0))
        else:
            return False


    def _close_gripper(self, current_arm):
        if current_arm == self._RIGHT_ARM:
            self._gripper_goal.command.position = self._GRIPPER_CLOSE
            self._right_gripper.send_goal(self._gripper_goal)
        elif current_arm == self._LEFT_ARM:
            self._gripper_goal.command.position = -self._GRIPPER_CLOSE
            self._left_gripper.send_goal(self._gripper_goal)
        else:
            return False


    def initialize_arms(self):
        self._move_arm_to_init_pose(self._RIGHT_ARM)
        self._move_arm_to_init_pose(self._LEFT_ARM)
        self._open_gripper(self._RIGHT_ARM)
        self._open_gripper(self._LEFT_ARM)


    def pick_up(self, check_result, position):
        # 一番高さが低いオブジェクトを持ち上げる
        rospy.loginfo("PICK UP")
        result = True
        self._current_arm = None # 制御対象を初期化

        # オブジェクトがなければ終了
        rospy.sleep(1.0)
        #if self.get_num_of_markers() == 0:
        #    rospy.logwarn("NO OBJECTS")
        #    return False

        object_pose = Pose()
	object_pose.position.x = position.x
	object_pose.position.y = position.y
	object_pose.position.z = position.z

        # オブジェクトの位置によって左右のどちらの手で取るかを判定する
        if object_pose.position.y < 0:
            self._current_arm = self._RIGHT_ARM
            rospy.loginfo("Set right arm")
        else:
            self._current_arm = self._LEFT_ARM
            rospy.loginfo("Set left arm")

        # 念の為手を広げる
        self._open_gripper(self._current_arm)

        # Z軸方向のオフセット meters
        APPROACH_OFFSET = 0.20
        PREPARE_OFFSET = 0.13
        LEAVE_OFFSET = 0.20

        #X_OFFSET = 0.015

        # 目標手先姿勢の生成
        target_pose = Pose()
        target_pose.position.x = object_pose.position.x# + X_OFFSET
        target_pose.position.y = object_pose.position.y
        target_pose.position.z = object_pose.position.z

	print(object_pose.position.x)
	print(object_pose.position.y)

        # 掴む準備をする
        target_pose.position.z = object_pose.position.z + APPROACH_OFFSET
        if self._move_arm(self._current_arm, target_pose) is False and check_result:
            rospy.logwarn("Approach failed")
            result = False

        else:
            rospy.sleep(1.0)
            # ハンドを下げる
            target_pose.position.z = object_pose.position.z + PREPARE_OFFSET
            if self._move_arm(self._current_arm, target_pose) is False and check_result:
                rospy.logwarn("Preparation grasping failed")
                result = False

            else:
                rospy.sleep(1.0)
                # つかむ
                if self._close_gripper(self._current_arm) is False and check_result:
                    rospy.logwarn("Grasping failed")
                    result = False

                rospy.sleep(1.0)
                # ハンドを上げる
                target_pose.position.z = object_pose.position.z + LEAVE_OFFSET
                self._move_arm(self._current_arm, target_pose)


        if result is False:
            rospy.sleep(1.0)
            # 失敗したときは安全のため手を広げる
            self._open_gripper(self._current_arm)

        rospy.sleep(10.0)
        # 初期位置に戻る
        self._move_arm_to_init_pose(self._current_arm)

        return result


    def place_on(self, check_result, position):
        result = True

        # 制御アームが設定されているかチェック
        if self._current_arm is None:
            rospy.logwarn("NO ARM SELECTED")
            return False

        # Z軸方向のオフセット meters
        APPROACH_OFFSET = 0.14
        PREPARE_OFFSET = 0.11
        LEAVE_OFFSET = 0.14

	#X_OFFSET = 0.03

        # 目標手先姿勢の生成
        target_pose = Pose()
        target_pose.position.x = position[0] #+ X_OFFSET
        target_pose.position.y = position[1]
        target_pose.position.z = position[2]

        # 置く準備をする
        target_pose.position.z = APPROACH_OFFSET
        if self._move_arm(self._current_arm, target_pose) is False and check_result:
            rospy.logwarn("Approach failed")
            result = False
        else:
            rospy.sleep(1.0)
            # ハンドを下げる
            target_pose.position.z = PREPARE_OFFSET
            if self._move_arm(self._current_arm, target_pose) is False and check_result:
                rospy.logwarn("Preparation release failed")
                result = False
            else:
                rospy.sleep(1.0)
                # はなす
                self._open_gripper(self._current_arm)
                # ハンドを上げる
                target_pose.position.z = LEAVE_OFFSET
                self._move_arm(self._current_arm, target_pose)

        if result is False:
            rospy.sleep(1.0)
            # 失敗したときは安全のため手を広げる
            self._open_gripper(self._current_arm)

        rospy.sleep(1.0)
        # 初期位置に戻る
        self._move_arm_to_init_pose(self._current_arm)

        return result

    def closest_color(self, rgb):

        COLORS = (
	    (0.78, 0.78, 0),
	    (0, 1.0, 0),
	    (0, 0, 1.0)
	    )

        r, g, b = rgb
        color_diffs = []
        for i in range(len(COLORS)):
	    cr, cg, cb = COLORS[i]
	    color_diff = abs(r - cr)**2 + abs(g - cg)**2 + abs(b - cb)**2
	    color_diffs.append((color_diff, i))
	min_color_diff = min(color_diffs)
	#if min_color_diff[0] > 1.0:
	#	return -1
        return min_color_diff[1]

    def get_object_positions(self):
	box_positions = copy.deepcopy(box_zero_positions)
	for marker in self._markers.markers:
		print("scale", marker.scale.x, marker.scale.y, marker.scale.z)
		if marker.scale.x > OBJECT_SIZE_MAX or marker.scale.y > OBJECT_SIZE_MAX:
			continue
		if marker.scale.x < OBJECT_SIZE_MIN or marker.scale.y < OBJECT_SIZE_MIN:
			continue
		marker_color = marker.color
		color_idx = self.closest_color((marker_color.r, marker_color.g, marker_color.b))
		if color_idx == -1:
			continue
		print("color_idx", color_idx, marker.scale.x, marker.scale.y, marker.scale.z)
		
		# marker.pose.position は箱の中心座標を表す
		position = marker.pose.position
		position.z += marker.scale.z * 0.5 # 箱の大きさ分高さを加算する
		box_positions[color_idx] = [position.x, position.y, position.z]

	return box_positions

def get_object_positions(r):

    angle_step = 20
    yaw_angles = [INITIAL_YAW_ANGLE,
		INITIAL_YAW_ANGLE - angle_step,
		INITIAL_YAW_ANGLE,
		INITIAL_YAW_ANGLE + angle_step,
		INITIAL_YAW_ANGLE,
		INITIAL_YAW_ANGLE - angle_step/2,
		INITIAL_YAW_ANGLE,
		INITIAL_YAW_ANGLE + angle_step/2,
		INITIAL_YAW_ANGLE
		]
    pitch_angles = [INITIAL_PITCH_ANGLE,
		INITIAL_PITCH_ANGLE + angle_step/2,
		INITIAL_PITCH_ANGLE + angle_step,
		INITIAL_PITCH_ANGLE + angle_step/2,
		INITIAL_PITCH_ANGLE + angle_step/4,
		INITIAL_PITCH_ANGLE + angle_step/2,
		INITIAL_PITCH_ANGLE + 3*angle_step/4,
		INITIAL_PITCH_ANGLE + angle_step/2,
		INITIAL_PITCH_ANGLE
		]

    box_positions = copy.deepcopy(box_zero_positions)
    box_appeared_count = [0,0,0]

    for i in range(len(yaw_angles)):
	yaw_angle = yaw_angles[i]
	pitch_angle = pitch_angles[i]
	neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle), 1.0)
	rospy.sleep(2)
        
	count = 0
	while not rospy.is_shutdown():
		if stacker.get_num_of_markers() > 0:
			print("---------------------")
			print("markers size", stacker.get_num_of_markers())
			box_positions_i = stacker.get_object_positions()
			print(box_positions_i)
			for j in range(len(box_positions_i)):
				if box_positions_i[j] != [0,0,0]:
					box_positions[j] = [x + y for x, y in zip(box_positions[j], box_positions_i[j])]
					box_appeared_count[j] += 1
			
			r.sleep()
			break
		else:
			count +=1
			if count >= 5:
				break
		r.sleep()


    for i in range(len(box_positions)):
	if box_appeared_count[i] != 0:
		box_positions[i] = [x / box_appeared_count[i] for x in box_positions[i]]
		last_box_positions[i] = copy.deepcopy(box_positions[i])

    print("++++++++++++++++++++++++++")
    for i in range(len(box_positions)):
    	print(box_positions[i])



def hook_shutdown():
    # 首の角度を戻す
    neck.set_angle(math.radians(0), math.radians(0), 3.0)
    # 両腕を初期位置に戻す
    stacker.initialize_arms()

def pixel_to_3d_point(x, y):
    #print depth_to_3d.cam_intrinsics
    depth = depth_to_3d.depth_img[y][x] / 1000.0
    X = (x - depth_to_3d.cam_intrinsics[2]) * depth / depth_to_3d.cam_intrinsics[0]
    Y = (y - depth_to_3d.cam_intrinsics[5]) * depth / depth_to_3d.cam_intrinsics[4]
    #print X, Y, depth
    return X, Y, depth
    #return x / 1000.0, y / 1000.0, depth

def main():
    global clicked_x
    global clicked_y

    r = rospy.Rate(60)

    rospy.on_shutdown(hook_shutdown)

    neck.set_angle(math.radians(INITIAL_YAW_ANGLE), math.radians(INITIAL_PITCH_ANGLE), 2.0)

    while not rospy.is_shutdown():
	if clicked_x == -1 and clicked_y == -1:
		rospy.sleep(1.0)
		continue
        print("hehe")
        X, Y, Z = pixel_to_3d_point(clicked_x,clicked_y)
	msg = Vector3(X, Y, Z)
	stacker._point_pub.publish(msg)

	clicked_x = -1
        clicked_y = -1

	r.sleep()


if __name__ == '__main__':
    rospy.init_node("tuantd_test_1")

    neck = NeckYawPitch()
    stacker = Stacker()
    depth_to_3d = DepthTo3D()

    main()

