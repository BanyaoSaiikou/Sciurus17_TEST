#! /usr/bin/env python
# coding: utf-8
import rospy
from std_msgs.msg import String
import rospy
import actionlib
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    FollowJointTrajectoryActionGoal
)
from trajectory_msgs.msg import JointTrajectoryPoint
import sys
import math
from std_msgs.msg import Float32MultiArray

import rospy
import math
import sys
import pickle
import copy
import socket
import cv2
import os
import glob
import numpy as np
import time
from scipy import spatial
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Vector3, Point
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
INITIAL_PITCH_ANGLE = -70

OBJECT_SIZE_MAX = 0.07
OBJECT_SIZE_MIN = 0.03

box_zero_positions = [[0,0,0],[0,0,0],[0,0,0]]
last_box_positions = copy.deepcopy(box_zero_positions)

class WaistYaw(object):
    # 初期化処理
    def __init__(self):
        self.__client = actionlib.SimpleActionClient("/sciurus17/controller3/waist_yaw_controller/follow_joint_trajectory",
                                                     FollowJointTrajectoryAction)
        self.__client.wait_for_server(rospy.Duration(5.0))

        if not self.__client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Action Server Not Found")
            rospy.signal_shutdown("Action Server not found")
            sys.exit(1)
        self.present_angle = 0.0

    # 現在角度を設定
    def set_present_angle(self, angle):
        self.present_angle = angle

    # 目標角度の設定と実行
    def set_angle(self, yaw_angle, goal_sec):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["waist_yaw_joint"]

        # 現在の角度から開始（遷移時間は現在時刻）
        yawpoint = JointTrajectoryPoint()
        yawpoint.positions.append(self.present_angle)
        yawpoint.time_from_start = rospy.Duration(nsecs=1)
        goal.trajectory.points.append(yawpoint)

        # 途中に角度と時刻をappendすると細かい速度制御が可能
        # 参考=> http://wiki.ros.org/joint_trajectory_controller/UnderstandingTrajectoryReplacement

        # ゴール角度を設定（指定されたゴール時間で到達）
        yawpoint = JointTrajectoryPoint()
        yawpoint.positions.append(yaw_angle)
        yawpoint.time_from_start = rospy.Duration(goal_sec)
        goal.trajectory.points.append(yawpoint)
        self.present_angle = yaw_angle

        # 軌道計画を実行
        self.__client.send_goal(goal)
        self.__client.wait_for_result(rospy.Duration(goal_sec))
        return self.__client.get_result()

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
	    self.rgb_img = input_image
            #cv2.circle(input_image, (320,250), 2, (255, 255, 0), -1)
	    #cv2.imshow('image', input_image)
            #cv2.waitKey(1)
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
    _GRIPPER_OPEN = 1.0
    _GRIPPER_CLOSE = 0.34 #0.42

    def __init__(self):
	self._markers = MarkerArray()

	self._my_object_sub = rospy.Subscriber("/sciurus17/example/my_object",
                MarkerArray, self._my_object_callback, queue_size=1)

	self._point_pub = rospy.Publisher("/sciurus17/example/my_point", MarkerArray, queue_size=1)

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
        self._markers = msg
	#print self._my_object

    def check_markers(self):
	if len(self._markers.markers) == 0:
	    print("no marker")
	    return
	
	for marker in self._markers.markers:
	    print(marker)

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
        if self.get_num_of_markers() == 0:
            rospy.logwarn("NO OBJECTS")
            return False

        object_pose = Pose()
	object_pose.position.x = position[0]
	object_pose.position.y = position[1]
	object_pose.position.z = position[2]

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
        APPROACH_OFFSET = 0.15
        PREPARE_OFFSET = 0.07
        LEAVE_OFFSET = 0.15

        #X_OFFSET = 0.015

        # 目標手先姿勢の生成
        target_pose = Pose()
        target_pose.position.x = object_pose.position.x #+ X_OFFSET
        target_pose.position.y = object_pose.position.y
        target_pose.position.z = object_pose.position.z

	print(object_pose.position.x)
	print(object_pose.position.y)
	print(object_pose.position.z)

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

        rospy.sleep(1.0)
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
        APPROACH_OFFSET = 0.15
        PREPARE_OFFSET = 0.058
        LEAVE_OFFSET = 0.15

	#X_OFFSET = 0.03

        # 目標手先姿勢の生成
        target_pose = Pose()
        target_pose.position.x = position[0] #+ X_OFFSET
        target_pose.position.y = position[1]
        target_pose.position.z = position[2]

        # 置く準備をする<frameName>
        target_pose.position.z = position[2] + APPROACH_OFFSET
        if self._move_arm(self._current_arm, target_pose) is False and check_result:
            rospy.logwarn("Approach failed")
            result = False
        else:
            rospy.sleep(1.0)
            # ハンドを下げる
            target_pose.position.z = position[2] + PREPARE_OFFSET
            if self._move_arm(self._current_arm, target_pose) is False and check_result:
                rospy.logwarn("Preparation release failed")
                result = False
            else:
                rospy.sleep(1.0)
                # はなす
                self._open_gripper(self._current_arm)
                # ハンドを上げる
                target_pose.position.z = position[2] + LEAVE_OFFSET
                self._move_arm(self._current_arm, target_pose)

        if result is False:
            rospy.sleep(1.0)
            # 失敗したときは安全のため手を広げる
            self._open_gripper(self._current_arm)

        rospy.sleep(1.0)
        # 初期位置に戻る
        self._move_arm_to_init_pose(self._current_arm)

        return result


def hook_shutdown():
    # 首の角度を戻す
    neck.set_angle(math.radians(0), math.radians(0), 3.0)
    #wy.set_angle(math.radians(0), 2.0)
    # 両腕を初期位置に戻す
    stacker.initialize_arms()
    conn.close()

def pixel_to_3d_point(x, y):
    #print depth_to_3d.cam_intrinsics
    depth = depth_to_3d.depth_img[y][x] / 1000.0
    X = (x - depth_to_3d.cam_intrinsics[2]) * depth / depth_to_3d.cam_intrinsics[0]
    Y = (y - depth_to_3d.cam_intrinsics[5]) * depth / depth_to_3d.cam_intrinsics[4]
    #print X, Y, depth
    return X, Y, depth

def move_neck_and_waist(yaw_angle, pitch_angle):
    #wy.set_angle(math.radians(yaw_angle), 1.0)
    #neck.set_angle(math.radians(INITIAL_YAW_ANGLE), math.radians(pitch_angle), 1.0)
    neck.set_angle(math.radians(yaw_angle), math.radians(pitch_angle), 1.0)
    








def get_object_positions(r):

    global frame_count

    MAX_YAW_ANGLE   = 120
    MIN_YAW_ANGLE   = -120
    MAX_PITCH_ANGLE = 80
    MIN_PITCH_ANGLE = INITIAL_PITCH_ANGLE

    OPERATION_GAIN_X = 30.0
    OPERATION_GAIN_Y = 25.0

    THRESH_X = 0.05
    THRESH_Y = 0.05

    angle_step = 20
    yaw_angles = [INITIAL_YAW_ANGLE,
		INITIAL_YAW_ANGLE - angle_step,
		INITIAL_YAW_ANGLE,
		INITIAL_YAW_ANGLE + angle_step
		]
    pitch_angles = [INITIAL_PITCH_ANGLE,
		INITIAL_PITCH_ANGLE + angle_step/2,
		INITIAL_PITCH_ANGLE + angle_step,
		INITIAL_PITCH_ANGLE + angle_step/2
		]

    yaw_angles = [INITIAL_YAW_ANGLE]
    pitch_angles = [INITIAL_PITCH_ANGLE]

    box_positions = copy.deepcopy(box_zero_positions)
    box_appeared_count = [0,0,0]

    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]

    current_box_idx = -1
    is_next_box_detected = False
    data = None
    while True:
	    is_detected = False
	    current_box_idx += 1
	    if current_box_idx >= len(box_positions):
		break
	    box_positions[current_box_idx] = []
	    for i in range(len(yaw_angles)):
		if is_next_box_detected == False:
			

			start0 = time.time()
			cv2.imwrite(os.path.join(DEPTH_DIR, str(frame_count) + '.png'),depth_to_3d.depth_img)
			frame_count += 1
			

			result, imgencode = cv2.imencode('.jpg', depth_to_3d.rgb_img, encode_param)
			data = np.array(imgencode)
			string_data = data.tostring()
		

			conn.send(str.encode(str(len(string_data)).ljust(16)))
			

			conn.send(string_data)

			end0 = time.time()
			print("++++++++++++timephotosaveandsend: ",(end0-start0))

			#--
			#---------------------------------------------------------------

			#--
			#data = pickle.loads(conn.recv(BUFFER_SIZE))#shou 18 ci
			#print("datashan",data)
			#('datashan', [1, [[[336, 153], [341, 191]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]])
 			start4 = time.time()
    			datalist1 = conn.recv(BUFFER_SIZE)
			#print("datalist1:",datalist1)
			datastr = datalist1.split(" ")
			#print("datastr:",datastr)

			newdatastr = [int(x) for x in datastr]
			#print("newdatastr:",newdatastr)			
                       
			data = [newdatastr[0], [[[newdatastr[1], newdatastr[2]], [newdatastr[3], newdatastr[4]]], [[newdatastr[5], newdatastr[6]], [newdatastr[7], newdatastr[8]]], [[newdatastr[9], newdatastr[10]], [newdatastr[11], newdatastr[12]]]]]
			print("data:",data)
			#-----
			end4 = time.time()
			#print("++++++++++++time4pickle.loads: ",(end4-start4))
			#---------------------------

			#--
			#---------------------------------------------------------------
		r.sleep()
		#print("data1",data)
		if data[0] == 1:
			

			#---------------------------------------------------------------
			start22 = time.time()
			#--
			markers = MarkerArray()
			X, Y, Z = 0, 0, 0
			if data[1][current_box_idx] != []:
				for marker_idx in range(len(data[1][current_box_idx])):
					X, Y, Z = 0, 0, 0
					if data[1][current_box_idx][marker_idx] != []:
						X, Y, Z = pixel_to_3d_point(data[1][current_box_idx][marker_idx][0],data[1][current_box_idx][marker_idx][1])
		
					marker = Marker()
					marker.pose.position.x = X
					marker.pose.position.y = Y
					marker.pose.position.z = Z
					markers.markers.append(marker)
					r.sleep()
			else:
				marker = Marker()
				marker.pose.position.x = X
				marker.pose.position.y = Y
				marker.pose.position.z = Z
				markers.markers.append(marker)
				r.sleep()
		
			stacker._markers = None
			stacker._point_pub.publish(markers)
		
			count = 0
			while not rospy.is_shutdown():
				
				if stacker._markers == None:
					count += 1
					if count > 100000000000:
						stacker._point_pub.publish(markers)
						stacker._markers = None
						count = 0
					continue
				
				box_positions[current_box_idx] = []
				for j in range(len(stacker._markers.markers)):
					marker_pos = [0,0,0]			
					X = stacker._markers.markers[j].pose.position.x
					Y = stacker._markers.markers[j].pose.position.y
					Z = stacker._markers.markers[j].pose.position.z
					if abs(X) > 0.5 or abs(Y) > 0.5 or (Z + OFFSET_Z) > 0.25 or (Z + OFFSET_Z) < -0.02:
						#print("invalid", stacker._markers.markers[j].pose.position)
						marker_pos = [0,0,0]

					if X != 0 or Y != 0 or Z != 0:
						X += OFFSET_X
						Y += OFFSET_Y
						Z += OFFSET_Z
						marker_pos = [X, Y, Z]
						is_detected = True

					box_positions[current_box_idx].append(marker_pos)
					r.sleep()
				
				if is_detected:
					#current_box_idx += 1

					if current_box_idx < len(box_positions) and data[1][current_box_idx] != []:
						is_next_box_detected = True
					else:
						is_next_box_detected = False


				break



			#-----
			end22 = time.time()
			#print("++++++++++++time222dto3d: ",(end22-start22))
			#---------------------------



			if is_detected:
				break
			r.sleep()
		r.sleep()
            
	    if current_box_idx >= len(box_positions):
		break

	#---------------------------
    #for box_idx in range(len(box_positions)):
	#last_box_positions[box_idx] = copy.deepcopy(box_positions[box_idx])
	#temp = last_box_positions[box_idx][2]
	#if abs(temp-0.05) < abs(temp-0.1):
	#	if abs(temp-0.05) < abs(temp-0.15):
	#		last_box_positions[box_idx][2] = 0.05
        #else:
	#	if abs(temp-0.1) < abs(temp-0.15):
	#		last_box_positions[box_idx][2] = 0.1
	#	else:
	#		last_box_positions[box_idx][2] = 0.15

    print("++++++++++++++++++++++++++")

    startpub = time.time()
    pub = rospy.Publisher('chatter', Float32MultiArray, queue_size=1)

    #rospy.init_node('talker', anonymous=True)

    #b=np.array(box_positions[0],box_positions[1])

    #a=([[box_positions[0][0][0],box_positions[0][0][1],box_positions[0][0][2]],[box_positions[0][1][0],box_positions[0][1][1],box_positions[0][1][2]],[box_positions[1][0][0],box_positions[1][0][1],box_positions[1][0][2]],[box_positions[1][1][0],box_positions[1][1][1],box_positions[1][1][2]],[box_positions[2][0][0],box_positions[2][0][1],box_positions[2][0][2]],[box_positions[2][1][0],box_positions[2][1][1],box_positions[2][1][2]]])
    a=([box_positions[0][0][0],box_positions[0][0][1],box_positions[0][0][2],box_positions[0][1][0],box_positions[0][1][1],box_positions[0][1][2],box_positions[1][0][0],box_positions[1][0][1],box_positions[1][0][2],box_positions[1][1][0],box_positions[1][1][1],box_positions[1][1][2],box_positions[2][0][0],box_positions[2][0][1],box_positions[2][0][2],box_positions[2][1][0],box_positions[2][1][1],box_positions[2][1][2]])

    left_top = Float32MultiArray(data=a)
    #rospy.loginfo(left_top)
    pub.publish(left_top)
    rate1=rospy.Rate(100)
    rate1.sleep()
    endpub = time.time()
    #print("++++++++++++timepub: ",(endpub-startpub))

    print(a)
	
   # for i in range(len(box_positions)):
    	#print(box_positions[i])
    	#print(box_positions[0][0],box_positions[0][1])



def main():
    r = rospy.Rate(100)

    rospy.on_shutdown(hook_shutdown)

    neck.set_angle(math.radians(INITIAL_YAW_ANGLE), math.radians(INITIAL_PITCH_ANGLE), 2.0)

    CHECK_RESULT = True

    human_state_idx = -1

    


    while not rospy.is_shutdown():
	
	start5 = time.time()
		# get and send final box positions performed by human
	get_object_positions(r)
	end5 = time.time()
	#print("++++++++++++time5: ",(end5-start5))
	r.sleep()




if __name__ == '__main__':
    rospy.init_node("tuantd_box_stacking",anonymous=True)

    frame_count = 0

    DEPTH_DIR = "/home/sciurus/Desktop/tuantd/depth"
    if not os.path.exists(DEPTH_DIR):
	os.makedirs(DEPTH_DIR)
    	print("dir made")

    files = glob.glob(os.path.join(DEPTH_DIR, "*"))
    for f in files:
    	os.remove(f)

    OFFSET_X = 0.204
    OFFSET_Y = 0.0325
    OFFSET_Z = 0.48

    #wy = WaistYaw()
    #wy.set_present_angle(math.radians(0.0))
    
    neck = NeckYawPitch()
    stacker = Stacker()
    depth_to_3d = DepthTo3D()

    TCP_IP = "10.40.1.84"
    TCP_PORT = 5432
    BUFFER_SIZE = 1024

    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.bind((TCP_IP, TCP_PORT))
    soc.listen(1)

    print("waiting for connection")
    conn, addr = soc.accept()
    print("connection address:", addr)

    main()

