#! /usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import moveit_commander
import geometry_msgs.msg
import rosnode
import actionlib
from tf.transformations import quaternion_from_euler
from control_msgs.msg import (GripperCommandAction, GripperCommandGoal)



from multiprocessing.connection import answer_challenge
import os
import re
from sre_constants import AT_BEGINNING_LINE
from tokenize import Double
from unittest import result
import math
import rospy
from std_msgs.msg import Float32MultiArray

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
from pickle import GLOBAL
import rospy
import actionlib
from visualization_msgs.msg import Marker, MarkerArray
from control_msgs.msg import (
    
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    FollowJointTrajectoryActionGoal,
    JointTrajectoryControllerState
    
)
from trajectory_msgs.msg import JointTrajectoryPoint
import sys
import math
from std_msgs.msg import String
import re
import math



import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np

rospy.init_node("sciurus17_pick_and_place_controller")
robot = moveit_commander.RobotCommander()
arm = moveit_commander.MoveGroupCommander("l_arm_group")
arm.set_max_velocity_scaling_factor(0.1)
gripper = actionlib.SimpleActionClient("/sciurus17/controller2/left_hand_controller/gripper_cmd", GripperCommandAction)
gripper.wait_for_server()
gripper_goal = GripperCommandGoal()
gripper_goal.command.max_effort = 2.0

linear_angular = np.zeros((1, 3))	# 4 * 2数组
# a=0.0
# b=0.0
# c=0.0
a=1000
b=1000
c=1000
i1 = 0
def callback(data):
	global a
	global b
	global c
	global i1  
	#linear_angular[0][0] = data.data	# data.data是取出一维数组
	#print("linear_angular[0]",linear_angular[0][0])
        #print(data.data[0], data.data[1], data.data[2])
        print("a",a)
        print("b",b)
        print("c",c)

	if abs(a-data.data[0])>0.01 :#(a,b,c)位置是否变化
		#arm.stop()
		i1=0#如果位置变化 需要规划路径
		print("1 time stop a")
	elif abs(b-data.data[1])>0.01 :
		#arm.stop()
		i1=0
		print("1 time stop b")
	elif abs(c-data.data[2])>0.01 :
		#arm.stop()
		i1=0
		print("1 time stop c")

	if i1==0:#i1=0时，规划路径/i1！=0时 无需规划
		pickup(data.data[0], data.data[1], 0.3)
		print(data.data[0], data.data[1], 0.3)
		print("1 time")
	a=data.data[0]
	b=data.data[1]
	c=data.data[2]
	i1=1#路径规划完成
	print("-----------")







def Array_sub():

  
	rospy.Subscriber("chatter", Float32MultiArray, callback)

	rospy.spin()


def pickup(x,y,z):
    # 掴みに行く
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = x
    target_pose.position.y = y
    target_pose.position.z = z
    q = quaternion_from_euler(-3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    plan=arm.plan()
    arm.execute(plan,wait=False)



def main():


    rospy.sleep(1.0)


    # アーム初期ポーズを表示
    arm_initial_pose = arm.get_current_pose().pose


    # 何かを掴んでいた時のためにハンドを開く
    gripper_goal.command.position = -0.9
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

    # SRDFに定義されている"home"の姿勢にする
    arm.set_named_target("l_arm_init_pose")
    arm.go()
    gripper_goal.command.position = 0.0
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))

    # 掴む準備をする
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.25
    target_pose.position.y = 0.0
    target_pose.position.z = 0.3
    q = quaternion_from_euler(-3.14/2.0, 0.0, 0.0)  # 上方から掴みに行く場合
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]
    arm.set_pose_target(target_pose)  # 目標ポーズ設定
    arm.go()  # 実行

    # ハンドを開く
    gripper_goal.command.position = -0.7
    gripper.send_goal(gripper_goal)
    gripper.wait_for_result(rospy.Duration(1.0))





if __name__ == '__main__':


  main()
  # pickup(0.25,0.0,0.13)
      # ハンドを閉じる
  Array_sub()
