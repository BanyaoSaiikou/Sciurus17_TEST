#!/usr/bin/env python
# coding: utf-8

import rospy
import actionlib
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal
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
def callback(data):
        
        
        a=list(data.actual.positions)[0]*180/math.pi
        np.set_angle(math.radians(-a), math.radians(-70.0))
        print(a)

    
def listener():
    
       # In ROS, nodes are uniquely named. If two nodes with the same
       # name are launched, the previous one is kicked off. The
       # anonymous=True flag means that rospy will choose a unique
       # name for our 'listener' node so that multiple listeners can
       # run simultaneously.

       goal=JointTrajectoryControllerState()
       rospy.Subscriber('/sciurus17/controller3/waist_yaw_controller/state', JointTrajectoryControllerState, callback)
       
       # spin() simply keeps python from exiting until this node is stopped
       rospy.spin()
class NeckPitch(object):
    def __init__(self):
        self.__client = actionlib.SimpleActionClient("/sciurus17/controller3/neck_controller/follow_joint_trajectory",
                                                     FollowJointTrajectoryAction)
        self.__client.wait_for_server(rospy.Duration(5.0))
        if not self.__client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Action Server Not Found")
            rospy.signal_shutdown("Action Server not found")
            sys.exit(1)
        self.yaw_angle = 0.0
        self.pitch_angle = 0.0

    def set_angle(self, yaw_angle, pitch_angle):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ["neck_yaw_joint", "neck_pitch_joint"]
        yawpoint = JointTrajectoryPoint()
        self.yaw_angle = yaw_angle
        self.pitch_angle = pitch_angle
        yawpoint.positions.append(self.yaw_angle)
        yawpoint.positions.append(self.pitch_angle)
        yawpoint.time_from_start = rospy.Duration(nsecs=1)

        goal.trajectory.points.append(yawpoint)
        #print("goal.trajectory.points2",goal.trajectory.points[0].positions)#!!
        self.__client.send_goal(goal)

        return self.__client.get_result()

if __name__ == '__main__':
    rospy.init_node("neck_test")

    np = NeckPitch()
    listener()
    try:
        pitch = float(sys.argv[1])
    except:
        pass
    
