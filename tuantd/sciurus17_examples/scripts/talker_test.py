#! /usr/bin/env python
# coding: utf-8
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
def callback(data):
    rospy.loginfo(data.data)

def listener():

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("chatter", Float32MultiArray, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
