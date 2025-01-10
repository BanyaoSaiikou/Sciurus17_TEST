
import rospy
from std_msgs.msg import String
import time  

def talker():
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('chatter', String, queue_size=10)

    rate = rospy.Rate(1)  

    while not rospy.is_shutdown():
        data = "Hello, ROS!"
        pub.publish(data)
        print(data)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
