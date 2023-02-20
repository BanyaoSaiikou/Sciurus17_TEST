#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <math.h> 

#include <pcl/point_types.h>
#include <pcl/common/common.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3Stamped.h>

#include <iostream>
using namespace std;

static ros::Publisher _pub_point;

const static std::string FRAME_ID = "base_link";
const static std::string CAMERA_FRAME_ID= "camera_color_optical_frame";

void my_point (const geometry_msgs::Vector3::ConstPtr& msg) {

    geometry_msgs::Vector3Stamped bt_v_in, bt_v_out;
    bt_v_in.vector.x = (*msg).x;
    bt_v_in.vector.y = (*msg).y;
    bt_v_in.vector.z = (*msg).z;
    bt_v_in.header.stamp = ros::Time();
    bt_v_in.header.frame_id = CAMERA_FRAME_ID;

    static tf::TransformListener listener;
    static tf::StampedTransform my_transform;

    // base_linkとカメラ間のTFを取得する
    while(true){
        try{
            //listener.lookupTransform(FRAME_ID, CAMERA_FRAME_ID, ros::Time(0), my_transform);
	    ros::Time now = ros::Time::now();
	    listener.waitForTransform(FRAME_ID, CAMERA_FRAME_ID, now, ros::Duration(0.5));
            listener.transformVector(FRAME_ID, bt_v_in, bt_v_out);
            break;
        }
        catch(tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
    }

    geometry_msgs::Vector3 msg_v;
    msg_v.x = bt_v_out.vector.x;
    msg_v.y = bt_v_out.vector.y;
    msg_v.z = bt_v_out.vector.z;
    _pub_point.publish(msg_v);
}


int main (int argc, char** argv)
{
    ros::init (argc, argv, "tuantd_test_2");
    ros::NodeHandle nh("~");

    _pub_point = nh.advertise<geometry_msgs::Vector3>("/sciurus17/example/my_object", 1);
    ros::Subscriber sub_my_point = nh.subscribe("/sciurus17/example/my_point", 1, my_point);

    ros::spin();
}

