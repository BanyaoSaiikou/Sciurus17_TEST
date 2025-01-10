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
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <iostream>
using namespace std;

static ros::Publisher _pub_point;

const static std::string FRAME_ID = "base_link";
const static std::string CAMERA_FRAME_ID= "camera_color_optical_frame";

void my_point (const visualization_msgs::MarkerArray::ConstPtr& msg) {

    static tf::TransformListener listener;
    static tf::StampedTransform my_transform;

    visualization_msgs::MarkerArray markers;
    for (int i=0; i < msg->markers.size(); i++) {
	visualization_msgs::Marker marker;
	marker.pose.position.x = 0;
	marker.pose.position.y = 0;
	marker.pose.position.z = 0;
	markers.markers.push_back(marker);
    }

    // base_linkとカメラ間のTFを取得する
    while(true){
        try{
	    ros::Time now = ros::Time::now();
            listener.waitForTransform(FRAME_ID, CAMERA_FRAME_ID, now, ros::Duration(1.0));
            //listener.lookupTransform(FRAME_ID, CAMERA_FRAME_ID, ros::Time(0), my_transform);
	    for (int i=0; i < msg->markers.size(); i++) {
		    float x = msg->markers[i].pose.position.x;
		    float y = msg->markers[i].pose.position.y;
		    float z = msg->markers[i].pose.position.z;
		    if (x == 0 && y == 0 && z == 0)
			continue;
	            geometry_msgs::Vector3Stamped bt_v_in, bt_v_out;
		    bt_v_in.vector.x = x;
		    bt_v_in.vector.y = y;
		    bt_v_in.vector.z = z;
		    bt_v_in.header.stamp = ros::Time();
		    bt_v_in.header.frame_id = CAMERA_FRAME_ID;
		    
	            listener.transformVector(FRAME_ID, bt_v_in, bt_v_out);
		    
		    markers.markers[i].pose.position.x = bt_v_out.vector.x;
		    markers.markers[i].pose.position.y = bt_v_out.vector.y;
		    markers.markers[i].pose.position.z = bt_v_out.vector.z;
		    
	    }
            break;
        }
        catch(tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
    }

    _pub_point.publish(markers);
}


int main (int argc, char** argv)
{
    ros::init (argc, argv, "tuantd_object_position");
    ros::NodeHandle nh("~");

    _pub_point = nh.advertise<visualization_msgs::MarkerArray>("/sciurus17/example/my_object", 1);
    ros::Subscriber sub_my_point = nh.subscribe("/sciurus17/example/my_point", 1, my_point);

    ros::spin();
}

