#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <iostream>
int main(int argc, char** argv)
{
    ros::init(argc, argv, "showline");
    ros::NodeHandle n;
    ros::Publisher markerPub = n.advertise<visualization_msgs::Marker>("TEXT_VIEW_FACING", 10);
    ros::Publisher markerArrayPub = n.advertise<visualization_msgs::MarkerArray>("TEXT_VIEW_ARRAY", 10);

    ros::Rate r(1);
    int k = 0;

    while(ros::ok())
    {

        visualization_msgs::MarkerArray costARROWs;
        visualization_msgs::Marker costARROW1;
        bool once = true;

    

            
        costARROW1.header.frame_id = "base_link";
        costARROW1.header.stamp = ros::Time::now();
        costARROW1.id = 0;
        costARROW1.type = visualization_msgs::Marker::ARROW;
        costARROW1.scale.x = 0.5;
        costARROW1.scale.y = 0.05;
        costARROW1.scale.z = 0.05;
    	costARROW1.pose.orientation.x = 0.0;
    	costARROW1.pose.orientation.y = 0.0;
    	costARROW1.pose.orientation.z = 0.0;
    	costARROW1.pose.orientation.w = 1.0;
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 0.0f;
        costARROW1.color.g = 1.0f;
        costARROW1.color.b = 0.0f;
        costARROW1.pose.position.x = 1;
        costARROW1.pose.position.y = 1;
	costARROW1.pose.position.z = 0;
        costARROWs.markers.push_back(costARROW1);


        costARROW1.id = 2;
        costARROW1.type = visualization_msgs::Marker::ARROW;
        costARROW1.scale.x = 0.5;
        costARROW1.scale.y = 0.05;//尺寸，固定
        costARROW1.scale.z = 0.05;
    	costARROW1.pose.orientation.x = 0.0;//弧度
    	costARROW1.pose.orientation.y = -1.5707963267948966;
    	costARROW1.pose.orientation.z = 0.0;
    	costARROW1.pose.orientation.w = 1.0;
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 1.0f;
        costARROW1.color.g = 0.0f;
        costARROW1.color.b = 0.0f;
        costARROW1.pose.position.x = 2;//位置
        costARROW1.pose.position.y = 2;
	costARROW1.pose.position.z = 0;
        costARROWs.markers.push_back(costARROW1);
        


        

        markerArrayPub.publish(costARROWs);
        r.sleep();
    }
    return 0;
}

