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
        costARROW1.scale.x = 0.3;
        costARROW1.scale.y = 0.03;
        costARROW1.scale.z = 0.03;
//--------------------------------------------------------------
    	costARROW1.pose.position.x =0.452864;
        costARROW1.pose.position.y = -0.0122468;
	costARROW1.pose.position.z = 0.100353;

	costARROW1.pose.orientation.x =-0.0 ;
    	costARROW1.pose.orientation.y =-0.994794;
    	costARROW1.pose.orientation.z = 0.0;
    	costARROW1.pose.orientation.w = 1.0;
//--------------------------------------------------------------
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 0.0f;
        costARROW1.color.g = 0.0f;
        costARROW1.color.b = 1.0f;


        costARROWs.markers.push_back(costARROW1);


        costARROW1.id = 2;
        costARROW1.type = visualization_msgs::Marker::ARROW;
        costARROW1.scale.x = 0.3;
        costARROW1.scale.y = 0.03;//尺寸，固定
        costARROW1.scale.z = 0.03;
//--------------------------------------------------------------

        costARROW1.pose.position.x =0.44329;//位置
        costARROW1.pose.position.y =-0.014389;
	costARROW1.pose.position.z =0.0320343;

    	costARROW1.pose.orientation.x = 0.0;//弧度
    	costARROW1.pose.orientation.y = 0.0;
    	costARROW1.pose.orientation.z =  (-0.997643+1) ;
    	costARROW1.pose.orientation.w = 1.0;
//--------------------------------------------------------------
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 0.0f;
        costARROW1.color.g = 0.0f;
        costARROW1.color.b = 1.0f;






        costARROWs.markers.push_back(costARROW1);


        costARROW1.id = 1;
        costARROW1.type = visualization_msgs::Marker::ARROW;
        costARROW1.scale.x = 0.3;
        costARROW1.scale.y = 0.03;//尺寸，固定
        costARROW1.scale.z = 0.03;
//--------------------------------------------------------------

        costARROW1.pose.position.x =0.473786;//位置
        costARROW1.pose.position.y =-0.016934;
	costARROW1.pose.position.z =0.101529;

    	costARROW1.pose.orientation.x = 0.0;//弧度
    	costARROW1.pose.orientation.y = -0.994922;
    	costARROW1.pose.orientation.z =  0.0 ;
    	costARROW1.pose.orientation.w = 1.0;
//--------------------------------------------------------------
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 1.0f;
        costARROW1.color.g = 1.0f;
        costARROW1.color.b = 0.0f;





        costARROWs.markers.push_back(costARROW1);


        costARROW1.id = 3;
        costARROW1.type = visualization_msgs::Marker::ARROW;
        costARROW1.scale.x = 0.3;
        costARROW1.scale.y = 0.03;//尺寸，固定
        costARROW1.scale.z = 0.03;
//--------------------------------------------------------------

        costARROW1.pose.position.x =0.443012;//位置
        costARROW1.pose.position.y =-0.0149054;
	costARROW1.pose.position.z =0.0797804;

    	costARROW1.pose.orientation.x = 0.0;//弧度
    	costARROW1.pose.orientation.y = 0.0;
    	costARROW1.pose.orientation.z =  (-0.979359+1) ;
    	costARROW1.pose.orientation.w = 1.0;
//--------------------------------------------------------------
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 1.0f;
        costARROW1.color.g = 1.0f;
        costARROW1.color.b = 0.0f;


        costARROWs.markers.push_back(costARROW1);
        



        costARROW1.id = 4;
        costARROW1.type = visualization_msgs::Marker::ARROW;
        costARROW1.scale.x = 0.3;
        costARROW1.scale.y = 0.03;//尺寸，固定
        costARROW1.scale.z = 0.03;
//--------------------------------------------------------------

        costARROW1.pose.position.x =0.48042;//位置
        costARROW1.pose.position.y =0.113865;
	costARROW1.pose.position.z =0.0529621;

    	costARROW1.pose.orientation.x = 0.0;//弧度
    	costARROW1.pose.orientation.y = -0.993828;
    	costARROW1.pose.orientation.z =  0.0 ;
    	costARROW1.pose.orientation.w = 1.0;
//--------------------------------------------------------------
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 0.0f;
        costARROW1.color.g = 1.0f;
        costARROW1.color.b = 0.0f;


        costARROWs.markers.push_back(costARROW1);



        costARROW1.id = 5;
        costARROW1.type = visualization_msgs::Marker::ARROW;
        costARROW1.scale.x = 0.3;
        costARROW1.scale.y = 0.03;//尺寸，固定
        costARROW1.scale.z = 0.03;
//--------------------------------------------------------------

        costARROW1.pose.position.x =0.464715;//位置
        costARROW1.pose.position.y =0.0906556;
	costARROW1.pose.position.z =0.0249471;

    	costARROW1.pose.orientation.x = 0.0;//弧度
    	costARROW1.pose.orientation.y = 0.0;
    	costARROW1.pose.orientation.z = -0.566311+1 ;
    	costARROW1.pose.orientation.w = 1.0;
//--------------------------------------------------------------
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 0.0f;
        costARROW1.color.g = 1.0f;
        costARROW1.color.b = 0.0f;


        costARROWs.markers.push_back(costARROW1);


        

        markerArrayPub.publish(costARROWs);
        r.sleep();
    }
    return 0;
}

