#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <iostream>

#include <std_msgs/Float32MultiArray.h>

float data10 = 0.0;
float data11 = 0.0;
float data12 = 0.0;
float data13 = 0.0;
float data14 = 0.0;
float data15 = 0.0;
float data16 = 0.0;
float data17 = 0.0;

void chatterCallback(const std_msgs::Float32MultiArray& msg)
{
  //int num = msg.data.size();
  //ROS_INFO("I susclibed [%i]", num);
  //for (int i = 0; i < num; i++)
  //{
  //ROS_INFO("%f",msg.data[0]);
  //}

  data10 = msg.data[0];
  data11 = msg.data[1];
  data12 = msg.data[2];
  data13 = msg.data[3];

  data14 = msg.data[4];
  data15 = msg.data[5];
  data16 = msg.data[6];
  data17 = msg.data[7];



  ros::NodeHandle n;
  ros::Publisher markerPub = n.advertise<visualization_msgs::Marker>("TEXT_VIEW_FACING", 10);//送一个
  ros::Publisher markerArrayPub = n.advertise<visualization_msgs::MarkerArray>("TEXT_VIEW_ARRAY", 10);//送一组
    ros::Rate r(20.0);
    int k = 0;
    int x1=0;
    //ROS_INFO("%f",data10);//输出接受的数值
    //ROS_INFO("%f",data11);//输出接受的数值
    //ROS_INFO("%f",data12);//输出接受的数值
    ROS_INFO("%f",data13);//输出接受的数值
    //ROS_INFO("%f",data14);//输出接受的数值
    //ROS_INFO("%f",data15);//输出接受的数值
    //ROS_INFO("%f",data16);//输出接受的数值
    ROS_INFO("%f",data17);//输出接受的数值

    while(x1<6)
    {
  	
//---------------------------------------------------发送箭头位置---

        visualization_msgs::MarkerArray costARROWs;
        visualization_msgs::Marker costARROW1;
        bool once = true;

    

            
            
        costARROW1.header.frame_id = "base_link";
        costARROW1.header.stamp = ros::Time::now();
        costARROW1.id = 0;
        costARROW1.type = visualization_msgs::Marker::ARROW;
        costARROW1.scale.x = 0.05;
        costARROW1.scale.y = 0.005;
        costARROW1.scale.z = 0.005;
//--------------------------------------------------------------
    	costARROW1.pose.position.x =data10;
        costARROW1.pose.position.y = data11;
	costARROW1.pose.position.z = data12;

	costARROW1.pose.orientation.x =-0.0 ;
    	costARROW1.pose.orientation.y =-1+data13;
    	costARROW1.pose.orientation.z = 0.0;
    	costARROW1.pose.orientation.w = 1.0;
//--------------------------------------------------------------
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 1.0f;
        costARROW1.color.g = 0.0f;
        costARROW1.color.b = 0.0f;


        costARROWs.markers.push_back(costARROW1);


        costARROW1.id = 2;
        costARROW1.type = visualization_msgs::Marker::ARROW;
        costARROW1.scale.x = 0.05;
        costARROW1.scale.y = 0.005;//尺寸，固定
        costARROW1.scale.z = 0.005;
//--------------------------------------------------------------

        costARROW1.pose.position.x =data14;//位置
        costARROW1.pose.position.y =data15;
	costARROW1.pose.position.z =data16;

    	costARROW1.pose.orientation.x = 0.0;//弧度
    	costARROW1.pose.orientation.y = 0.0;
    	costARROW1.pose.orientation.z =  data17 ;
    	costARROW1.pose.orientation.w = 1.0;
//--------------------------------------------------------------
        costARROW1.color.a = 1.0;
        costARROW1.color.r = 0.0f;
        costARROW1.color.g = 0.0f;
        costARROW1.color.b = 1.0f;






        costARROWs.markers.push_back(costARROW1);
        

        markerArrayPub.publish(costARROWs);


	x1=x1+1;
        r.sleep();
    }
        ros::NodeHandle nh;
    	ros::Subscriber sub = nh.subscribe("chattertorviz", 10, chatterCallback);//收msg

    	












}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "showline");
    ros::NodeHandle n;


//---------------------------
    ros::Subscriber sub = n.subscribe("chattertorviz", 10, chatterCallback);//收msg


    ros::spin();
//---------------------------


    return 0;
}

