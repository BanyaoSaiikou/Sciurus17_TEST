#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <std_msgs/Float32MultiArray.h>
#include <pcl/visualization/pcl_visualizer.h>
#include<pcl/visualization/cloud_viewer.h>



#include <pcl_conversions/pcl_conversions.h> 
#include <sensor_msgs/PointCloud2.h> 

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>


using namespace std_msgs;
using namespace message_filters;

float data1tx = 0.0;
float data1ty = 0.0;
float data1tz = 0.0;

float data1sx = 0.0;
float data1sy = 0.0;
float data1sz = 0.0;

float data2tx = 0.0;
float data2ty = 0.0;
float data2tz = 0.0;

float data2sx = 0.0;
float data2sy = 0.0;
float data2sz = 0.0;

float data3tx = 0.0;
float data3ty = 0.0;
float data3tz = 0.0;

float data3sx = 0.0;
float data3sy = 0.0;
float data3sz = 0.0;



void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    const static std::string FRAME_ID = "base_link";
    static tf::TransformListener listener;
    static tf::StampedTransform transform;


    sensor_msgs::PointCloud2 cloud_transformed;

    // point cloudの座標を変換
    // base_linkとカメラ間のTFを取得する
    while(true){
        try{
            listener.lookupTransform(FRAME_ID, cloud_msg->header.frame_id, ros::Time(0), transform);
            break;
        }
        catch(tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
    }
    pcl_ros::transformPointCloud(FRAME_ID, transform, *cloud_msg, cloud_transformed);

    // sensor_msgs/PointCloud2からpcl/PointCloudに変換
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>()); 
    pcl::fromROSMsg(cloud_transformed, *cloud);






  

    // Z方向の範囲でフィルタリング
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_passthrough (new pcl::PointCloud<pcl::PointXYZ>()); 
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.02, 1.0);
    pass.filter (*cloud_passthrough);
    pass.setInputCloud (cloud);


    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud_passthrough);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
	    //输出数据集
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    
	    //使用半径在查询点周围3厘米范围内的所有邻元素
    ne.setRadiusSearch(0.01);
	    //计算特征值
    ne.compute(*cloud_normals);



    //std::cout << data11 << std::endl;
    for(int nIndex=0;nIndex<cloud_passthrough->points.size();nIndex++){
        double x = cloud_passthrough->points[nIndex].x;
        double y = cloud_passthrough->points[nIndex].y;
        double z= cloud_passthrough->points[nIndex].z;

        double nx = cloud_normals->points[nIndex].normal[0];
        double ny = cloud_normals->points[nIndex].normal[1];
        double nz = cloud_normals->points[nIndex].normal[2];



    }



    
    
}
    
void chatterCallback(const std_msgs::Float32MultiArray& msg)
{

  data1tx = msg.data[0];
  data1ty = msg.data[1];
  data1tz = msg.data[2];

  data1sx = msg.data[3];
  data1sy = msg.data[4];
  data1sz = msg.data[5];

  data2tx = msg.data[6];
  data2ty = msg.data[7];
  data2tz = msg.data[8];

  data2sx = msg.data[9];
  data2sy = msg.data[10];
  data2sz = msg.data[11];

  data3tx = msg.data[12];
  data3ty = msg.data[13];
  data3tz = msg.data[14];

  data3sx = msg.data[15];
  data3sy = msg.data[16];
  data3sz = msg.data[17];
  ROS_INFO("%f",data1tx);
//(data1tx,data1ty,data1tz,data1sx,data1sy,data1sz,data2tx,data2ty,data2tz,data2sx,data2sy,data2sz,data3tx,data3ty,data3tz,data3sx,data3sy,data3sz);




}

//-------
void callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg, const std_msgs::Float32MultiArray& msg)
{
  // Solve all of perception here...
}
//-------------

int main (int argc, char** argv)
{
    ros::init (argc, argv, "object_detection");
    ros::NodeHandle nh("~");
    ros::NodeHandle n; 

    //message_filters::Subscriber<sensor_msgs::PointCloud2ConstPtr> sub(nh, "/sciurus17/camera/depth_registered/points", 1);
    //message_filters::Subscriber<std_msgs::Float32MultiArray> sub2(n, "chatter", 1);
    //TimeSynchronizer<sensor_msgs::PointCloud2ConstPtr, std_msgs::Float32MultiArray> sync(sub, sub2, 10);
    //sync.registerCallback(boost::bind(&callback, _1, _2));
    

    //ros::Subscriber sub = nh.subscribe("/sciurus17/camera/depth_registered/points", 1, cloud_cb);
    ros::Subscriber sub2 = n.subscribe("chatter", 1, chatterCallback);


    ros::spin ();
    return 0;
}
