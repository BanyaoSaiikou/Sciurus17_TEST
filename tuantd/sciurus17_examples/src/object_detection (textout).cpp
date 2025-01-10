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

static ros::Publisher _pub_point_cloud;
static ros::Publisher _pub_marker_array;


void convert_to_marker(visualization_msgs::Marker *marker, const int marker_id,
        const std::string &frame_id,
        const pcl::PointXYZ &min_pt, const pcl::PointXYZ &max_pt)
{
    // pcl::Pointの最大最小値をBox型のマーカに変換する

    marker->header.frame_id = frame_id;
    marker->header.stamp = ros::Time();
    marker->ns = "/sciurus17/example";
    marker->id = marker_id;
    marker->type = visualization_msgs::Marker::CUBE;
    marker->action = visualization_msgs::Marker::ADD;
    marker->lifetime = ros::Duration(0.5);

    marker->pose.position.x = (max_pt.x + min_pt.x) * 0.5;
    marker->pose.position.y = (max_pt.y + min_pt.y) * 0.5;
    marker->pose.position.z = (max_pt.z + min_pt.z) * 0.5;
    marker->pose.orientation.x = 0.0;
    marker->pose.orientation.y = 0.0;
    marker->pose.orientation.z = 0.0;
    marker->pose.orientation.w = 1.0;
    marker->scale.x = (max_pt.x - min_pt.x);
    marker->scale.y = (max_pt.y - min_pt.y);
    marker->scale.z = (max_pt.z - min_pt.z);
    marker->color.a = 1.0;
    marker->color.r = 1.0;
    marker->color.g = 1.0;
    marker->color.b = 1.0;
}




void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    const static std::string FRAME_ID = "base_link";
    static tf::TransformListener listener;
    static tf::StampedTransform transform;
    enum COLOR_RGB{
        RED=0,
        GREEN,
        BLUE,
        COLOR_MAX
    };
    const int CLUSTER_MAX = 10;
    const int CLUSTER_COLOR[CLUSTER_MAX][COLOR_MAX] = {
        {230, 0, 18},{243, 152, 18}, {255, 251, 0},
        {143, 195, 31},{0, 153, 68}, {0, 158, 150},
        {0, 160, 233},{0, 104, 183}, {29, 32, 136},
        {146, 7, 131}
    };

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




    int cluster_i=0;
    visualization_msgs::MarkerArray markers;

    int j = 0;
    std::stringstream ss;
    ss << j;
    std::string str = ss.str();


        std::cout << "PointCloud representing the Cluster: " << cloud_passthrough->points.size () << " data points." << std::endl;
        
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


        ofstream File(j+"1.txt");
        j++;
       for(int nIndex=0;nIndex<cloud_passthrough->points.size();nIndex++){
            double x = cloud_passthrough->points[nIndex].x;
            double y = cloud_passthrough->points[nIndex].y;
            double z= cloud_passthrough->points[nIndex].z;

            double nx = cloud_normals->points[nIndex].normal[0];
            double ny = cloud_normals->points[nIndex].normal[1];
            double nz = cloud_normals->points[nIndex].normal[2];

            File<<x<<"  ";
            File<<y<<"  ";
            File<<z<<"  ";

            File<<nx<<"  ";
            File<<ny<<"  ";
            File<<nz<<"\n";

    }
        File.close();

  
    
    
}
    


int main (int argc, char** argv)
{
    ros::init (argc, argv, "object_detection");
    ros::NodeHandle nh("~");

    _pub_point_cloud = nh.advertise<sensor_msgs::PointCloud2> ("/sciurus17/example/points", 1);
    _pub_marker_array = nh.advertise<visualization_msgs::MarkerArray>("/sciurus17/example/markers", 1);

    ros::Subscriber sub = nh.subscribe("/sciurus17/camera/depth_registered/points", 1, cloud_cb);

    ros::spin ();
}
