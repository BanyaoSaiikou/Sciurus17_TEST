#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <math.h> 

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>

#include <iostream>
using namespace std;

static ros::Publisher _pub_point_cloud;
static ros::Publisher _pub_marker_array;


void convert_to_marker(visualization_msgs::Marker *marker, const int marker_id,
        const std::string &frame_id,
        const pcl::PointXYZRGB &min_pt, const pcl::PointXYZRGB &max_pt, int r, int g, int b)
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
    marker->color.r = r / 255.0;
    marker->color.g = g / 255.0;
    marker->color.b = b / 255.0;
}

int get_nearest_color(float r, float g, float b) {

    int min_dist_idx = -1;
    float min_dist = 10000;
    float max_dist = 200;
    float color[3][3] = {{200.0, 200.0, 0}, {0, 255.0, 0}, {0, 0, 255.0}};

    for (int i = 0; i < 3; i++) {

	float dist = sqrt((color[i][0]-r)*(color[i][0]-r) + (color[i][1]-g)*(color[i][1]-g) + (color[i][2]-b)*(color[i][2]-b));
	if(dist <= min_dist) {
		min_dist = dist;
		min_dist_idx = i;
	}
    }

    if(min_dist <= max_dist)
	return min_dist_idx;

    return -1;
}


void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    const static std::string FRAME_ID = "base_link";
    const static std::string CAMERA_FRAME_ID= "camera_depth_optical_frame";
    static tf::TransformListener listener;
    static tf::StampedTransform transform;
    enum COLOR_RGB{
        RED=0,
        GREEN,
        BLUE,
        COLOR_MAX
    };
    const int CLUSTER_MAX = 20;
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
            listener.lookupTransform(FRAME_ID, CAMERA_FRAME_ID, ros::Time(0), transform);
            break;
        }
        catch(tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
    }
    pcl_ros::transformPointCloud(FRAME_ID, transform, *cloud_msg, cloud_transformed);

    // sensor_msgs/PointCloud2からpcl/PointCloudに変換
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>()); 
    pcl::fromROSMsg(cloud_transformed, *cloud);

    // Z方向の範囲でフィルタリング
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_passthrough_1 (new pcl::PointCloud<pcl::PointXYZRGB>()); 
    pcl::PassThrough<pcl::PointXYZRGB> pass_1;
    pass_1.setInputCloud (cloud);
    pass_1.setFilterFieldName ("z");
    //pass_1.setFilterLimits (0.02, 1.0);
    pass_1.setFilterLimits (0.02, 0.3);
    pass_1.filter (*cloud_passthrough_1);

    // X方向の範囲でフィルタリング
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_passthrough_2 (new pcl::PointCloud<pcl::PointXYZRGB>()); 
    pcl::PassThrough<pcl::PointXYZRGB> pass_2;
    pass_2.setInputCloud (cloud_passthrough_1);
    pass_2.setFilterFieldName ("x");
    pass_2.setFilterLimits (0.0, 1.0);
    pass_2.filter (*cloud_passthrough_2);

    // Y方向の範囲でフィルタリング
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_passthrough (new pcl::PointCloud<pcl::PointXYZRGB>()); 
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud_passthrough_2);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-0.5, 1.0);
    pass.filter (*cloud_passthrough);

    // voxelgridでダウンサンプリング
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_voxelgrid (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::VoxelGrid<pcl::PointXYZRGB> voxelgrid;
    voxelgrid.setInputCloud (cloud_passthrough);
    voxelgrid.setLeafSize (0.01, 0.01, 0.01);
    voxelgrid.filter (*cloud_voxelgrid);

    // pointがなければ処理を抜ける
    if(cloud_voxelgrid->size() <= 0){
        return;
    }

    // クラスタ抽出のためKdTreeオブジェクトを作成
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>());
    tree->setInputCloud (cloud_voxelgrid);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance (0.02);
    ec.setMinClusterSize (10);
    ec.setMaxClusterSize (250);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_voxelgrid);
    ec.extract (cluster_indices); 


    int cluster_i=0;
    visualization_msgs::MarkerArray markers;

    // クラスタごとにPointのRGB値を変更する
    // クラスタをもとにMarkerを生成する
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_output (new pcl::PointCloud<pcl::PointXYZRGB>());
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); 
            it != cluster_indices.end (); ++it)
    {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_0 (new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_1 (new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_2 (new pcl::PointCloud<pcl::PointXYZRGB>());

        // クラスタ内のPointRGB値を変更
	int count[3] = {0,0,0};
	int r[3] = {0,0,0};
	int g[3] = {0,0,0};
	int b[3] = {0,0,0};
        for (std::vector<int>::const_iterator pit = it->indices.begin (); 
                pit != it->indices.end (); ++pit){
	    int color_idx = get_nearest_color(cloud_voxelgrid->points[*pit].r, cloud_voxelgrid->points[*pit].g, cloud_voxelgrid->points[*pit].b);
	    if(color_idx == -1)
	    	continue;
	    //color_idx = 0;	
	    r[color_idx] += cloud_voxelgrid->points[*pit].r;
	    g[color_idx] += cloud_voxelgrid->points[*pit].g;
	    b[color_idx] += cloud_voxelgrid->points[*pit].b;
            //cloud_voxelgrid->points[*pit].r = CLUSTER_COLOR[cluster_i][RED];
            //cloud_voxelgrid->points[*pit].g = CLUSTER_COLOR[cluster_i][GREEN];
            //cloud_voxelgrid->points[*pit].b = CLUSTER_COLOR[cluster_i][BLUE];
	    if (color_idx == 0)
	            cloud_cluster_0->points.push_back (cloud_voxelgrid->points[*pit]);
    	    else if (color_idx == 1)
	            cloud_cluster_1->points.push_back (cloud_voxelgrid->points[*pit]);
	    else if (color_idx == 2)
	            cloud_cluster_2->points.push_back (cloud_voxelgrid->points[*pit]);
	    count[color_idx]++;
        }

	for (int i=0; i<3; i++) {
		if(count[i] == 0)
			continue;
		
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster;
		if (i == 0)
			cloud_cluster = cloud_cluster_0;
		else if (i == 1)
			cloud_cluster = cloud_cluster_1;
		else if (i == 2)
			cloud_cluster = cloud_cluster_2;

		r[i] = r[i] / count[i];
		g[i] = g[i] / count[i];
		b[i] = b[i] / count[i];
		// Unorganized datasetsとしてwidth, heightを入力
		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		// 無効なpointがないのでis_denseはtrue
		cloud_cluster->is_dense = true;
		*cloud_output += *cloud_cluster;

		// Markerの作成
		visualization_msgs::Marker marker;
		pcl::PointXYZRGB min_pt, max_pt;
		pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt);
		convert_to_marker(&marker, cluster_i, FRAME_ID, min_pt, max_pt, r[i], g[i], b[i]);
		markers.markers.push_back(marker);

		cluster_i++;
		if(cluster_i >= CLUSTER_MAX){
            		break;
        	}
	}

        if(cluster_i >= CLUSTER_MAX){
            break;
        }
    }

    // pcl/PointCloudからsensor_msgs/PointCloud2に変換
    sensor_msgs::PointCloud2 output;
    // pcl::toROSMsg(*cloud, output);
    // pcl::toROSMsg(*cloud_passthrough, output);
    // pcl::toROSMsg(*cloud_voxelgrid, output);
    pcl::toROSMsg(*cloud_output, output);
    output.header.frame_id = FRAME_ID;

    _pub_point_cloud.publish(output);
    _pub_marker_array.publish(markers);
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

