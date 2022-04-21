//
// Created by zyq on 2021/9/12.
//

#ifndef SRC_SCENE_MODEL_GET_H
#define SRC_SCENE_MODEL_GET_H

#endif //SRC_SCENE_MODEL_GET_H
#include <pcl/io/pcd_io.h>   //PCD读写类相关的头文件
#include <pcl/point_types.h> //PCL中支持的点类型的头文件

#ifndef SRC_BOTTLE_DETECTION_H
#define SRC_BOTTLE_DETECTION_H

#endif //SRC_BOTTLE_DETECTION_H

#include <ros/ros.h>
#include <string>
#include <iostream>
#include  <vector>
#include <Eigen/Core>
// PCL
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
// Filters
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include<pcl/filters/radius_outlier_removal.h>
#include <pcl/keypoints/uniform_sampling.h>
//segment
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
//feature
#include <pcl/features/normal_3d.h>
#include <image_transport/image_transport.h>
//IO
#include <pcl/io/pcd_io.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include "sensor_msgs/image_encodings.h"
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace ros;
using namespace pcl;
using namespace cv;
using namespace Eigen;
typedef pcl::PointXYZRGB PointType;
class save_pointcloud
{
public:
    ros::Subscriber sub_model;
    ros::Subscriber sub_scene;
    ros::Subscriber sub_model_rgb;
    ros::Publisher pub_model;
    ros::Publisher pub_scene;
    bool scene_ok;
    bool model_ok;
public:
    save_pointcloud(ros::NodeHandle &n);
    void save_scene_cb(const sensor_msgs::PointCloud2& cloud_msg);
    void save_model_cb(const sensor_msgs::PointCloud2& cloud_msg);
    ~save_pointcloud();

};