//
// Created by zyq on 2021/11/29.
//

#ifndef SRC_MAKE_DATASET_H
#define SRC_MAKE_DATASET_H

#endif //SRC_MAKE_DATASET_H
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
typedef pcl::PointXYZ PointType;
// 用于分割传感器输入点云的类
class make_dataset
{
public:
    ros::Subscriber sub;
    ros::Publisher pub;
public:
    make_dataset(ros::NodeHandle &n,string name,int number,string path);
    void save_dataset_cb(const sensor_msgs::PointCloud2& cloud_msg);
    PointCloud<PointType>::Ptr best_obj(const vector<pcl::PointIndices> &cluster_indices, const PointCloud<PointType>::Ptr &msg);
    ~make_dataset();
private:
    string m_name;
    string m_path;
    int m_number;
    int m_index;
};