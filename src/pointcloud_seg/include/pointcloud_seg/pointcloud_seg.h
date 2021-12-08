//
// Created by zyq on 2021/11/19.
//

#ifndef SRC_POINTCLOUD_SEG_H
#define SRC_POINTCLOUD_SEG_H

#endif //SRC_POINTCLOUD_SEG_H
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
// torch
#include "torch/script.h"
#include "torch/torch.h"
using namespace std;
using namespace ros;
using namespace pcl;
using namespace cv;
using namespace Eigen;
typedef pcl::PointXYZ PointType;
// 用于分割传感器输入点云的类
class pointcloud_seg
{
public:
    ros::Subscriber sub;
    ros::Publisher pub_object;
public:
    pointcloud_seg(ros::NodeHandle &n);
    void get_pointcloud_cb(const sensor_msgs::PointCloud2& cloud_msg);
    void save_model_cb(const sensor_msgs::PointCloud2& cloud_msg);
    PointCloud<PointType>::Ptr best_obj(const vector<pcl::PointIndices> &cluster_indices, const PointCloud<PointType>::Ptr &msg);
    ~pointcloud_seg();

};
//用于对点云进行分类的类
class pointcloud_classification
{
public:
    Subscriber sub;
    Publisher pub_predict;
    Publisher pub_object;

public:
    pointcloud_classification(ros::NodeHandle &n,string path);
    void model_predict_cb(const sensor_msgs::PointCloud2& cloud_msg);
    int process_data(const pcl::PointCloud<PointType>::Ptr &msg);//点云下采样
    void test_module();
    ~pointcloud_classification();

private:
    torch::jit::script::Module module;
};
