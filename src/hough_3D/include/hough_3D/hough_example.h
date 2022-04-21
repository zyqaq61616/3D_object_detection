//
// Created by zyq on 2021/9/7.
//

#ifndef SRC_HOUGH_EXAMPLE_H
#define SRC_HOUGH_EXAMPLE_H

#endif //SRC_HOUGH_EXAMPLE_H
//基础头文件
#include <ros/ros.h>
#include <string>
#include <iostream>
#include  <vector>
#include <Eigen/Core>
#include<time.h>
//pcl头文件
//
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h> //io模块
#include <pcl/point_cloud.h>//数据类型
#include <pcl/point_types.h>
#include <pcl/correspondence.h>//通信
#include <pcl/features/normal_3d_omp.h>//特征 法线（omp）加速
#include <pcl/features/shot_omp.h> //shot特征 （omp）加速
#include <pcl/features/board.h> //board特征
#include <pcl/filters/uniform_sampling.h> //滤波 均匀采样
#include <pcl/recognition/cg/hough_3d.h> //识别 霍夫——3d
#include <pcl/recognition/cg/geometric_consistency.h> //识别 几何一致性
#include <pcl/visualization/pcl_visualizer.h> //可视化
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h> //树（快速最近邻搜索）
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h> //转换模块
#include <pcl/console/parse.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include<pcl/features/normal_3d_omp.h>
#include<boost/thread/thread.hpp>
#include<pcl/keypoints/iss_3d.h>
#include<pcl/keypoints/sift_keypoint.h>
#include<pcl/keypoints/harris_3d.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/filters/radius_outlier_removal.h>


using namespace std;
using namespace ros;
using namespace Eigen;

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;


class object_detection
{
public:
    //目标是发布一个6D坐标和识别匹配后的对应图
ros::Publisher pub_coordinate;//发布匹配后的6D坐标
ros::Publisher pub_result;//发布匹配后的对应图
ros::Subscriber sub_model;//接受模型点云
ros::Subscriber sub_scene;//接受场景点云
// Load parameters from launch file
ros::NodeHandle nh_private;
bool model_ok;//成功读取模型点云标志位
bool scene_ok;//成功读取场景点云标志位
bool show_keypoints;
bool show_correspondences;
bool use_cloud_resolution;
double model_ss; //下采样
double scene_ss;
double descr_rad;
double rf_rad;
double cg_size;
double cg_thresh;

public:
    object_detection(ros::NodeHandle &n);//构造函数
    pcl::PointCloud<PointType>::Ptr read_scene_file();

    pcl::PointCloud<PointType>::Ptr read_model_file();

    bool ready(bool model, bool sence);//是否成功获取场景和目标点云

    //void model_file_cb(const sensor_msgs::PointCloud2& model_cloud_msg);//获取模型的回调函数

    //void scene_file_cb(const sensor_msgs::PointCloud2& scene_cloud_msg);//获取场景的回调函数

    pcl::PointCloud<PointType>::Ptr voxel_downsample(const pcl::PointCloud<PointType>::Ptr &model_keypoints);
    void hough_3d_object_detection(const pcl::PointCloud<PointType>::Ptr &model_keypoints,
                                   const pcl::PointCloud<NormalType>::Ptr &model_normals,
                                   const pcl::PointCloud<PointType>::Ptr &model,
                                   const pcl::PointCloud<PointType>::Ptr &scene_keypoints,
                                   const pcl::PointCloud<NormalType>::Ptr &scene_normals,
                                   const pcl::PointCloud<PointType>::Ptr &scene,
                                   const pcl::CorrespondencesPtr& model_scene_corrs );//基于霍夫投票 需要法线 下采样点 原图像点
                                   void geometric_consistency_object_detection(const pcl::PointCloud<PointType>::Ptr &model,
                                                                               const pcl::PointCloud<PointType>::Ptr &scene,const pcl::PointCloud<PointType>::Ptr &model_keypoints,
                                                const pcl::PointCloud<PointType>::Ptr &scene_keypoints,
                                                const pcl::CorrespondencesPtr& model_scene_corrs);//基于几何一致性

    pcl::PointCloud<NormalType>::Ptr get_normals(const pcl::PointCloud<PointType>::Ptr &msg);//获取法线

    pcl::PointCloud<PointType>::Ptr get_downsample(const pcl::PointCloud<PointType>::Ptr &msg,double index);//点云下采样

    pcl::PointCloud<PointType>::Ptr move_outliner_sample(const pcl::PointCloud<PointType>::Ptr &msg);//点云下采样

    pcl::PointCloud<PointType>::Ptr iss_keypoint(const pcl::PointCloud<PointType>::Ptr &msg);//iss关键点提取
    pcl::PointCloud<PointType>::Ptr sift_keypoint(const pcl::PointCloud<PointType>::Ptr &msg);//sift关键点提取
    //pcl::PointCloud<PointType>::Ptr harri3D_keypoint(const pcl::PointCloud<PointType>::Ptr &msg);//harri3D关键点提取
    pcl::PointCloud<DescriptorType>::Ptr shot_features(const pcl::PointCloud<NormalType>::Ptr& normal,const pcl::PointCloud<PointType>::Ptr& keypoint,const pcl::PointCloud<PointType>::Ptr& surface,double index);

    pcl::CorrespondencesPtr Model_Scene_Correspondences(const pcl::PointCloud<DescriptorType>::Ptr& scene_shot_keypoints,const pcl::PointCloud<DescriptorType>::Ptr& model_shot_keypoints);

    void print_result(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations,std::vector<pcl::Correspondences> clustered_corrs);

    void visualization_result(const pcl::PointCloud<PointType>::Ptr &scene,
                              const pcl::PointCloud<PointType>::Ptr &model,
                              const pcl::PointCloud<PointType>::Ptr &model_keypoints,
                              const pcl::PointCloud<PointType>::Ptr &scene_keypoints,std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations,
                              std::vector<pcl::Correspondences> clustered_corrs);

    void pre_visualization(const pcl::PointCloud<PointType>::Ptr &scene,
                           const pcl::PointCloud<PointType>::Ptr &model,
                           const pcl::PointCloud<PointType>::Ptr &model_keypoints,
                           const pcl::PointCloud<PointType>::Ptr &scene_keypoints,std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations,
                           std::vector<pcl::Correspondences> clustered_corrs);
    void one_picture(const pcl::PointCloud<PointType>::Ptr &scene,
                     const pcl::PointCloud<PointType>::Ptr &model);
    int only_print_the_most_likely(const std::vector<pcl::Correspondences>& clustered_corrs);
    ~object_detection();
};