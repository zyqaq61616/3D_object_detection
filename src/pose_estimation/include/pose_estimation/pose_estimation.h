//
// Created by zyq on 2021/11/29.
//

#ifndef SRC_POSE_ESTIMATION_H
#define SRC_POSE_ESTIMATION_H

#endif //SRC_POSE_ESTIMATION_H
#include <pcl/io/pcd_io.h>   //PCD读写类相关的头文件
#include <pcl/point_types.h> //PCL中支持的点类型的头文件

#ifndef SRC_BOTTLE_DETECTION_H
#define SRC_BOTTLE_DETECTION_H

#endif //SRC_BOTTLE_DETECTION_H

#include <ros/ros.h>
#include <iostream>
#include  <vector>
#include <Eigen/Core>
////消息同步的头文件
//#include <message_filters/subscriber.h>
//#include <message_filters/synchronizer.h>
//#include <message_filters/sync_policies/approximate_time.h>
//#include <boost/thread/thread.hpp>
// PCL
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
// registration
# include <pcl/registration/sample_consensus_prerejective.h>
//feature
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
//IO
#include <pcl/io/pcd_io.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include "sensor_msgs/image_encodings.h"
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Int16.h>
using namespace std;
using namespace ros;
using namespace pcl;
using namespace cv;
using namespace Eigen;
//点云类型
typedef pcl::PointXYZ PointType;
//法线类型
typedef pcl::PointNormal PointNT;
//存储法线的容器
typedef pcl::PointCloud<PointNT> PointCloudT;
//fpfh特征
typedef pcl::FPFHSignature33 FeatureT;
//
typedef pcl::FPFHEstimation<PointType ,PointNT,FeatureT> FeatureEstimationT;
//存储fpfh特征的容器
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
class pose_estimation
{
    struct information
            {
                pcl::PointCloud<PointType>::Ptr clo;
                FeatureCloudT::Ptr fea=nullptr;

            };
public:
    ros::Subscriber sub_predict_result;
    ros::Subscriber sub_object_cloud;
    ros::Publisher pub_aligned;
    ros::Publisher pub_result;
public:
    pose_estimation(ros::NodeHandle &n,int number);
    void get_classification_result_cb(const std_msgs::Int16& cla);
    void get_object_cb(const sensor_msgs::PointCloud2& cloud_msg);
    void get_keypoint(const sensor_msgs::PointCloud2& cloud_msg);
    void load_keypoint(int number,vector<information>& vec);
    ~pose_estimation();
private:
    //记录当前分类的结果
    int m_class;
    vector<information> model;

};