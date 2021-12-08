//
// Created by zyq on 2021/11/19.
//
//
// Created by zyq on 2021/9/13.
#include "pointcloud_seg/pointcloud_seg.h"
// main function
int main(int argc, char **argv)
{
    // Initialize ROS
    ros::init(argc, argv, "object_recognition");
    ros::NodeHandle n;
//    pointcloud_seg seg(n);
    string path="/home/zyq/catkin_ws/src/pointcloud_seg/model/script_model_1.pt";
    pointcloud_seg seg(n);
    pointcloud_classification cls(n,path);
    ros::spin();
    return 0;
}
