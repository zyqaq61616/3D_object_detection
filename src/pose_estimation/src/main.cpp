//
// Created by zyq on 2021/11/29.
#include "pose_estimation/pose_estimation.h"

// main function
int main(int argc, char **argv)
{
    // Initialize ROS
    ros::init(argc, argv, "object_recognition");
    ros::NodeHandle n;
    //设置要加载的模型数量，最多为8个
    int number=6;
     pose_estimation pos(n,number);
    ros::spin();
    return 0;
}
