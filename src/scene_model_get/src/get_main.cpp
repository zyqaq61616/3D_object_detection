//
// Created by zyq on 2021/9/13.
////用于保存传感器获取的点云 用于后续匹配
#include"scene_model_get/scene_model_get.h"
// Definitions
ros::Publisher pub;

std::string subscribed_topic;
std::string published_topic; //经过地面分割的节点

// main function
int main(int argc, char **argv)
{
    // Initialize ROS
    ros::init(argc, argv, "object_recognition");
    ros::NodeHandle n;
    save_pointcloud sav(n);
    ros::spin();
    return 0;
}

