//
// Created by zyq on 2021/11/29
//

#include "make_dataset/make_dataset.h"

// main function
int main(int argc, char **argv)
{
    // Initialize ROS
    ros::init(argc, argv, "make_dataset");
    ros::NodeHandle n;
    string name="seesaw"
                "";
    string path="/home/zyq/catkin_ws/src/make_dataset/model_dataset/";
    int number=1100;
    n.getParam("name",name);
    n.getParam("number",number);
    n.getParam("path",path);
    make_dataset mkd (n,name,number,path);
    ros::spin();
    return 0;
}