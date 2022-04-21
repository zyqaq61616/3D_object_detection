#include "hough_3D/hough_example.h"
//1.先实现读取场景和模型点云
//2.获取关键点
//3.根据霍夫投票或者几何一致性进行匹配
//4.位姿估计，获取6D坐标
//5.得出摄像头对应回到正面的旋转矩阵
//6.考虑速度问题
ros::Publisher pub;

pcl::PointCloud<PointType>::Ptr read_model_file()
{

}
// main function
int main(int argc, char **argv)
{

    // Initialize ROS
    ros::init(argc, argv, "object_recognition");
    ros::NodeHandle n;
    object_detection  obj(n);
    pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr scene(new pcl::PointCloud<PointType> ());
    scene=obj.read_scene_file();
    model=obj.read_model_file();
    obj.one_picture(scene,model);
    //open_serial ope(n);
    ros::spin();
    return 0;
}