//
// Created by zyq on 2021/11/25.
#include <pointcloud_seg/pointcloud_seg.h>
#include <iostream>
#include "torch/script.h"
#include <string>
#include "boost/filesystem.hpp"
#include <numeric>
#include <algorithm>
#include <pcl/filters/random_sample.h>
#include<typeinfo>
#include<std_msgs/Int16.h>
#include <cmath>
//using namespace nc;
vector<float> load_pointcloud(const pcl::PointCloud<PointType>::Ptr &msg)
{

    vector<float> vec;
    for(int i=0;i < msg->points.size();i++)
    {
        vec.push_back(msg->points[i].x);
        vec.push_back(msg->points[i].y);
        vec.push_back(msg->points[i].z);
    }
    return vec;

}
//将数据去均值和归一化
vector<float> pc_normalize(const pcl::PointCloud<PointType>::Ptr &msg)
{
    vector<float> x;
    vector<float> y;
    vector<float> z;
    vector<float> m;
    //先将点云的X,Y,Z坐标分别放入三个vector容器中
    for(int i=0;i < msg->points.size();i++)
    {
        x.push_back(msg->points[i].x);
        y.push_back(msg->points[i].y);
        z.push_back(msg->points[i].z);

    }
    //求每个通道的均值
    float sumValueX = accumulate(x.begin(), x.end(), 0.0);   // accumulate函数就是求vector和的函数；
    float meanValueX = sumValueX / x.size();
    float sumValueY = accumulate(y.begin(), y.end(), 0.0);   // accumulate函数就是求vector和的函数；
    float meanValueY = sumValueY / y.size();
    float sumValueZ = accumulate(z.begin(), z.end(), 0.0);   // accumulate函数就是求vector和的函数；
    float meanValueZ = sumValueZ / z.size();
    //
    vector<float> res;
    for(int i=0;i<x.size();i++)
    {
        float X=x[i]-meanValueX;
        float Y=y[i]-meanValueY;
        float Z=z[i]-meanValueZ;
        res.push_back((X));
        res.push_back(Y);
        res.push_back(Z);
        float sum=X*X+Y*Y+Z*Z;
        m.push_back(sqrt(sum));
    }
    auto max=max_element(m.begin(),m.end());
    float maxValue=*max;
    for(int i=0;i<res.size();i++)
    {
        res[i]=res[i]/maxValue;
    }
    return res;

}

pointcloud_classification::pointcloud_classification(ros::NodeHandle &n, string path)
{
    this->sub=n.subscribe("/Object",1,&pointcloud_classification::model_predict_cb,this);
    //发布
    this->pub_predict=n.advertise<std_msgs::Int16>("classification_result",1000); //发布平面点
    this->pub_object=n.advertise<sensor_msgs::PointCloud2>("Object_1024",1000); //发布平面点
    try
    {
        this->module=torch::jit::load(path);
        ROS_INFO("Module is loaded sucessfully!");
    }
    catch(const c10::Error& e)
    {
        std::cerr << "error loading the model\n";
        ROS_INFO("Can not load module!");
        exit(1);
    }
}

void pointcloud_classification::model_predict_cb(const sensor_msgs::PointCloud2 &cloud_msg)
{
    //接收点云转化为pcl格式
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(cloud_msg,*cloud);
    //对点云数据进行处理
    pcl::PointCloud<PointType>::Ptr cloud_processed(new pcl::PointCloud<PointType>);
    int result=this->process_data(cloud);
    vector<string> class_type{"chair", "stair", "box", "seesaw", "railing", "shelf"};
    cout<<"检测到的目标种类为："<<class_type[result]<<endl;
    cout<<"检测到的目标种类为："<<class_type[result]<<endl;
    cout<<"检测到的目标种类为："<<class_type[result]<<endl;
    cout<<"检测到的目标种类为："<<class_type[result]<<endl;
    cout<<"检测到的目标种类为："<<class_type[result]<<endl;

}

//处理接受到的点云数据
int pointcloud_classification::process_data(const pcl::PointCloud<PointType>::Ptr &msg)
{
    ros::Time begin = ros::Time::now();
    pcl::PointCloud<PointType>::Ptr points1024(new pcl::PointCloud<PointType>);
    pcl::RandomSample<PointType> rs;
    rs.setInputCloud(msg);
    rs.setSample(1024);
    rs.filter(*points1024);
    sensor_msgs::PointCloud2 obj_output;
    pcl::toROSMsg(*points1024, obj_output);//第一个参数是输入，后面的是输出
    pub_object.publish(obj_output);
    vector<float> Ipoints=pc_normalize(msg);
    torch::Tensor npoints=torch::from_blob(Ipoints.data(), {1, 1024, 3 }, torch::kFloat32);
    npoints=npoints.transpose(2,1);
    npoints=npoints.to(at::kCUDA);
    module.to(at::kCUDA);
    std::vector<torch::jit::IValue> input;
    input.emplace_back(npoints);
    ros::Time data_process = ros::Time::now();
    ros::Duration data_process_time = data_process - begin;
    cout<<"Time of data_process is"<<data_process_time.toSec()<<"."<<endl;
    torch::Tensor output = module.forward(input).toTuple()->elements()[0].toTensor();
    auto max_result = output.max(1, true);
    auto max_index = std::get<1>(max_result).item<float>();
    std_msgs::Int16 pubData;
    pubData.data = int(max_index);
    ros::Time predict = ros::Time::now();
    ros::Duration predict_time = predict - data_process;
    cout<<"Time of predict is"<<predict_time.toSec()<<"."<<endl;
    this->pub_predict.publish(pubData);
    return int(max_index);


}
pointcloud_classification::~pointcloud_classification()
{

}