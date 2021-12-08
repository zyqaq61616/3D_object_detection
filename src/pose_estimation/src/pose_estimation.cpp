//
// Created by zyq on 2021/12/5.
//
#include "pose_estimation/pose_estimation.h"
#include <eigen_conversions/eigen_msg.h>
//构造函数
pose_estimation::pose_estimation(ros::NodeHandle &n,int number)
{
    //接受分割出的目标点云
    this->sub_object_cloud=n.subscribe("/Object",1,&pose_estimation::get_object_cb,this);
    //接受分类的结果
    this->sub_predict_result=n.subscribe("/classification_result",1,&pose_estimation::get_classification_result_cb,this);
    //发布分割的结果和旋转平移矩阵
    this->pub_aligned=n.advertise<sensor_msgs::PointCloud2>("aligned",1000); //发布平面点
    this->pub_result=n.advertise<std_msgs::Float64MultiArray>("aligned",1000); //发布平面点
    //接收分类的结果，-1代表不需要进行匹配
    this->m_class=-1;
    vector<information> infor(number);
    //加载线下模型的keypoints
    this->load_keypoint(number,infor);
    this->model=infor;
}
//获取分类的结果
void pose_estimation::get_classification_result_cb(const std_msgs::Int16 & cla)
{
    this->m_class=int(cla.data);
}
//获取待估计障碍物的点云
void pose_estimation::get_object_cb(const sensor_msgs::PointCloud2 &cloud_msg) {
    //没有得到分类的结果，就返回
    ros::Time begin = ros::Time::now();
    if (this->m_class == -1) {
        return;
    }
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    PointCloudT::Ptr cloud_normal(new PointCloudT);
    FeatureCloudT::Ptr cloud_features(new FeatureCloudT);
    pcl::PointCloud<PointType>::Ptr cloud_aligned(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(cloud_msg, *cloud);
    //估计模型的法线信息
    pcl::NormalEstimation<PointType, PointNT> nest;
    nest.setRadiusSearch(0.01);
    nest.setInputCloud(cloud);
    nest.compute(*cloud_normal);
    //计算特征
    FeatureEstimationT fest;
    fest.setRadiusSearch(0.025);//该搜索半径决定FPFH特征描述的范围，一般设置为分辨率10倍以上
    fest.setInputCloud(cloud);
    fest.setInputNormals(cloud_normal);
    fest.compute(*cloud_features);
    ros::Time data_process = ros::Time::now();
    ros::Duration feature_get_time = data_process - begin;
    cout<<"Time of get_feature is"<<feature_get_time.toSec()<<"."<<endl;
    //姿态估计
    // 实施配准
    pcl::SampleConsensusPrerejective<PointType, PointType, FeatureT> align;//基于采样一致性的位姿估计
    align.setInputSource(cloud);
    align.setSourceFeatures(cloud_features);
    //判断用哪一类模型和他配准
    //不知道这块会不会越界
    switch (this->m_class) {
        case 0:
            align.setInputTarget(this->model[0].clo);
            align.setTargetFeatures(this->model[0].fea);
            break;
        case 1:
            align.setInputTarget(this->model[1].clo);
            align.setTargetFeatures(this->model[1].fea);
            break;
        case 2:
            align.setInputTarget(this->model[2].clo);
            align.setTargetFeatures(this->model[2].fea);
            break;
        case 3:
            align.setInputTarget(this->model[3].clo);
            align.setTargetFeatures(this->model[3].fea);
            break;
        case 4:
            align.setInputTarget(this->model[4].clo);
            align.setTargetFeatures(this->model[4].fea);
            break;
        case 5:
            align.setInputTarget(this->model[5].clo);
            align.setTargetFeatures(this->model[5].fea);
            break;
        case 6:
            align.setInputTarget(this->model[6].clo);
            align.setTargetFeatures(this->model[6].fea);
            break;
        case 7:
            align.setInputTarget(this->model[7].clo);
            align.setTargetFeatures(this->model[7].fea);
            break;

    }
    align.setMaximumIterations(20000);  //  采样一致性迭代次数
    align.setNumberOfSamples(3);          //  创建假设所需的样本数，为了正常估计至少需要3点
    align.setCorrespondenceRandomness(5); //  使用的临近特征点的数目
    align.setSimilarityThreshold(0.9f);   //  多边形边长度相似度阈值
    align.setMaxCorrespondenceDistance(2.5f * 0.005); //  判断是否为内点的距离阈值
    align.setInlierFraction(0.25f);       //接受位姿假设所需的内点比例
    //实施配准
    align.align(*cloud_aligned);
    //配准时间
    ros::Time aligen_time = ros::Time::now();
    ros::Duration aligened_time = aligen_time - data_process;
    cout<<"Time of aligen is"<<aligened_time.toSec()<<"."<<endl;
    //发布配准后的点云结果
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud_aligned, output);
    pub_aligned.publish(output);
    if (align.hasConverged()) {
        // 返回配准的结果
        printf("\n");
        Eigen::Matrix4f transformation = align.getFinalTransformation();
        std_msgs::Float64MultiArray array;
        tf::matrixEigenToMsg(transformation,array);
        pub_result.publish(array);
        pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(0, 0), transformation(0, 1),
                                 transformation(0, 2));
        pcl::console::print_info("R = | %6.3f %6.3f %6.3f | \n", transformation(1, 0), transformation(1, 1),
                                 transformation(1, 2));
        pcl::console::print_info("    | %6.3f %6.3f %6.3f | \n", transformation(2, 0), transformation(2, 1),
                                 transformation(2, 2));
        pcl::console::print_info("\n");
        pcl::console::print_info("t = < %0.3f, %0.3f, %0.3f >\n", transformation(0, 3), transformation(1, 3),
                                 transformation(2, 3));
        pcl::console::print_info("\n");
        pcl::console::print_info("Inliers: %i/%i\n", align.getInliers().size(), cloud->size());

    }
    else
    {
        ROS_INFO("Alignment failed!\n");
        return;
    }
}
//初始化以后加载已知模型的关键点
void pose_estimation::load_keypoint(int number,vector<information>& infor)
{
    //循环读入模型的法线和FPFH特征
    for(int i=0;i<number;i++)
    {
        pcl::PointCloud<PointType>::Ptr object(new PointCloud<PointType>);
        PointCloudT::Ptr object_normal(new PointCloudT);
        FeatureCloudT::Ptr object_features(new FeatureCloudT);
        string path="../model/model"+to_string(i)+".pcd";
        if (pcl::io::loadPCDFile<PointType>(path, *object) < 0)
        {
            ROS_INFO("Error loading object file!");
            cout<<"Error loading object file!"<<endl;
            exit(1);
        }
        //估计模型的法线信息
        pcl::NormalEstimation<PointType , PointNT> nest;
        nest.setRadiusSearch(0.01);
        nest.setInputCloud(object);
        nest.compute(*object_normal);
        //计算特征
        FeatureEstimationT fest;
        fest.setRadiusSearch(0.025);//该搜索半径决定FPFH特征描述的范围，一般设置为分辨率10倍以上
        fest.setInputCloud(object);
        fest.setInputNormals(object_normal);
        fest.compute(*object_features);
        //判断加载的是哪个模型
        switch(i)
        {
            case 0:
                infor[0].clo=object;
                infor[0].fea=object_features;
                break;
            case 1:
                infor[1].clo=object;
                infor[1].fea=object_features;
                break;
            case 2:
                infor[2].clo=object;
                infor[2].fea=object_features;
                break;
            case 3:
                infor[3].clo=object;
                infor[3].fea=object_features;
                break;
            case 4:
                infor[4].clo=object;
                infor[4].fea=object_features;
                break;
            case 5:
                infor[5].clo=object;
                infor[5].fea=object_features;
                break;
            case 6:
                infor[6].clo=object;
                infor[6].fea=object_features;
                break;
            case 7:
                infor[7].clo=object;
                infor[7].fea=object_features;
                break;
        }

    }
}
//析构函数
pose_estimation::~pose_estimation()
{

}
