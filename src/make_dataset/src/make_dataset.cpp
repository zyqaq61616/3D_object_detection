//
// Created by zyq on 2021/11/29.
//
#include "make_dataset/make_dataset.h"
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
make_dataset::make_dataset(ros::NodeHandle &n,string name,int number,string path)
{
    //用于接受点云的节点和发布点云的节点
    this->sub=n.subscribe("/points2",1,&make_dataset::save_dataset_cb,this);
    this->pub=n.advertise<sensor_msgs::PointCloud2_<PointType>>("Save_result",100);
    this->m_name=name;
    this->m_number=number;
    this->m_index=0;
    this->m_path=path;
    string folderPath=path+name;
    try {
        if (access(folderPath.c_str(), 0) == -1) 
        {
            int flag = mkdir(folderPath.c_str(), S_IRWXU);
            if (flag == 0)
            {  //创建成功
                ROS_INFO("Create directory successfully!");
            }
            else
            { //创建失败
                ROS_INFO("Fail to create directory!");
                throw std::exception();
            }
        }
        else
        {
            ROS_INFO("Fail to create directory！");
            throw std::exception();
        }
    }
    catch (exception& e)
    {
        exit(1);
    }
}
void make_dataset::save_dataset_cb(const sensor_msgs::PointCloud2 &cloud_msg)
{
    if(m_index>m_number)
    {
        ROS_INFO("The number of data is enough!");
        return;
    }
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(cloud_msg,*cloud);
    ///////////////////////////////直通滤波↓
    pcl::PointCloud<PointType>::Ptr cloudPtrx(new pcl::PointCloud<PointType>);
    pcl::PassThrough<PointType> px_filter;
    px_filter.setInputCloud(cloud); // Pass filtered_cloud to the filter
    px_filter.setFilterFieldName("x"); // Set axis x
    px_filter.setFilterLimits(-1, 1); // Set limits min_value to max_value
    px_filter.filter(*cloudPtrx); // Restore output data in second_cloud

    // Perform the PassThrough Filter to the Y axis
    pcl::PointCloud<PointType>::Ptr cloudPtry(new pcl::PointCloud<PointType>);
    pcl::PassThrough<PointType> py_filter;
    py_filter.setInputCloud(cloudPtrx); // Pass filtered_cloud to the filter
    py_filter.setFilterFieldName("y"); // Set axis y
    py_filter.setFilterLimits(-1, 1); // Set limits min_value to max_value
    py_filter.filter(*cloudPtry); // Restore output data in second_cloud

    // Perform the PassThrough Filter to the Z axis
    pcl::PointCloud<PointType>::Ptr cloudPtrz(new pcl::PointCloud<PointType>);
    pcl::PassThrough<PointType> pz_filter;
    pz_filter.setInputCloud(cloudPtry); // Pass filtered_cloud to the filter
    pz_filter.setFilterFieldName("z"); // Set axis z
    pz_filter.setFilterLimits(0, 1.5); // Set limits min_value to max_value
    pz_filter.filter(*cloudPtrz); // Restore output data in second_cloud
    if(cloudPtrz->empty())
    {
        ROS_INFO("There are no objects in the given range!");
        return;
    }
    // Perform the VoxelGrid Filter
    pcl::PointCloud<PointType>::Ptr cloudPtrv(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> v_filter;
    v_filter.setInputCloud(cloudPtrz); // Pass raw_cloud to the filter
    v_filter.setLeafSize(0.003, 0.003,0.003); // Set leaf size
    v_filter.filter(*cloudPtrv); // Store output data in first_cloud
    pcl::PointCloud<PointType>::Ptr ground_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr obj_cloud(new pcl::PointCloud<PointType>);
    if (cloudPtrv->size() > 0) {
        //创建分割时所需要的模型系数对象，coefficients及存储内点的点索引集合对象inliers
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // 创建分割对象
        pcl::SACSegmentation<PointType> seg;
        // 可选择配置，设置模型系数需要优化
        seg.setOptimizeCoefficients(true);
        // 必要的配置，设置分割的模型类型，所用的随机参数估计方法，距离阀值，输入点云
        seg.setModelType(pcl::SACMODEL_PLANE);//设置模型类型
        seg.setMethodType(pcl::SAC_RANSAC);//设置随机采样一致性方法类型
        // you can modify the parameter below
        seg.setMaxIterations(1000);//设置最大迭代次数
        seg.setDistanceThreshold(0.02);//设定距离阀值，距离阀值决定了点被认为是局内点是必须满足的条件
        seg.setInputCloud(cloudPtrv);
        //引发分割实现，存储分割结果到点几何inliers及存储平面模型的系数coefficients
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
            ROS_INFO("Error! Could not found any inliers!");
            return;
        }
        // extract ground
        // 从点云中抽取分割的处在平面上的点集
        pcl::ExtractIndices<PointType> extractor;//点提取对象
        extractor.setInputCloud(cloudPtrv);
        extractor.setIndices(inliers);
        extractor.setNegative(true);
        extractor.filter(*obj_cloud);
        //为提取点云时使用的搜素对象利用输入点云cloud_filtered创建Kd树对象tree。
        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
        tree->setInputCloud(obj_cloud);//创建点云索引向量，用于存储实际的点云信息
        //用于存储聚类点的容器
        vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointType> ec;
        ec.setClusterTolerance(0.02); // 设置近邻搜索的搜索半径为2cm
        ec.setMinClusterSize(1024);//设置一个聚类需要的最少点数目为100
        ec.setMaxClusterSize(10000);//设置一个聚类需要的最大点数目为25000
        ec.setSearchMethod(tree);//设置点云的搜索机制
        ec.setInputCloud(obj_cloud);
        ec.extract(cluster_indices);//从点云中提取聚类，并将点云索引保存在cluster_indices中
        //迭代访问点云索引cluster_indices，直到分割出所有聚类
        if(cluster_indices.empty())
        {
            ROS_INFO("There are no objects!");
            return;
        }
        pcl::PointCloud<PointType>::Ptr cloud_cluster(new pcl::PointCloud<PointType>);
        cloud_cluster = this->best_obj(cluster_indices,obj_cloud);
        if(cloud_cluster->width!=obj_cloud->width)
        {
            sensor_msgs::PointCloud2 output;
            pcl::toROSMsg(*cloud_cluster, output);//第一个参数是输入，后面的是输出
            pub.publish(output);
            if(this->m_index>200)
            {
                string folderPath = this->m_path + this->m_name + "/" + this->m_name;
                pcl::io::savePCDFileASCII(folderPath + "_" + to_string(this->m_index-200) + ".pcd", *cloud_cluster);
                ROS_INFO("Save successfully!");
            }
            else
            {
                ROS_INFO("You can adjust your perspective during this time!");
            }
            this->m_index++;
        }

    }
}
PointCloud<PointType>::Ptr make_dataset::best_obj(const vector<pcl::PointIndices>& cluster_indices,const pcl::PointCloud<PointType>::Ptr &msg)
{
    int index=cluster_indices.size();
    pcl::PointCloud<PointType>::Ptr cloud_cluster(new pcl::PointCloud<PointType>);
    int clu=0;
    if(index ==0)
    {
        cout<<"There are no object in this pointcloud!"<<endl;
        return msg;
    }
    int maxmum=cluster_indices[0].indices.size();
    for(int i=0;i<index;i++)
    {
        if(cluster_indices[i].indices.size()>maxmum)
        {
            maxmum=cluster_indices[i].indices.size();
            clu = i;
        }
    }
    pcl::PointIndices max=cluster_indices[clu];
    for (vector<int>::const_iterator pit = max.indices.begin(); pit != max.indices.end(); pit++) {
        cloud_cluster->points.push_back(msg->points[*pit]);
        cloud_cluster->header = msg->header;
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
    }
    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points."<< std::endl;
    return cloud_cluster;
}
make_dataset::~make_dataset() {}
