
#include"scene_model_get/scene_model_get.h"
//构造函数
save_pointcloud::save_pointcloud(ros::NodeHandle &n) {
    //场景保存成功标志位
   scene_ok=false;
   //模型保存成功标志位
   model_ok=false;
   //用于接受场景的topic
   sub_scene=n.subscribe("/points2",1,&save_pointcloud::save_scene_cb,this);
   //用于接受模型topic
   sub_model=n.subscribe("/points2",1,&save_pointcloud::save_model_cb,this);
   //发布处理后的场景节点
   pub_scene=n.advertise<sensor_msgs::PointCloud2>("OUT_PUT_SCENE",1000);
   //发布处理后的模型节点
   pub_model=n.advertise<sensor_msgs::PointCloud2>("OUT_PUT_MODEL",1000);
    int rate  = 30;
    Mat globalImage(Size(640,480),CV_8UC3);
    int chess_num =0;
}
//场景点云很容易 滤波以后 直接保存 设置成S按键保存
void save_pointcloud::save_scene_cb(const sensor_msgs::PointCloud2& cloud_msg) {
    //类型转换将传感器的点云转换为PCL格式
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(cloud_msg,*cloud);

    //直通滤波
    pcl::PointCloud<PointType>::Ptr passthroughed(new pcl::PointCloud<PointType>);
    pcl::PassThrough<PointType> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0, 3);
    pass.filter(*passthroughed);

//    //下采样
//    pcl::VoxelGrid<PointType> v_filter;
//    pcl::PointCloud<PointType>::Ptr voxelGrided(new pcl::PointCloud<PointType>);
//    v_filter.setInputCloud(passthroughed); // Pass raw_cloud to the filter
//    v_filter.setLeafSize(0.005, 0.005, 0.005); // Set leaf size
//    v_filter.filter(*voxelGrided); // Store output data in first_cloud

    //离群点去除
//    pcl::RadiusOutlierRemoval<PointType> pcFilter;  //创建滤波器对象
//    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
//    pcFilter.setInputCloud(passthroughed);             //设置待滤波的点云
//    pcFilter.setRadiusSearch(0.8);               // 设置搜索半径
//    pcFilter.setMinNeighborsInRadius(2);      // 设置一个内点最少的邻居数目
//    pcFilter.filter(*cloud_filtered);        //滤波结果存储到cloud_filtered

    //把点云输出到RVIZ
    sensor_msgs::PointCloud2 output;//声明的输出的点云的格式
    pcl::toROSMsg(*passthroughed, output);//第一个参数是输入，后面的是输出
    pub_scene.publish(output);
    //加一个判断,按键捕获背景
    //把PointCloud对象数据存储在 test_pcd.pcd文件中
    //pcl::io::savePCDFileASCII("/home/zyq/catkin_ws/src/scene_model_get/save/scene_bottle*.pcd", *cloud_filtered);
}
//模型点云比较费劲 滤波加提取以后保存 设置成按F保存
void save_pointcloud::save_model_cb(const sensor_msgs::PointCloud2& cloud_msg) {
    ///////////////////////////////测量passthrough的数值与实际的对应关系↓
    pcl::PointCloud<PointType>::Ptr cloud_msg_xyz(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(cloud_msg, *cloud_msg_xyz);
    pcl::PointCloud<PointType>::Ptr cloudPtrx(new pcl::PointCloud<PointType>);
    pcl::PassThrough<PointType> px_filter;
    px_filter.setInputCloud(cloud_msg_xyz); // Pass filtered_cloud to the filter
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
//    // Perform the VoxelGrid Filter
    pcl::PointCloud<PointType>::Ptr cloudPtrv(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> v_filter;
    v_filter.setInputCloud(cloudPtrz); // Pass raw_cloud to the filter
    v_filter.setLeafSize(0.01, 0.01,0.01); // Set leaf size
    v_filter.filter(*cloudPtrv); // Store output data in first_cloud
    // Perform the Statistical Outlier Removal Filter
    //离群点去除
    pcl::RadiusOutlierRemoval<PointType> pcFilter;  //创建滤波器对象
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    pcFilter.setInputCloud(cloudPtrv);             //设置待滤波的点云
    pcFilter.setRadiusSearch(0.8);               // 设置搜索半径
    pcFilter.setMinNeighborsInRadius(2);      // 设置一个内点最少的邻居数目
    pcFilter.filter(*cloud_filtered);        //滤波结果存储到cloud_filtered
//    //基于RANSAC实时的地面分割
    pcl::PointCloud<PointType>::Ptr ground_cloud(new pcl::PointCloud<PointType>);
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
//                SACMODEL_PLANE, 三维平面
//                SACMODEL_LINE,    三维直线
//                SACMODEL_CIRCLE2D, 二维圆
//                SACMODEL_CIRCLE3D,  三维圆
//                SACMODEL_SPHERE,      球
//                SACMODEL_CYLINDER,    柱
//                SACMODEL_CONE,        锥
//                SACMODEL_TORUS,       环面
//                SACMODEL_PARALLEL_LINE,   平行线
//                SACMODEL_PERPENDICULAR_PLANE, 垂直平面
//                SACMODEL_PARALLEL_LINES,  平行线
//                SACMODEL_NORMAL_PLANE,    法向平面
//                SACMODEL_NORMAL_SPHERE,   法向球
//                SACMODEL_REGISTRATION,
//                SACMODEL_REGISTRATION_2D,
//                SACMODEL_PARALLEL_PLANE,  平行平面
//                SACMODEL_NORMAL_PARALLEL_PLANE,   法向平行平面
//                SACMODEL_STICK
        seg.setMethodType(pcl::SAC_RANSAC);//设置随机采样一致性方法类型
        // you can modify the parameter below
        seg.setMaxIterations(10000);//设置最大迭代次数
        seg.setDistanceThreshold(0.01);//设定距离阀值，距离阀值决定了点被认为是局内点是必须满足的条件
        seg.setInputCloud(cloudPtrv);
        //引发分割实现，存储分割结果到点几何inliers及存储平面模型的系数coefficients
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
            cout << "error! Could not found any inliers!" << endl;
        }
        // extract ground
        // 从点云中抽取分割的处在平面上的点集
        pcl::ExtractIndices<PointType> extractor;//点提取对象
        extractor.setInputCloud(cloudPtrv);
        extractor.setIndices(inliers);
        extractor.setNegative(true);
        extractor.filter(*ground_cloud);
        cout << "filter done." << endl;

        //实时聚类算法
        //为提取点云时使用的搜素对象利用输入点云cloud_filtered创建Kd树对象tree。
        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
        tree->setInputCloud(ground_cloud);//创建点云索引向量，用于存储实际的点云信息
        vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointType> ec;
        ec.setClusterTolerance(0.2); // 设置近邻搜索的搜索半径为2cm
        ec.setMinClusterSize(80);//设置一个聚类需要的最少点数目为100
        ec.setMaxClusterSize(200);//设置一个聚类需要的最大点数目为25000
        ec.setSearchMethod(tree);//设置点云的搜索机制
        ec.setInputCloud(ground_cloud);
        ec.extract(cluster_indices);//从点云中提取聚类，并将点云索引保存在cluster_indices中
        //迭代访问点云索引cluster_indices，直到分割出所有聚类
        pcl::PointCloud<PointType>::Ptr cloud_cluster(new pcl::PointCloud<PointType>);
        int j = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();it != cluster_indices.end(); ++it)
        {

            //创建新的点云数据集cloud_cluster，将所有当前聚类写入到点云数据集中
            for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
                cloud_cluster->points.push_back(ground_cloud->points[*pit]); //*
            cloud_cluster->header=ground_cloud->header;
            cloud_cluster->width=cloud_cluster->points.size();
            cloud_cluster->height=1;
            cloud_cluster->is_dense=true;

            std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points."
                      << std::endl;
            j++;
        }
        sensor_msgs::PointCloud2 output;//声明的输出的点云的格式
        pcl::toROSMsg(*cloud_cluster, output);//第一个参数是输入，后面的是输出
        pub_model.publish(output);
        // pcl::io::savePCDFileASCII("/home/zyq/catkin_ws/src/scene_model_get/save/model_chair*.pcd", *ground_cloud);
  }
}
save_pointcloud::~save_pointcloud()
{

}