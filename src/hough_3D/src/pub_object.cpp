// Created by zyq on 2021/9/7.
//获取场景和模型的点云函数的实现
#include <pcl_conversions/pcl_conversions.h>
#include"hough_3D/hough_example.h"
//构造函数 参数初始化 初始ros节点
object_detection::object_detection(ros::NodeHandle &n)
{
    model_ok=false;
    scene_ok=false;
    show_keypoints= true;
    show_correspondences= true;
    use_cloud_resolution=false;
    model_ss=0.03;
    scene_ss=0.03;
    descr_rad=0.03;
    rf_rad=0.035;
    cg_size=0.04;
    cg_thresh=3.0f;
//    pub_result = n.advertise<sensor_msgs::PointCloud2>("result", 1000);                                     //发布匹配后的图像
//    pub_coordinate = n.advertise<geometry_msgs::Vector3>("6D_coordinates", 1000);                           //发布6D位姿
//    sub_model=n.subscribe("/camera/depth_registered/points", 1, &object_detection::model_file_cb, this);//接收模型点云
//    sub_scene=n.subscribe("/camera/depth_registered/points", 1, &object_detection::scene_file_cb, this);//接收场景点云
}

//当读取文件成功才会执行后续代码
bool object_detection::ready(bool model,bool sence)
{
if(model==sence==true)
{
   cout<<"model and sence is ready !"<<endl;
   return true;
}
else
{
    cout<<"Can't get point cloud! "<<endl;
    return false;
}
}
//获取法线函数(法线没问题，数据集有问题，需要不采样的原始点云数据)
pcl::PointCloud<NormalType>::Ptr object_detection::get_normals(const pcl::PointCloud<PointType>::Ptr &msg)
{
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    pcl::PointCloud<NormalType>::Ptr normals (new pcl::PointCloud<NormalType> ());
    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setNumberOfThreads(10);
    norm_est.setInputCloud (msg);
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(10);
    norm_est.compute (*normals);

////    //可视化
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normal viewer"));
//    viewer->setBackgroundColor(0.3,0.3,0.3);
//    viewer->addText("faxian",10,10,"text");
//    pcl::visualization::PointCloudColorHandlerCustom<PointType> single_color(msg,0,255,0);
//    viewer->addCoordinateSystem(0.1);
//    viewer->addPointCloud<PointType>(msg,single_color,"sample cloud");
//    viewer->addPointCloudNormals<PointType,pcl::Normal>(msg,normals,20,0.02,"normals");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2,"sample cloud");
//    while(!viewer->wasStopped())
//    {
//        viewer->spinOnce(100);
//        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//    }
    return normals;
}
//用体素网格进行下采样
pcl::PointCloud<PointType>::Ptr object_detection::voxel_downsample(
        const pcl::PointCloud<PointType>::Ptr &msg) {
    pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType> ());
    pcl::VoxelGrid<PointType> sor;
    sor.setInputCloud (msg);
    sor.setLeafSize (0.02f, 0.02f, 0.02f);
    sor.filter (*cloud_filtered);
    cout << "PointCloud after VoxelGrid filtering: " << *cloud_filtered<<endl;
    return cloud_filtered;
}
//点云下采样输出关键点个数 注意参数调节
pcl::PointCloud<PointType>::Ptr object_detection::get_downsample(const pcl::PointCloud<PointType>::Ptr &msg,double index)
{
    pcl::PointCloud<PointType>::Ptr keypoints (new pcl::PointCloud<PointType> ());
    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud (msg);
    uniform_sampling.setRadiusSearch (index);
    uniform_sampling.filter (*keypoints);
    std::cout << "total points: " << msg->size () << "; Selected Keypoints: " << keypoints->size () << std::endl;
    return keypoints;
}
//点云去除离群点
pcl::PointCloud<PointType>::Ptr object_detection::move_outliner_sample(const pcl::PointCloud<PointType>::Ptr &msg) {
    //离群点去除
    pcl::RadiusOutlierRemoval<PointType> pcFilter;  //创建滤波器对象
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    pcFilter.setInputCloud(msg);             //设置待滤波的点云
    pcFilter.setRadiusSearch(0.1);               // 设置搜索半径
    pcFilter.setMinNeighborsInRadius(20);      // 设置一个内点最少的邻居数目
    pcFilter.filter(*cloud_filtered);        //滤波结果存储到cloud_filtered
    return cloud_filtered;
}
//iss 关键点
pcl::PointCloud<PointType>::Ptr object_detection::iss_keypoint(const pcl::PointCloud<PointType>::Ptr &msg) {
    pcl::PointCloud<PointType>::Ptr keypoints (new pcl::PointCloud<PointType> ());
    pcl::ISSKeypoint3D<PointType,PointType> iss_det;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    iss_det.setInputCloud(msg);
    iss_det.setSearchMethod(tree);
    iss_det.setNumberOfThreads(4);
    iss_det.setSalientRadius(0.03f);//设置用于计算协方差矩阵的球邻域半径
    iss_det.setNonMaxRadius(0.01);//设置非极大值抑制应用算法半径
    iss_det.setThreshold21(0.65);//设置第二个和第一个特征值之比的上限
    iss_det.setThreshold32(0.5);//设置第三个和第二个特征值之比的上限
    iss_det.setMinNeighbors(5);//在应用非极大值抑制算法时，设置必须找到的最小邻居数
    iss_det.compute(*keypoints);
    //可视化部分
    pcl::visualization::PCLVisualizer viewer("iss_keypoints");
    viewer.setWindowName("iss关键点检测");
    viewer.setBackgroundColor(255,255,255);
    viewer.addPointCloud(msg,"cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,0,1,0,"cloud");
    viewer.addPointCloud(keypoints,"keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,5,"keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,1,0,0,"keypoints");
    while(!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
    //可视化部分结束
    return keypoints;

}
//sift关键点提取
pcl::PointCloud<PointType>::Ptr object_detection::sift_keypoint(const pcl::PointCloud<PointType>::Ptr &msg) {
    pcl::PointCloud<PointType>::Ptr keypoints (new pcl::PointCloud<PointType> ());
    pcl::SIFTKeypoint<PointType,pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    sift.setInputCloud(msg);
    sift.setSearchMethod(tree);
    sift.setScales(0.0005,5,8);//设置尺度空间中最小尺度的标准偏差  设置尺度空间层数，越小特征点越多 设置制度空间中计算的尺度个数
    sift.setMinimumContrast(0.0005);//设置限制关键点检测的阈值
    sift.compute(result);
    copyPointCloud(result,*keypoints);
//    //可视化部分
//    pcl::visualization::PCLVisualizer viewer("sift_keypoints");
//    viewer.setWindowName("SIFT关键点检测");
//    viewer.setBackgroundColor(0,0,0);
//    viewer.addPointCloud(msg,"cloud");
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,0,1,0,"cloud");
//    viewer.addPointCloud(keypoints,"keypoints");
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"keypoints");
//    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,1,0,0,"keypoints");
//    while(!viewer.wasStopped())
//    {
//        viewer.spinOnce();
//    }
//    //可视化部分结束
    return keypoints;
}
//sift3D

//关键点shot特征提取
pcl::PointCloud<DescriptorType>::Ptr object_detection::shot_features(const pcl::PointCloud<NormalType>::Ptr &normal,
                                                                      const pcl::PointCloud<PointType>::Ptr &keypoint,
                                                                      const pcl::PointCloud<PointType>::Ptr &surface,double index){
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
    pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    descr_est.setRadiusSearch (index);
    descr_est.setInputCloud (keypoint);
    descr_est.setInputNormals (normal);
    descr_est.setSearchSurface (surface);
    descr_est.compute (*model_descriptors);
    return model_descriptors;

}

//特征点比对
pcl::CorrespondencesPtr object_detection::Model_Scene_Correspondences(const pcl::PointCloud<DescriptorType>::Ptr &scene_shot_keypoints,
                                                                      const pcl::PointCloud<DescriptorType>::Ptr &model_shot_keypoints) {
    //使用Kdtree找出 Model-Scene 匹配点
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
    pcl::KdTreeFLANN<DescriptorType> match_search;//设置配准方式
    match_search.setInputCloud (model_shot_keypoints);//模型点云的描述子

    //  一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中(这块有点问题可以更新 不一定百分百找到)
    for (std::size_t i = 0; i < scene_shot_keypoints->size (); ++i)
    {
        std::vector<int> neigh_indices (1);//设置最近邻点的索引
        std::vector<float> neigh_sqr_dists (1);//设置最近邻平方距离值
        if (!std::isfinite (scene_shot_keypoints->at(i).descriptor[0]))  //忽略 NaNs点
        {
            continue;
        }
        int found_neighs = match_search.nearestKSearch (scene_shot_keypoints->at (i), 1, neigh_indices, neigh_sqr_dists);
        if(found_neighs == 1 && neigh_sqr_dists[0] < 0.35f) //  //仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配（默认为0.25）
        {
            //neigh_indices[0]给定点，i是配准数neigh_sqr_dists[0]与临近点的平方距离
            pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
            model_scene_corrs->push_back (corr);//把配准的点存储在容器中
        }
    }
    std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
    //最终得到的是一个容器（pcl::CorrespondencesPtr）model_scene_corrs
    return model_scene_corrs;
}
//打印旋转矩阵和平移向量函数
void object_detection::print_result(
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations,std::vector<pcl::Correspondences> clustered_corrs) {
//    std::cout << "Model instances found: " << rototranslations.size () << std::endl;
//   int index=this->only_print_the_most_likely(clustered_corrs);
//        std::cout << "\n    Instance " << index + 1 << ":" << std::endl;
//        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[index].size () << std::endl;
//
//        // Print the rotation matrix and translation vector
//        Eigen::Matrix3f rotation = rototranslations[index].block<3,3>(0, 0);
//        Eigen::Vector3f translation = rototranslations[index].block<3,1>(0, 3);
//
//        printf ("\n");
//        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
//        printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
//        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
//        printf ("\n");
//        printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "Model instances found: " << rototranslations.size () << std::endl;
    for (std::size_t i = 0; i < rototranslations.size (); ++i)
    {
        std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

        // Print the rotation matrix and translation vector
        Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
        Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

        printf ("\n");
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
        printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
        printf ("\n");
        printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
    }
}

//筛选最像的函数
int object_detection::only_print_the_most_likely(const std::vector<pcl::Correspondences>& clustered_corrs) {
    int maxindex=0;
    int temp=clustered_corrs[0].size();
 for(std::size_t i = 0; i <clustered_corrs.size (); ++i)
 {
    if(temp<clustered_corrs[i].size ())
    {
        temp=clustered_corrs[i].size ();
        maxindex=i;
    }

 }
 return maxindex;
}
////用rviz显示
void object_detection::pre_visualization(const pcl::PointCloud<PointType>::Ptr &model,
                                         const pcl::PointCloud<PointType>::Ptr &model_keypoints,
                                         const pcl::PointCloud<PointType>::Ptr &scene,
                                         const pcl::PointCloud<PointType>::Ptr &scene_keypoints,
                                         std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations,
                                         std::vector<pcl::Correspondences> clustered_corrs) {
//    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
//    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());
//    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
//    viewer.addPointCloud (scene, "scene_cloud");
//    if (show_correspondences || show_keypoints)
//    {
//        //  We are translating the model so that it doesn't end in the middle of the scene representation
//        //对场景点云进行平移和旋转
//        pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
//        pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
////每个点会变为给定的RGB值
//        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
//        viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
//    }
//    if (show_keypoints)
//    {   //设置场景点云颜色
//        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
//        viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
//        //设置点云类型
//        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");
//        //设置点云颜色和点的size
//        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
//        viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
//        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
//    }
//        int index=this->only_print_the_most_likely(clustered_corrs);
//         pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
//        pcl::transformPointCloud (*model, *rotated_model, rototranslations[index]);
//
//        std::stringstream ss_cloud;
//        ss_cloud << "instance" << index;
//
//        pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
//        viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
//
//        if (show_correspondences)
//        {
//            for (std::size_t j = 0; j < clustered_corrs[index].size (); ++j)
//            {
//                std::stringstream ss_line;
//                ss_line << "correspondence_line" << index << "_" << j;
//                PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[index][j].index_query);
//                PointType& scene_point = scene_keypoints->at (clustered_corrs[index][j].index_match);
//
//                //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
//                viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
//            }
//        }
//    while (!viewer.wasStopped ())
//    {
//        viewer.spin ();
//    }
    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
    viewer.addPointCloud (scene, "scene_cloud");

    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

    if (show_correspondences || show_keypoints)
    {
        //  We are translating the model so that it doesn't end in the middle of the scene representation
        pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
        pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
        viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
    }

    if (show_keypoints)
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
        viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
        viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
    }

    for (std::size_t i = 0; i < rototranslations.size (); ++i)
    {
        pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
        pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

        std::stringstream ss_cloud;
        ss_cloud << "instance" << i;

        pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
        viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

        if (show_correspondences)
        {
            for (std::size_t j = 0; j < clustered_corrs[i].size (); ++j)
            {
                std::stringstream ss_line;
                ss_line << "correspondence_line" << i << "_" << j;
                PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
                PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

                //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
                viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
            }
        }
    }

    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }

}
//基于霍夫投票
void object_detection::hough_3d_object_detection(const pcl::PointCloud<PointType>::Ptr &model_keypoints,
                                                 const pcl::PointCloud<NormalType>::Ptr &model_normals,
                                                 const pcl::PointCloud<PointType>::Ptr &model,
                                                 const pcl::PointCloud<PointType>::Ptr &scene_keypoints,
                                                 const pcl::PointCloud<NormalType>::Ptr &scene_normals,
                                                 const pcl::PointCloud<PointType>::Ptr &scene,
                                                 const pcl::CorrespondencesPtr& model_scene_corrs
) {
    //估计器显式计算 LRF 集
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;
    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (this->rf_rad);

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);//模型的LRF集

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);
    //聚类
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (this->cg_size);
    clusterer.setHoughThreshold (this->cg_thresh);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);
    this->print_result(rototranslations,clustered_corrs);//打印输出结果
    this->pre_visualization(model,model_keypoints,scene,scene_keypoints,rototranslations,clustered_corrs);//可视化结果
}
//获取场景点云
//void object_detection::scene_file_cb(const sensor_msgs::PointCloud2 &scene_cloud_msg) {
//    //转换数据类型
//    pcl::PointCloud<PointType>::Ptr scene_cloud(new pcl::PointCloud<PointType>);
//    pcl::fromROSMsg(scene_cloud_msg, *scene_cloud);
//    //预处理
//    pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
//    pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
//    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
//    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
//    scene_normals=this->get_normals(scene_cloud);//获取场景法线
//    scene_keypoints=this->get_downsample(scene_cloud,this->scene_ss);//获取下采样点
//    pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());;//获取shot特征
//    scene_descriptors=this->shot_keypoints(scene_normals,scene_keypoints,scene,descr_rad);//场景的shot特征
//
//}
void object_detection::one_picture(const pcl::PointCloud<PointType>::Ptr &scene_cloud,
                                   const pcl::PointCloud<PointType>::Ptr &model_cloud) {
    //预处理
    pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
    pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
    pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr scene_outliner (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr model_outliner (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr model_down (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr scene_down (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());;//获取shot特征
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());;//获取shot特征
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
    std::vector<pcl::Correspondences> clustered_corrs;
    double startTime1=clock();
    model_outliner=this->voxel_downsample(model_cloud);//下采样
    scene_outliner=this->voxel_downsample(scene_cloud);//下采样
    scene_down=this->move_outliner_sample(scene_outliner);
    model_down=this->move_outliner_sample(model_outliner);
    double endTime1=clock();
    cout<<"下采样需要的时间为："<<(double)(endTime1-startTime1)/CLOCKS_PER_SEC<<"s."<<endl;
    scene_normals=this->get_normals(scene_down);//获取场景法线
    model_normals=this->get_normals(model_down);//获取场景法线
    double endTime2=clock();
    cout<<"获取法线需要的时间为："<<(double)(endTime2-endTime1)/CLOCKS_PER_SEC<<"s."<<endl;
//    scene_keypoints=this->iss_keypoint(scene_down);
//    model_keypoints=this->iss_keypoint(model_down);
    scene_keypoints=this->sift_keypoint(scene_down);
    model_keypoints=this->sift_keypoint(model_down);
    double endTime3=clock();
    cout<<"提取关键点需要的时间为："<<(double)(endTime3-endTime2)/CLOCKS_PER_SEC<<"s."<<endl;
//    scene_keypoints=this->get_downsample(scene_cloud,this->scene_ss);//获取下采样点
//    model_keypoints=this->get_downsample(model_cloud,this->model_ss);//获取下采样点
    model_descriptors=this->shot_features(model_normals,model_keypoints,model_down,descr_rad);//模型的shot特征
    scene_descriptors=this->shot_features(scene_normals,scene_keypoints,scene_down,descr_rad);//场景的shot特征
    double endTime4=clock();
    cout<<"提取特征需要的时间为："<<(double)(endTime4-endTime3)/CLOCKS_PER_SEC<<"s."<<endl;
    model_scene_corrs=Model_Scene_Correspondences(scene_descriptors,model_descriptors);//得到比对后的对应关系
    double endTime5=clock();
    cout<<"匹配相关点需要的时间为："<<(double)(endTime5-endTime4)/CLOCKS_PER_SEC<<"s."<<endl;
    hough_3d_object_detection(model_keypoints,model_normals,model_down,scene_keypoints,scene_normals,scene_down,model_scene_corrs);//用霍夫
    double endTime6=clock();
    cout<<"投票和获取旋转矩阵需要的时间为："<<(double)(endTime6-endTime5)/CLOCKS_PER_SEC<<"s."<<endl;

}
pcl::PointCloud<PointType>::Ptr object_detection::read_scene_file()
{
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType>("/home/zyq/catkin_ws/src/hough_3D/save/scene_bag*.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file rabbit.pcd\n");
        this->scene_ok=false;
    }
    std::cout << "Loaded:" << cloud->width*cloud->height<<"data points from scene.pcd with the following fields:"<< std::endl;
    this->scene_ok=true;
    return cloud;
}
pcl::PointCloud<PointType>::Ptr object_detection::read_model_file()
{
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType>("/home/zyq/catkin_ws/src/hough_3D/save/model_bag*.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file rabbit.pcd\n");
        this->model_ok=false;
    }
    std::cout << "Loaded:" << cloud->width*cloud->height<<"data points from model.pcd with the following fields:"<< std::endl;
    this->model_ok=true;
    return cloud;
}
//析构函数
object_detection::~object_detection()
{}

