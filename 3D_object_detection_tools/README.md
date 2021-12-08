# Some Functions Of Pointnet++ 
" Northeastern University
Action "
Some implementations of training, testing and model deployment using PointNet++

## Install
The latest codes are tested on Ubuntu 18.04, CUDA10.1, PyTorch 1.6 and Python 3.7:
You can use Anaconda to manage you python environment,how to install Anaconda you can search in www.baidu.com. 
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```
Some python packages you should pip install.
```shell

```

## Make your own dataset
You can make your own dataset use make_dataset ros package in "/src/make_dataset" which depend on ros.
Data set requirements can be referenced data/rgbd_dataset.
There are some keypoints(Take 6 classification problem as an example):

1.modelnet6_shape_names.txt. It is used to store the name of the model you want to classify.

2.modelnet6_test.txt.  You can use the first 3 / 4 of the data set as the training set and the last 1 / 4 as the test set
                        This file is used to store the test dataset filename.

3.modelnet6_train.txt. You can use the first 3 / 4 of the data set as the training set and the last 1 / 4 as the test set
                        This file is used to store the train dataset filename.

4.One type of model is stored in a separate folder.(data/rgbd_dataset/box/box_0001.pcd)

## train
You just need to set the number you want to classify.
However, If you want to change the amount of classift，some parameters in file “load_data.py” need to be modified.
```shell
python train_classification.py --num_category 6
```
The weight file obtained from training is saved in /log/checkpoints/best_model.pth .

### Performance
| Model | Accuracy |
|--|--|
| PointNet2 (Official) | 91.9 |
| PointNet (Pytorch without normal) |  90.6|

### test
You can test the performance of your code on the test set, or you can test a single PCD file separately.
If you want to change the amount of classift，some parameters in file “predict.py” need to be modified.
```shell
python test_pcd_rgbd_test.py
```
### transform_model_to_libtorch
Output libtorch model
You can generate libtorch models that can be deployed in C++.
If you want to change the amount of classift，some parameters in file “output_model.py” need to be modified.
```shell
python output_model.py
```
The filename of output model is script_model_1.pt.
You can use it in ROS.
