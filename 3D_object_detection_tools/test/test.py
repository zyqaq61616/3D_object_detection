"""
Northeastern University
Action
"""
from torch.utils.data import Dataset
from load_data import *

# def abspath
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

if __name__ == "__main__":
    '''
        please input the kind of the file which you want to predict. It's necessary!
    '''
    # "modelnet40" "pcd" "off" "txt"
    data_from = "pcd"
    # create object
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point_cloud")
    point_cloud = o3d.geometry.PointCloud()
    args = parse_args()
    # list of name and color
    obj_list = ['chair', 'stair', 'box', 'seesaw', 'railing', 'shelf']
    point = []
    if data_from == "modelnet40":
        # load ModelNet40 file
        data = ModelNetDataLoader(ROOT_DIR+'/../data/modelnet40_normal_resampled/', args, split='test')
        DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,num_workers=6)
        for data in DataLoader:
            points, target = data
            points, target = points.cuda(), target.cuda()
            points = points.transpose(2, 1)
            pred_class = main(args, points)
            pred_choice = pred_class.data.max(1)[1]
            pred = pred_choice.cpu().detach()
            a = int(pred[0])
            target = target.cpu()
            target = target.numpy()
            b = int(target)
            print("检测到的目标为：{}，实际目标为{}".format(obj_list[a], obj_list[b]))
    elif data_from == "rgbd_dataset":
        # load ModelNet40 file
        data = MyModelNetDataLoader(ROOT_DIR + '/../data/rgbd_dataset/', args, split='test')
        DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,num_workers=6)
        for data in DataLoader:
            points, target = data
            points, target = points.cuda(), target.cuda()
            points = points.transpose(2, 1)
            print(points.size())
            pred_class = main(args, points)
            pred_choice = pred_class.data.max(1)[1]
            pred = pred_choice.cpu().detach()
            a = int(pred[0])
            target = target.cpu()
            target = target.numpy()
            b = int(target)
            print("检测到的目标为：{}，实际目标为{}".format(obj_list[a], obj_list[b]))
    else:
        if data_from == "off":
            # load .off file
            off_point, vis_off = load_off(ROOT_DIR+'/../data/cup1.off')
            point_cloud.points = o3d.utility.Vector3dVector(vis_off)
            point = off_point
        elif data_from == "pcd":
            pcd_point, vis_pcd = load_pcd(ROOT_DIR + '/../data/chair.pcd')
            point_cloud.points = o3d.utility.Vector3dVector(vis_pcd)
            point = pcd_point
        elif data_from == "txt":
            txt_point, vis_txt = load_txt(ROOT_DIR+'/../data/bottle_0001.txt')
            point_cloud.points = o3d.utility.Vector3dVector(vis_txt)
            point = txt_point
            # visualization
        vis.add_geometry(point_cloud)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        # predict
        pred = main(args, point)
        pred = pred.cpu().detach()
        a = int(pred[0])
        print("检测到的目标为：{}".format(obj_list[a]))
        vis.run()

