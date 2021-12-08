# 单  位：东北大学ACTION实验室
# 工程师：ZYQ
import os
# 保存模型用的模型文件
from pointnet2_cls_won_model import *
# 定义当前路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
if __name__ == "__main__":
    # 训练好的模型的.pth权重文件
    checkpoint = torch.load(ROOT_DIR + '/../best_model.pth')
    # 要分类的数量，要和训练时的数量一致，默认不使用法线，第二个值不要修改
    classifier = get_model(num_class=6, normal_channel=False)
    # 加载权重文件，不需要修改
    classifier.load_state_dict(checkpoint['model_state_dict'])
    # 不更新权重
    classifier = classifier.eval()
    # 在GPU上训练
    classifier.cuda()
    # 以script方式保存模型为.pt文件
    scripted_gate = torch.jit.script(classifier)
    # .pt模型的保存路径
    scripted_gate.save(ROOT_DIR + "/script_model_1.pt")