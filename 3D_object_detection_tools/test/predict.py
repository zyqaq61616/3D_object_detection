"""
Northeastern University
Action
"""
import torch
import torchvision.transforms
from torch import nn
import os
import argparse
import sys
import numpy as np
from pointnet2_cls_won_model import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Pred')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=6, type=int, help='training number of category')
    parser.add_argument('--num_point', type=int, default=1024, help='input Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


# test
def pred(model, loader, vote_num=3, cpu=False):

    # eval
    classifier = model.eval()
    if not cpu:
        # to cuda
        loader = loader.cuda()
    loader = loader.transpose(2, 1)
    vote_pool = torch.zeros(1, 6).cuda()
    # vote
    for _ in range(vote_num):
        pred, _ = classifier(loader)
        print(loader.shape)
        vote_pool += pred
    pred = vote_pool / vote_num
    pred_choice = pred.data.max(1)[1]
    return pred_choice


# 定义main函数
def main(args, npoint):
    # is there a GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # load parameter
    args = parse_args()
    num_class = args.num_category
    # load models
    classifier = get_model(num_class, normal_channel=args.use_normals)
    # whether cpu
    if not args.use_cpu:
        classifier = classifier.cuda()
        npoint = npoint.cuda()
    # load weight file
    checkpoint = torch.load(ROOT_DIR + '/../checkpoints/best_model.pth')
    # import weight
    classifier.load_state_dict(checkpoint['model_state_dict'])
    # no grad
    with torch.no_grad():
        Flag = args.use_cpu
        pred_class = pred(classifier, npoint, vote_num=args.num_votes, cpu=Flag)
        return pred_class


if __name__ == '__main__':
    npoint = []
    args = parse_args()
    pred_class = main(args, npoint)