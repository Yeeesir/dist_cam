# _make_proto means produce prototypes with pretrained models

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import faiss
import argparse
import os
import pickle

# from utils.resnet12 import ResNet12
from utils.dataset import ImageNet_224
import pretrainedmodels
import pretrainedmodels.utils
from utils.utils import show_cam_on_image, Backbone, BboxReader, cal_iou, set_gpu, unnormalized,Backbone_cam


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

    
def make_proto(dataset, backbone, backbone_cam, path_proto):
    #用1000个csv，每个类别生成一个prototype，并保存在pickle中，保存路径为path_proto
    # dataset为整个数据集
    # backbone是有gap的网络，提取的特征或原型用于分类
    # backbone是没有gap的网络，提取的特征或原型用于生成distcam
    
    # proto_dict中 key:1000个类别；value:用于分类的类别原型
    # proto_cam_dict中 key:1000个类别；value:用于生成distcam的类别原型
    proto_dict = {}
    proto_cam_dict = {}
    idx = 0
    for part in dataset:
        # 每个part为一个类别的样本，全部提取特征后，建立类别原型
        idx+=1
        print('the {}th part'.format(idx))
        embeddings, labels = get_all_embeddings(dataset[part], backbone)
        proto = np.mean(embeddings,axis=0)
        del embeddings
        del labels
        embeddings_cam, labels = get_all_embeddings(dataset[part], backbone_cam)
        proto_cam = np.mean(embeddings_cam,axis=0)
        del embeddings_cam
        del labels
        proto_dict[part]=proto
        proto_cam_dict[part]=proto_cam
        # break
    save_proto = {}
    save_proto['proto'] = proto_dict
    save_proto['proto_cam'] = proto_cam_dict

    with open(path_proto,'wb') as fp:
        pickle.dump(save_proto,fp,protocol = pickle.HIGHEST_PROTOCOL)

def main():
    device = torch.device("cuda")
    model_names = sorted(name for name in pretrainedmodels.__dict__
                        if not name.startswith("__")
                        and name.islower()
                        and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser()
    # csv_path1 为imagenet1000个类的csv文件的路径，在交接数据集文件夹中
    parser.add_argument('--csv_path1', default='/home/shenyq/data/ILSVRC2012_train/csv1000/')
    parser.add_argument('--img_path1', default='/home/shenyq/data/ILSVRC2012_train/')
    parser.add_argument('--proto_path', default='/home/shenyq/dist_cam/')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
    parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
    args = parser.parse_args()

    # imagenet太大，因此每个类单独建立csv
    csv_list = os.listdir(args.csv_path1)
    assert len(csv_list)==1000
    #dataset_train中 key:1000个类别,value:利用每个类别的csv文件建立的dataset类
    dataset_train = {}
    for csv in csv_list:
        dataset_train[csv.split('.')[0]] = ImageNet_224(csv_path=args.csv_path1,img_path=args.img_path1,mode=csv.split('.')[0])

    print("=> using pre-trained parameters '{}'".format(args.pretrained))
    # backbone是有gap的网络，提取的特征或原型用于分类
    # backbone是没有gap的网络，提取的特征或原型用于生成distcam
    backbone = pretrainedmodels.__dict__[args.arch+'_backbone'](num_classes=1000,
                                                                pretrained=args.pretrained).to(device)
    backbone_cam = pretrainedmodels.__dict__[args.arch+'_backbone_cam'](num_classes=1000,
                                                                pretrained=args.pretrained).to(device)
    proto_name = 'ImageNet2012_'+args.arch+'_Proto.pickle'
    if not os.path.exists(os.path.join(args.proto_path,'saved_proto')):
        os.mkdir(os.path.join(args.proto_path,'saved_proto'))
    path_proto = os.path.join(args.proto_path,'saved_proto',proto_name)
    make_proto(dataset_train, backbone, backbone_cam, path_proto)
    

if __name__=='__main__':
    main()
        

