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
import cv2
from PIL import Image
import pickle

from utils.dataset import ImageNet_224_name, ImageNet2
import pretrainedmodels
import pretrainedmodels.utils
from utils.utils import show_cam_on_image, Backbone, BboxReader,LabelReader, cal_iou, set_gpu, unnormalized,Backbone_cam,accuracy,AverageMeter



def cal_distCam(proto,emb,proto_pool,emb_pool):
    # distcam生成函数
    emb_pool = emb_pool.cpu()
    proto_pool =proto_pool.cpu()
    proto = proto.cpu()
    emb = emb.cpu()
    proto_pool = proto_pool.squeeze()
    emb_pool = emb_pool.squeeze()
    # 权重向量weight为距离向量distance的倒数
    distance = torch.abs(proto_pool-emb_pool)
    weight = 1/(distance+0.00001)
    weight = weight.unsqueeze(1).unsqueeze(1).repeat(1,7,7)
    # 用权重向量对特征图的不同通道进行加权
    dist = weight*emb
    dist = torch.sum(dist,dim=0).squeeze()
    dist_cam = dist.detach().numpy()
    return dist_cam


def main():
    device = torch.device("cuda")

    model_names = sorted(name for name in pretrainedmodels.__dict__
                        if not name.startswith("__")
                        and name.islower()
                        and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser()
    # parser.add_argument('--csv_path1', default='/home/shenyq/data/ILSVRC2012_train/')
    # parser.add_argument('--img_path1', default='/home/shenyq/data/ILSVRC2012_train/')
    # 预先存储的prototype向量
    parser.add_argument('--proto_path', default='/home/shenyq/metric/exp_cam/save_proto_v2/')
    parser.add_argument('--csv_path2', default='/home/shenyq/data/ILSVRC2012_val/')
    parser.add_argument('--img_path2', default='/home/shenyq/data/ILSVRC2012_val/')
    # 选择的backbone类型
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
    parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
    args = parser.parse_args()

    # (1)加载预先存储的imagenet1000类的prototype
    proto_name = 'ImageNet2012_'+args.arch+'_Proto.pickle'
    with open(args.proto_path+proto_name,'rb') as fp:
        proto = pickle.load(fp)
    
    proto_mat= torch.zeros(1000,proto['proto']['n01440764'].shape[0])
    proto_mat_cam = torch.zeros(1000,proto['proto_cam']['n01440764'].shape[0])
    idx2str = {}
    idx_cls=0
    for key in proto['proto']:
        proto_mat[idx_cls]=torch.from_numpy(proto['proto'][key])
        proto_mat_cam[idx_cls]=torch.from_numpy(proto['proto_cam'][key])
        idx2str[idx_cls]=key
        idx_cls+=1
    # (1)

    # (2) 加载预训练的模型，并生成backbone和backbone_cam，二者的区别是前者有gap用于分类，后者没有gap用于生成distcam
    print("=> using pre-trained parameters '{}'".format(args.pretrained))
    model = pretrainedmodels.__dict__[args.arch](num_classes=1000,
                                                    pretrained=args.pretrained).to(device)
    
    backbone = pretrainedmodels.__dict__[args.arch+'_backbone'](num_classes=1000,
                                                                pretrained=args.pretrained).to(device)
    backbone_cam = pretrainedmodels.__dict__[args.arch+'_backbone_cam'](num_classes=1000,
                                                                pretrained=args.pretrained).to(device)
    # (2)
    
    tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

    # distcam保存路径
    path_cam_out = '/home/shenyq/dist_cam/exp/result/'
    # 输入图像路径
    path_img = '/home/shenyq/dist_cam/exp/result/'

    # 这里以1-shot任务的distcam生成为例，因此只用一个spt样本生成prototype
    # 如果是用整个imagenet数据集生成的prototype，可以用make_prototype文件生成该backbone下的prototype
    # 如果是多shot任务，可在此处读取多个spt样本，然后在得到emb_pool_spt和emb_spt计算所有样本的均值为prototype
    name_spt = path_img + 'img_Olive_Sided_Flycatcher_0075_30712.jpg'
    name_qry = path_img + 'img_Orange_Crowned_Warbler_0097_168004.jpg'

    img_spt = tf(Image.open(name_spt).convert('RGB'))
    img_spt = img_spt.to(device)
    img_qry = tf(Image.open(name_qry).convert('RGB'))
    img_qry = img_qry.to(device)

    # np形式的图像用于生成distcam的可视化结果
    img_np = cv2.imread(name_qry)   


    with torch.no_grad():
        model.eval()
        backbone.eval()
        backbone_cam.eval()
        # 用backbone对输入图像
        img_spt = img_spt.unsqueeze(0)
        emb_spt = backbone_cam(img_spt)
        emb_pool_spt = backbone(img_spt)

        img_qry = img_qry.unsqueeze(0)
        emb_qry = backbone_cam(img_qry)
        emb_pool_qry = backbone(img_qry)

        wh_emb = np.int(np.sqrt(emb_spt.size(1)/emb_pool_spt.size(1)))
        
        # emb_re_spt进行reshape和归一化
        emb_re_spt = emb_spt.reshape(emb_pool_spt.size(0),emb_pool_spt.size(1),wh_emb,wh_emb)
        emb_re_spt = emb_re_spt.squeeze().cpu()
        emb_re_spt = emb_re_spt+torch.min(emb_re_spt)
        emb_re_spt = emb_re_spt/torch.max(emb_re_spt)

        # emb_re_qry进行reshape和归一化
        emb_re_qry = emb_qry.reshape(emb_pool_spt.size(0),emb_pool_spt.size(1),wh_emb,wh_emb)
        emb_re_qry = emb_re_qry.squeeze().cpu()
        emb_re_qry = emb_re_qry+torch.min(emb_re_qry)
        emb_re_qry = emb_re_qry/torch.max(emb_re_qry)

        # 生成dist_cam
        dist_cam = cal_distCam(emb_re_spt,emb_re_qry,emb_pool_spt,emb_pool_qry)
        cam, heatmap,gray_heatmap = show_cam_on_image(img_np, dist_cam)
        # 保存生成的distcam，cam为在原图上的可视化形式，heatmap为没有原图的热图形式
        cv2.imwrite(path_cam_out+'distcam_vis_'+name_qry.split('/')[-1],cam)
        cv2.imwrite(path_cam_out+'distcam_heat_'+name_qry.split('/')[-1],heatmap)

if __name__=='__main__':
    main()
        

