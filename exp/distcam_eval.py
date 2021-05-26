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

# from utils.resnet12 import ResNet12
from utils.dataset import ImageNet_224_name, ImageNet2
import pretrainedmodels
import pretrainedmodels.utils
from utils.utils import show_cam_on_image, Backbone, BboxReader,LabelReader, cal_iou, set_gpu, unnormalized,Backbone_cam,accuracy,AverageMeter



### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###

def cal_accuracy(embeddings,embeddings2,labels,label2):
    #用于计算度量学习准确率
    nlist = 100
    assert nlist<embeddings.shape[0]
    d = embeddings.shape[1]
    k=1
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    assert not index.is_trained
    index.train(embeddings2)   
    # 索引训练
    assert index.is_trained
    index.add(embeddings2)
    # 向量添加
    D, I = index.search(embeddings, k)
    cnt = 0
    for idx in range(embeddings.shape[0]):
        for kk in range(k):
            if labels[idx]==label2[I[idx][kk]]:
                cnt+=1
                break
    print(cnt/embeddings.shape[0])
    return cnt/embeddings.shape[0]

def findInd(embeddings,embeddings2):
    nlist = 10
    d = embeddings.shape[1]
    k=5
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    assert not index.is_trained
    index.train(embeddings)   
    # 索引训练
    assert index.is_trained
    index.add(embeddings)
    # 向量添加
    D, I = index.search(embeddings2, k)

    return I


# def cal_distCam_old(proto,emb,proto_pool,emb_pool):
#     proto_pool = proto_pool.squeeze()
#     emb_pool = emb_pool.squeeze()
#     weight = torch.abs(proto_pool-emb_pool)
#     weight = torch.max(weight)-weight
#     weight = weight.unsqueeze(1).unsqueeze(1).repeat(1,7,7)
#     dist = proto-emb
#     dist = weight*dist
#     dist = torch.sum(dist,dim=0).squeeze()
#     dist_cam = dist.detach().numpy()
#     return dist_cam

def cal_distCam(proto,emb,proto_pool,emb_pool):
    # distcam生成函数
    proto_pool = proto_pool.squeeze()
    emb_pool = emb_pool.squeeze()
    weight = torch.abs(proto_pool-emb_pool)
    weight = 1/weight
    weight = weight/torch.max(weight)
    weight = weight.unsqueeze(1).unsqueeze(1).repeat(1,7,7)
    # dist = proto-emb
    # dist = weight*dist
    dist_cam = weight*emb
    dist_cam = torch.sum(dist_cam,dim=0).squeeze()
    dist_cam = dist_cam.detach().numpy()
    return dist_cam

def eval(proto, dataset2, backbone):
    label_p = np.expand_dims(np.array(range(1000)),axis=1)
    embeddings2, labels2 = get_all_embeddings(dataset2,backbone)
    print(embeddings2.shape)
    acc = cal_accuracy(proto,embeddings2,label_p,labels2)
    return acc

    


def main():
    device = torch.device("cuda")

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    batch_size = 256


    model_names = sorted(name for name in pretrainedmodels.__dict__
                        if not name.startswith("__")
                        and name.islower()
                        and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser()
    # parser.add_argument('--csv_path1', default='/home/shenyq/data/ILSVRC2012_train/')
    # parser.add_argument('--img_path1', default='/home/shenyq/data/ILSVRC2012_train/')
    # make_prototype文件生成的prototype所在的路径
    parser.add_argument('--proto_path', default='/home/shenyq/dist_cam/exp/saved_proto/')
    parser.add_argument('--csv_path2', default='/home/shenyq/data/ILSVRC2012_val/')
    parser.add_argument('--img_path2', default='/home/shenyq/data/ILSVRC2012_val/')

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
    
    # imagenet 验证集存放路径
    # val_img_dir 为图片路径
    # val_ann_dir 为txt图像类别文件
    # ann_path 为存放bbox的xml文件路径
    val_img_dir = '/home/shenyq/data/ILSVRC2012_val/images/'
    val_ann_dir = '/home/shenyq/data/ILSVRC2012_val/val.txt'
    ann_path = '/home/shenyq/data/ILSVRC2012_val/bbox/'
    
    
    tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    val_dataset = ImageNet2(val_img_dir,val_ann_dir,transformer=tf)
    #当前版本要把val batchsize设为1
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    # 存放准确率结果的变量
    fc_cls_top1 = AverageMeter() #全连接层分类top1准确率
    fc_cls_top5 = AverageMeter() #全连接层分类top5准确率
    fc_loc_top1 = AverageMeter() #全连接层定位top1准确率
    fc_loc_top5 = AverageMeter() #全连接层定位top5准确率
    metric_cls_top1 = AverageMeter() #度量分类top1准确率
    metric_cls_top5 = AverageMeter() #度量分类top5准确率
    metric_loc_top1 = AverageMeter() #度量定位top1准确率
    metric_loc_top5 = AverageMeter() #度量定位top5准确率

    # 将pretrained的模型中最后的linear层权重保存下来，用于后面生成cam
    (name_m, para) = list(model.children())[-1].parameters()
    weights = name_m.data.cpu()

    #读取验证集的bbox标签和类别标签
    bboxreader = BboxReader(ann_path)
    labelreader = LabelReader(ann_path)
    
    with torch.no_grad():
        model.eval()
        backbone.eval()
        backbone_cam.eval()
        for i, (input, target, name) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()
            img = cv2.imread(val_img_dir+name[0])
            img = cv2.resize(img,(224,224))
            
            # 全连接层分类
            output = model(input)
            # 生成两种embedding，emb是没有经过gap的，emb是经过gap的
            emb = backbone_cam(input)
            emb_pool = backbone(input)
            fc_cls_prec1, fc_cls_prec5 = accuracy(output.data, target.data, topk=(1, 5))
            # fc_cls_prec1.item()是一个float数字
            fc_cls_top1.update(fc_cls_prec1.item(), input.size(0))
            fc_cls_top5.update(fc_cls_prec5.item(), input.size(0))
            
            # 生成cam
            max, argmax = output.data.max(1)
            class_ids = argmax.cpu().numpy()
            out_sort = output.argsort().cpu().numpy()
            class_ids_top5 = out_sort[0,-5:]

            wh_emb = np.int(np.sqrt(emb.size(1)/emb_pool.size(1)))
            emb_reshape = emb.reshape(emb_pool.size(0),emb_pool.size(1),wh_emb,wh_emb)
            emb_reshape = emb_reshape.squeeze().cpu()

            # 读取bbox标注
            bboxes = bboxreader.bbox(name[0].split('.')[0]+'.xml')
            label_str = labelreader.label(name[0].split('.')[0]+'.xml')

            f_l_t1=0
            f_l_t5=0
            # for循环是为了计算top定位准确率
            for kk in range(5):
                idd = class_ids_top5[4-kk]
                weight = weights[idd].squeeze().unsqueeze(1).unsqueeze(1)
                weight_repeat = weight.repeat(1,7,7)
                emb_mul = emb_reshape*weight_repeat
                mask = torch.sum(emb_mul,dim=0).squeeze()
                mask = mask.detach().numpy()
                cam, heatmap,gray_heatmap = show_cam_on_image(img, mask)

                iou = cal_iou(bboxes,gray_heatmap)
                if iou>=0.5 and kk==0:
                    f_l_t1=1
                if iou>=0.5:
                    f_l_t5=1 
            fc_loc_top1.update(f_l_t1, input.size(0))
            fc_loc_top5.update(f_l_t5, input.size(0))
            
            #度量学习分类
            emb_pool = emb_pool.cpu()
            I = findInd(proto_mat.numpy(),emb_pool.numpy())

            m_c_t1=0
            m_c_t5=0
            m_l_t1=0
            m_l_t5=0
            # 计算度量学习top1,top5的分类和定位准确率
            for kk in range(5):
                if I[0][kk]<0 or I[0][kk]>999:
                    continue
                pred = idx2str[I[0][kk]]
                
                proto_pred = proto_mat_cam[I[0][kk]]

                proto_pred = proto_pred+torch.min(proto_pred)
                proto_pred = proto_pred/torch.max(proto_pred)

                emb_reshape = emb_reshape+torch.min(emb_reshape)
                emb_reshape = emb_reshape/torch.max(emb_reshape)

                proto_pred_reshape = proto_pred.reshape(emb_pool.size(0),emb_pool.size(1),wh_emb,wh_emb)
                proto_pred_reshape = proto_pred_reshape.squeeze().cpu()
                
                dist_cam = cal_distCam(proto_pred_reshape,emb_reshape,proto_mat[I[0][kk]],emb_pool)
                cam, heatmap,gray_heatmap = show_cam_on_image(img, dist_cam)


                iou = cal_iou(bboxes,gray_heatmap)
                if pred == label_str and kk==0:
                    m_c_t1=1
                if pred == label_str:
                    m_c_t5=1
                
                if iou>=0.5 and kk==0:
                    m_l_t1=1
                if iou>=0.5:
                    m_l_t5=1 
            metric_cls_top1.update(m_c_t1, input.size(0))
            metric_cls_top5.update(m_c_t5, input.size(0))
            metric_loc_top1.update(m_l_t1, input.size(0))
            metric_loc_top5.update(m_l_t5, input.size(0))

            if i % 10==0:
                print('Test: [{0}/{1}]\t'
                      'FC_Cls_Acc@1 {fc_cls_top1.val:.3f} ({fc_cls_top1.avg:.3f})\t'
                      'FC_Cls_Acc@5 {fc_cls_top5.val:.3f} ({fc_cls_top5.avg:.3f})'.format(
                       i, len(val_loader),fc_cls_top1=fc_cls_top1, fc_cls_top5=fc_cls_top5))
                print('FC_Loc_Acc@1 {fc_loc_top1.val:.3f} ({fc_loc_top1.avg:.3f})\t'
                      'FC_Loc_Acc@5 {fc_loc_top5.val:.3f} ({fc_loc_top5.avg:.3f})'.format(
                       fc_loc_top1=fc_loc_top1, fc_loc_top5=fc_loc_top5))
                print('M_Cls_Acc@1 {metric_cls_top1.val:.3f} ({metric_cls_top1.avg:.3f})\t'
                      'M_Cls_Acc@5 {metric_cls_top5.val:.3f} ({metric_cls_top5.avg:.3f})'.format(
                       metric_cls_top1=metric_cls_top1, metric_cls_top5=metric_cls_top5))
                print('M_Loc_Acc@1 {metric_loc_top1.val:.3f} ({metric_loc_top1.avg:.3f})\t'
                      'M_Loc_Acc@5 {metric_loc_top5.val:.3f} ({metric_loc_top5.avg:.3f})'.format(
                       metric_loc_top1=metric_loc_top1, metric_loc_top5=metric_loc_top5))

if __name__=='__main__':
    main()
        

