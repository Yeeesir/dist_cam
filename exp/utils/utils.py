import numpy as np
import cv2
import torch.nn as nn
import xml.etree.ElementTree as et 
import os


def unnormalized(img):
    me=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = img * std + me     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)
    
def cal_iou(bboxes, gray_heatmap):
    label = np.zeros_like(gray_heatmap)
    for n in range(bboxes.shape[0]):
        label[bboxes[n][2]:bboxes[n][3],bboxes[n][0]:bboxes[n][1]]=1
    bn_heatmap = gray_heatmap.copy()
    bn_heatmap[bn_heatmap<50]=0
    bn_heatmap[bn_heatmap>=50]=1
    
    inter = label*bn_heatmap
    union = label+bn_heatmap
    union[union!=0]=1
    sum_inter = np.sum(inter)
    sum_union = np.sum(union)
    return np.float(sum_inter)/np.float(sum_union)


def show_cam_on_image(img, mask):
    mask = cv2.resize(mask,(img.shape[1],img.shape[0]))
    max_mask = np.max(mask)
    min_mask = np.min(mask)
    mask = (mask-min_mask)/(max_mask-min_mask)

    gray_heatmap = np.uint8(255 * mask)
    # gray_heatmap = np.expand_dims(gray_heatmap,axis=2)
    # gray_heatmap = gray_heatmap.repeat(3,axis=2)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    # bn_heatmap =  img.copy()
    # bn_heatmap[gray_heatmap>150]=255
    
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    # bn_cam = gray_heatmap + np.float32(img)
    # bn_cam = bn_cam / np.max(bn_cam)

    return np.uint8(255 * cam),np.uint8(heatmap),np.uint8(gray_heatmap)



def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # ck = correct[:k]
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(1,-1).float().sum(1)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Backbone(nn.Module):
  def __init__(self, model):
    super(Backbone, self).__init__()
    self.l1 = nn.Sequential(*list(model.children())[:-2])
    
    #self.last = list(model.children())[-1].to('cuda:0')

  def forward(self, x):
    self.a= list(self.l1.children())
    x = self.l1(x)
    x = x.view(x.size()[0], -1)
    #x = self.last(x)
    return x

class Backbone_cam(nn.Module):
  def __init__(self, model):
    super(Backbone_cam, self).__init__()
    self.l1 = nn.Sequential(*list(model.children())[:-2])
    #self.last = list(model.children())[-1].to('cuda:0')

  def forward(self, x):
    x = self.l1(x)
    # x = x.view(x.size()[0], -1)
    #x = self.last(x)
    return x

class BboxReader:
    def __init__(self, ann_path):
        self.ann_path = ann_path
    
    def bbox(self, name):
        ann = self.ann_path+name
        tree = et.parse(ann)
        objs = tree.findall('object')
        bboxes = []
        for o in objs:
            name = o.find('name').text
            bndbox = o.find('bndbox')
            xmin = bndbox.find('xmin').text
            xmax = bndbox.find('xmax').text
            ymin = bndbox.find('ymin').text
            ymax = bndbox.find('ymax').text
            bboxes.append(np.array([xmin,xmax,ymin,ymax],dtype=np.uint32))
        bboxes = np.stack(bboxes,axis=0)
        return bboxes

class LabelReader:
    def __init__(self, ann_path):
        self.ann_path = ann_path
    
    def label(self, name):
        ann = self.ann_path+name
        tree = et.parse(ann)
        objs = tree.findall('object')
        label = objs[0].find('name').text
        return label

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count