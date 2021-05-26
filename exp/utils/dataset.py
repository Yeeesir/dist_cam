import os.path as osp
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms



class MiniImageNet(Dataset):

    def __init__(self,csv_path,img_path, mode):
        self.img_path = img_path
        csv_path = osp.join(csv_path, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(img_path, 'images', name)
            # path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def class_num(self):
        return len(self.wnids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

class ImageNet_224(Dataset):

    def __init__(self,csv_path,img_path, mode):
        self.img_path = img_path
        csv_path = osp.join(csv_path, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(img_path, 'images', name)
            # path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def class_num(self):
        return len(self.wnids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

class ImageNet_224_name(Dataset):

    def __init__(self,csv_path,img_path, mode):
        self.img_path = img_path
        csv_path = osp.join(csv_path, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(img_path, 'images', name)
            # path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def class_num(self):
        return len(self.wnids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        name = path.split('/')[-1]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, name


class MiniImageNet_Aug(Dataset):

    def __init__(self, csv_path, img_path, aug_path_list, mode):
        self.img_path = img_path
        csv_path = osp.join(csv_path, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(img_path, 'images', name)
            # path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)
            if mode == 'train' or mode=='train_tr':
                for aug_path in aug_path_list:
                    path = osp.join(aug_path, 'images', name)
                    data.append(path)
                    label.append(lb)

        self.data = data
        self.label = label
        

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def class_num(self):
        return len(self.wnids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

class ImageNet(Dataset):
    def __init__(self, img_path, ann_path, transformer):
        self.img_path = img_path
        f = open(ann_path,'r')
        data = []
        label = []
        for line in f.readlines():
            name, cate = line.split(' ')
            data.append(name)
            cate = int(cate)
            label.append(cate)
        
        self.data = data
        self.label = label
        self.transform = transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        name, label = self.data[i], self.label[i]
        image = self.transform(Image.open(os.path.join(self.img_path,name)).convert('RGB'))
        return image, label, name

class ImageNet2(Dataset):
    def __init__(self, img_path, ann_path, transformer):
        self.img_path = img_path
        f = open(ann_path,'r')
        data = []
        label = []
        for line in f.readlines():
            name, cate = line.split(' ')
            data.append(name)
            cate = int(cate)
            label.append(cate)
        
        self.data = data
        self.label = label
        self.transform = transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        name, label = self.data[i], self.label[i]
        image = self.transform(Image.open(os.path.join(self.img_path,name)).convert('RGB'))
        return image, label, name

if __name__ == '__main__':

    csv_path = '/home/shenyq/data/mini/rgb/'
    img_path = '/home/shenyq/data/mini/rgb/'
    aug_path_list=[]
    aug_path_list.append('/home/shenyq/data/mini/aug/enhance')
    aug_path_list.append('/home/shenyq/data/mini/aug/flip')
    mini = MiniImageNet_Aug(csv_path,img_path,aug_path_list,'train')
    print(1)
