import os, pickle, json, csv
import random
from PIL import Image, ImageFilter
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data.randaugment import rand_augment_transform

def get_data(csv_file):
    """Load our data from file."""
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def aug_plus(dataset='ImageNet_LT', mode='train', randaug_n=2, randaug_m=10, transform_type='none'):
    # PaCo's aug: https://github.com/jiequancui/ Parametric-Contrastive-Learning

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]) if dataset == 'inat' \
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augmentation_regular = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)
    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    if transform_type == 'none':
        transform_train = transforms.Compose(augmentation_regular)
    elif transform_type == 'sim_aug':
        transform_train = transforms.Compose(augmentation_sim)
    else:
        transform_train = transforms.Compose(augmentation_randncls)

    if mode == 'train':
        return transform_train
    else:
        return val_transform

class iNaturaList(Dataset):
    def __init__(self, root_path, cfg, is_train=True, transform_type='none'):
        super(iNaturaList, self).__init__()
        self.root_path = root_path
        self.cfg = cfg
        self.is_train = is_train
        self.data_list, self.label_list = self.read_data()
        print('Using data augmentation: ', transform_type)
        if self.is_train:
            mode = 'train'
        else:
            mode = 'val'
        self.transform = aug_plus(dataset='inat', mode=mode, transform_type=transform_type)

    def read_data(self):
        data_list, label_list = [], []
        if self.is_train:
            json_file = os.path.join(self.root_path, 'train2018.json')
        else:
            json_file = os.path.join(self.root_path, 'val2018.json')
        f = open(json_file, 'r')
        json_data = json.load(f)
        json_data = json_data['images']
        for row in json_data:
            img_path = os.path.join(self.root_path, row['file_name'])
            img_label = int(row['file_name'].split('/')[-2])
            data_list.append(img_path)
            label_list.append(img_label)
        return data_list, label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        img_label = self.label_list[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        try:
            img = self.transform(img)
        except:
            print('idx has error: ', idx)
            print(img.shape)
        img_label = torch.tensor(img_label, dtype=torch.float32)
        return img, img_label


def get_iNaturaList(cfg):
    datasets = {}
    datasets['train'] = iNaturaList(root_path='./datasets', cfg=cfg, is_train=True, transform_type=cfg.data_aug)
    datasets['test'] = iNaturaList(root_path='./datasets', cfg=cfg, is_train=False)
    print('#Examples in train-set:', len(datasets['train']))
    print('#Examples in test-set:', len(datasets['test']))
    new_labelList = datasets['train'].label_list
    dataloaders = {set_name: DataLoader(datasets[set_name],
                                        batch_size=cfg.batch_size,
                                        shuffle=set_name == 'train',
                                        num_workers=16,  # num_work can be set to batch_size
                                        pin_memory=True)
                   for set_name in ['train', 'test']}

    print('#train batch:', len(dataloaders['train']), '\t#test batch:', len(dataloaders['test']))
    return dataloaders, new_labelList


class ImageNet_LT(Dataset):
    def __init__(self, root_path, cfg, is_train=True, transform_type='none'):
        super(ImageNet_LT, self).__init__()
        self.root_path = root_path
        self.cfg = cfg
        self.is_train = is_train
        self.data_list, self.label_list = self.read_data()
        self.transform_type = transform_type
        print('Using data augmentation: ', self.transform_type)
        if self.is_train:
            mode = 'train'
        else:
            mode = 'val'
        self.transform = aug_plus(dataset='ImageNet_LT', mode=mode, transform_type=self.transform_type)

    def read_data(self):
        data_list, label_list = [], []
        if self.is_train:
            csv_file = os.path.join(self.root_path, 'train.csv')
        else:
            csv_file = os.path.join(self.root_path, 'val.csv')

        full_data = get_data(csv_file)
        for row in full_data:
            data_list.append(os.path.join(self.root_path, row[0], row[1], row[2]))
            label_list.append(int(row[1]))
        return data_list, label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        img_label = self.label_list[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        try:
            img = self.transform(img)
        except:
            print('idx has error: ', idx)
            print(img.shape)
        img_label = torch.tensor(img_label, dtype=torch.float32)
        return img, img_label

def get_ImageNetLT(cfg):
    datasets = {}
    datasets['train'] = ImageNet_LT(root_path='./datasets/ImageNet-LT', cfg=cfg, is_train=True, transform_type=cfg.data_aug)
    datasets['test'] = ImageNet_LT(root_path='./datasets/ImageNet-LT', cfg=cfg, is_train=False)
    print('#Examples in train-set:', len(datasets['train']))
    print('#Examples in test-set:', len(datasets['test']))
    new_labelList = datasets['train'].label_list
    dataloaders = {set_name: DataLoader(datasets[set_name],
                                        batch_size=cfg.batch_size,
                                        shuffle=set_name == 'train',
                                        num_workers=16,  # num_work can be set to batch_size
                                        pin_memory=True)
                   for set_name in ['train', 'test']}

    print('#train batch:', len(dataloaders['train']), '\t#test batch:', len(dataloaders['test']))
    return dataloaders, new_labelList