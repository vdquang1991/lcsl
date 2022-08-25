import os, pickle
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from data.autoaug import CIFAR10Policy, Cutout


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, train=True,
                 transform=None, target_transform=None, download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.root = root
        self.transform = transform
        if train:
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(self.img_num_list)
        self.current_set_len = len(self.targets)
        self.labelnames = self.get_label_names()

    def __len__(self):
        return self.current_set_len

    def get_label_names(self):
        if self.base_folder == 'cifar-100-python':
            setname = 'meta'
        else:
            setname = 'batches.meta'
        with open(os.path.join(self.root, self.base_folder, setname), 'rb') as obj:
            labelnames = pickle.load(obj, encoding='bytes')
            if setname == 'meta':
                labelnames = labelnames[b'fine_label_names']
            else:
                labelnames = labelnames[b'label_names']
        for i in range(len(labelnames)):
            labelnames[i] = labelnames[i].decode("utf-8")
        return labelnames

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, idx):
        curImage = self.data[idx]
        curLabel = np.asarray(self.targets[idx])
        curImage = Image.fromarray(curImage)
        curImage = self.transform(curImage)
        curLabel = torch.from_numpy(curLabel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return curImage, curLabel

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

def get_cifar100(cfg):
    augmentation_autoaug = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         CIFAR10Policy(),  # add AutoAug
         transforms.ToTensor(),
         Cutout(n_holes=1, length=16),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])

    augmentation_regular = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    if cfg.data_aug == 'none' or cfg.data_aug == 'None' or cfg.data_aug == '':
        train_transform = augmentation_regular
        print('Using regular transform')
    else:
        train_transform = augmentation_autoaug
        print('Using Auto-Augmentation transform')

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    datasets = {}
    datasets['train'] = IMBALANCECIFAR100(root='./datasets', train=True, download=True, transform=train_transform,
                                          imb_type=cfg.imb_type, imb_factor=cfg.imb_factor)
    datasets['test'] = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=val_transform)
    print('#Examples in train-set:', len(datasets['train']))
    print('#Examples in test-set:', len(datasets['test']))

    img_num_per_cls = datasets['train'].img_num_list
    new_labelList = datasets['train'].targets
    labelnames = datasets['train'].labelnames

    dataloaders = {set_name: DataLoader(datasets[set_name],
                                        batch_size=cfg.batch_size,
                                        shuffle=set_name == 'train',
                                        num_workers=8)  # num_work can be set to batch_size
                   for set_name in ['train', 'test']}

    print('#train batch:', len(dataloaders['train']), '\t#test batch:', len(dataloaders['test']))
    return dataloaders, img_num_per_cls, new_labelList, labelnames

def get_cifar10(cfg):
    augmentation_autoaug = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         CIFAR10Policy(),  # add AutoAug
         transforms.ToTensor(),
         Cutout(n_holes=1, length=16),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])

    augmentation_regular = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    if cfg.data_aug == 'none' or cfg.data_aug == 'None' or cfg.data_aug == '':
        train_transform = augmentation_regular
    else:
        train_transform = augmentation_autoaug # Using auto_aug for training data

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    datasets = {}
    datasets['train'] = IMBALANCECIFAR10(root='./datasets', train=True, download=True, transform=train_transform,
                                          imb_type=cfg.imb_type, imb_factor=cfg.imb_factor)
    datasets['test'] = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=val_transform)
    print('#Examples in train-set:', len(datasets['train']))
    print('#Examples in test-set:', len(datasets['test']))

    img_num_per_cls = datasets['train'].img_num_list
    new_labelList = datasets['train'].targets
    labelnames = datasets['train'].labelnames

    dataloaders = {set_name: DataLoader(datasets[set_name],
                                        batch_size=cfg.batch_size,
                                        shuffle=set_name == 'train',
                                        num_workers=8)  # num_work can be set to batch_size
                   for set_name in ['train', 'test']}

    print('#train batch:', len(dataloaders['train']), '\t#test batch:', len(dataloaders['test']))
    return dataloaders, img_num_per_cls, new_labelList, labelnames


