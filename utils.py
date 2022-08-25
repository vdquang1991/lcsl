import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def createMontage(imList, dims, times2rot90=0):
    '''
    imList isi N x HxWx3
    making a montage function to assemble a set of images as a single image to display
    '''
    imy, imx, k = dims
    rows = round(math.sqrt(k))
    cols = math.ceil(k / rows)
    imMontage = np.zeros((imy * rows, imx * cols, 3))
    idx = 0

    y = 0
    x = 0
    for idx in range(k):
        imMontage[y * imy:(y + 1) * imy, x * imx:(x + 1) * imx, :] = imList[idx, :, :,
                                                                     :]  # np.rot90(imList[:,:,idx],times2rot90)
        if (x + 1) * imx >= imMontage.shape[1]:
            x = 0
            y += 1
        else:
            x += 1
    return imMontage

def horizontal_flip_aug(model):
    def aug_model(data):
        logits = model(data)
        h_logits = model(data.flip(3))
        return (logits+h_logits)/2
    return aug_model

def print_accuracy(model, dataloaders, new_labelList, device='cpu', test_aug=True):
    model.eval()

    if test_aug:
        model = horizontal_flip_aug(model)

    predList = np.array([])
    grndList = np.array([])
    for sample in dataloaders['test']:
        with torch.no_grad():
            images, labels = sample
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            logits = model(images)
            softmaxScores = F.softmax(logits, dim=1)

            predLabels = softmaxScores.argmax(dim=1).detach().squeeze().cpu().numpy()
            predList = np.concatenate((predList, predLabels))
            grndList = np.concatenate((grndList, labels))

    confMat = confusion_matrix(grndList, predList)

    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1, 1))
    confMat = confMat / a

    acc_avgClass = 0
    for i in range(confMat.shape[0]):
        acc_avgClass += confMat[i, i]

    acc_avgClass /= confMat.shape[0]
    print('acc avgClass: ', "{:.1%}".format(acc_avgClass))

    breakdownResults = shot_acc(predList, grndList, np.array(new_labelList), many_shot_thr=100, low_shot_thr=20, acc_per_cls=False)
    print('Many:', "{:.1%}".format(breakdownResults[0]), 'Medium:', "{:.1%}".format(breakdownResults[1]), 'Few:', "{:.1%}".format(breakdownResults[2]))


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    # This function is excerpted from a publicly available code [commit 01e52ed, BSD 3-Clause License]
    # https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/utils.py

    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def plot_per_epoch_accuracy(trackRecords, save_path, figname):
    train_acc = trackRecords['acc_train']
    test_acc = trackRecords['acc_test']

    plt.title("Training and validation accuracy per epoch")
    plt.plot(torch.Tensor(train_acc).cpu(), label='Train accuracy')
    plt.plot(torch.Tensor(test_acc).cpu(), label='Validation accuracy')

    plt.xlabel('training epochs')
    plt.ylabel('accuracy')
    plt.legend()
    figpath = os.path.join(save_path, figname)
    plt.savefig(figpath)


def get_per_class_acc(model, dataloaders, nClasses=100, device='cpu'):
    predList = np.array([])
    grndList = np.array([])
    model.eval()
    for sample in dataloaders['test']:
        with torch.no_grad():
            images, labels = sample
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            logits = model(images)
            softmaxScores = F.softmax(logits, dim=1)

            predLabels = softmaxScores.argmax(dim=1).detach().squeeze().cpu().numpy()
            predList = np.concatenate((predList, predLabels))
            grndList = np.concatenate((grndList, labels))

    confMat = confusion_matrix(grndList, predList)

    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1, 1))
    confMat = confMat / a

    acc_avgClass = 0
    for i in range(confMat.shape[0]):
        acc_avgClass += confMat[i, i]

    acc_avgClass /= confMat.shape[0]

    acc_per_class = [0] * nClasses

    for i in range(nClasses):
        acc_per_class[i] = confMat[i, i]

    return acc_per_class

def plot_per_class_accuracy(models_dict, dataloaders, labelnames, img_num_per_cls, nClasses=100, device='cuda', save_path='./', figname='fig.png'):
    result_dict = {}
    for label in models_dict:
        model = models_dict[label]
        acc_per_class = get_per_class_acc(model, dataloaders, nClasses=nClasses, device=device)
        result_dict[label] = acc_per_class

    plt.figure(figsize=(15, 4), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(nClasses)), labelnames, rotation=90, fontsize=8);  # Set text labels.
    plt.title('per-class accuracy vs. per-class #images', fontsize=20)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for label in result_dict:
        ax1.bar(list(range(nClasses)), result_dict[label], alpha=0.7, width=1, label=label, edgecolor="black")

    ax1.set_ylabel('accuracy', fontsize=16, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

    ax2.set_ylabel('#images', fontsize=16, color='r')
    ax2.plot(img_num_per_cls, linewidth=4, color='r')
    ax2.tick_params(axis='y', labelcolor='r', labelsize=16)

    ax1.legend(prop={'size': 14})
    figpath = os.path.join(save_path, figname)
    plt.savefig(figpath)

class Normalizer():
    def __init__(self, LpNorm=2, tau=1):
        self.LpNorm = LpNorm
        self.tau = tau

    def apply_on(self, model):  # this method applies tau-normalization on the classifier layer
        for curLayer in [model.fc.weight]:  # change to last layer: Done
            curparam = curLayer.data

            curparam_vec = curparam.reshape((curparam.shape[0], -1))
            neuronNorm_curparam = (
                        torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1) ** self.tau).detach().unsqueeze(-1)
            scalingVect = torch.ones_like(curparam)

            idx = neuronNorm_curparam == neuronNorm_curparam
            idx = idx.squeeze()
            tmp = 1 / (neuronNorm_curparam[idx].squeeze())
            for _ in range(len(scalingVect.shape) - 1):
                tmp = tmp.unsqueeze(-1)

            scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
            curparam[idx] = scalingVect[idx] * curparam[idx]

class Normalizer_multi_GPU():
    def __init__(self, cfg, LpNorm=2, tau=1):
        self.LpNorm = LpNorm
        self.tau = tau
        self.cfg = cfg

    def apply_on(self, model):  # this method applies tau-normalization on the classifier layer
        if self.cfg.dataset == 'cifar10' or self.cfg.dataset == 'cifar100':
            for curLayer in [model.module.encoder.fc.weight]:  # change to last layer: Done
                curparam = curLayer.data

                curparam_vec = curparam.reshape((curparam.shape[0], -1))
                neuronNorm_curparam = (
                            torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1) ** self.tau).detach().unsqueeze(-1)
                scalingVect = torch.ones_like(curparam)

                idx = neuronNorm_curparam == neuronNorm_curparam
                idx = idx.squeeze()
                tmp = 1 / (neuronNorm_curparam[idx].squeeze())
                for _ in range(len(scalingVect.shape) - 1):
                    tmp = tmp.unsqueeze(-1)

                scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
                curparam[idx] = scalingVect[idx] * curparam[idx]
        else:
            for curLayer in [model.module.fc.weight]:  # change to last layer: Done
                curparam = curLayer.data

                curparam_vec = curparam.reshape((curparam.shape[0], -1))
                neuronNorm_curparam = (
                        torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1) ** self.tau).detach().unsqueeze(-1)
                scalingVect = torch.ones_like(curparam)

                idx = neuronNorm_curparam == neuronNorm_curparam
                idx = idx.squeeze()
                tmp = 1 / (neuronNorm_curparam[idx].squeeze())
                for _ in range(len(scalingVect.shape) - 1):
                    tmp = tmp.unsqueeze(-1)

                scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
                curparam[idx] = scalingVect[idx] * curparam[idx]
