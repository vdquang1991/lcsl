import os, copy
import json
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import progressbar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math

import utils
from config import get_parser
from models.resnet import ResnetEncoder
from torchvision.models import resnext50_32x4d, resnet50
from data.dataset_cifar import get_cifar100, get_cifar10
from data.dataset_LT import get_iNaturaList, get_ImageNetLT

import warnings # ignore warnings
warnings.filterwarnings("ignore")

def adjust_learning_rate(optimizer, epoch, cfg, verbose=1):
    """Decay the learning rate based on schedule"""
    lr = cfg.lr
    if epoch < cfg.warmup_epochs:
       lr = lr / cfg.warmup_epochs * (epoch + 1 )
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - cfg.warmup_epochs + 1 ) / (cfg.epochs - cfg.warmup_epochs + 1 )))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if verbose==1:
        print('Adjusting learning rate to ', lr)

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cal_med_tail_classes(labelList):
    labelList = np.asarray(labelList)
    label_list = np.unique(labelList)
    train_class_count = []

    for l in np.unique(label_list):
        train_class_count.append(len(labelList[labelList == l]))

    avg_num_samples_per_class = len(labelList) / cfg.nClasses
    train_class_count = np.asarray(train_class_count)
    med_tail_classes = np.where(train_class_count < avg_num_samples_per_class)[0]
    return med_tail_classes

def check_in_med_tail(targets_teacher, med_tail_classes):
    masks = torch.any(
        torch.stack([torch.eq(targets_teacher, label).logical_or_(torch.eq(targets_teacher, label)) for label in med_tail_classes], dim=0),
        dim=0).float()
    return masks

def print_and_savelog(grndList, predList, log_filename, epoch_error, print2screen_avgAccRate, running_time, running_type='training'):
    confMat = confusion_matrix(grndList, predList)
    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1, 1))
    confMat = confMat / a
    curPerClassAcc = 0
    for i in range(confMat.shape[0]):
        curPerClassAcc += confMat[i, i]
    curPerClassAcc /= confMat.shape[0]

    # Print to screen
    print('\tloss: {:.6f} | acc-all: {:.5f} | acc-avg-cls: {:.5f} | {} time: {:.5f}'.format(epoch_error, print2screen_avgAccRate, curPerClassAcc, running_type, running_time))

    # Save log
    fn = open(log_filename, 'a')
    fn.write('\t{}: loss:{:.6f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}\n'.format(running_type, epoch_error, print2screen_avgAccRate, curPerClassAcc))
    fn.close()
    return curPerClassAcc

def train_model(dataloaders, model, pseudo_teacher, lossFunc, loss_SelfKD, optimizer, scheduler, cfg, labelList=[], num_epochs=200,
                device='cuda', work_dir='./exp', model_name='baseline'):
    trackRecords = {}
    trackRecords['loss_train'] = []
    trackRecords['loss_test'] = []
    trackRecords['acc_train'] = []
    trackRecords['acc_test'] = []
    trackRecords['lr'] = []
    log_filename = os.path.join(work_dir, model_name + '_train.log')

    # ---------------- Load the model weights and continuous training if start_epoch > 1 ----------------
    if cfg.start_epoch > 1:
        path_to_clsnet = os.path.join(work_dir, model_name + '_best.pth')
        pseudo_teacher.load_state_dict(torch.load(path_to_clsnet, map_location=device))
        checkpoints_path = os.path.join(work_dir, 'checkpoints')
        tmp = torch.load('%s/epoch_%03d_model.pth' % (checkpoints_path, cfg.start_epoch), map_location=device)
        model.load_state_dict(tmp)
        jsonfile = os.path.join(work_dir, 'checkpoints', 'log.json')
        trackRecords = json.loads(open(jsonfile).read())
        for k in trackRecords.keys():
            trackRecords[k] = trackRecords[k][:cfg.start_epoch]
        best_perClassAcc = max(trackRecords['acc_test'])
        print('Current best accuracy: ', best_perClassAcc)
    else:
        best_perClassAcc = 0.0

    def get_training_batch():
        while True:
            for sequence in dataloaders['train']:
                yield sequence
    def get_testing_batch():
        while True:
            for sequence in dataloaders['test']:
                yield sequence

    training_batch_generator = get_training_batch()
    num_training_steps = len(dataloaders['train'])
    testing_batch_generator = get_testing_batch()
    num_testing_steps = len(dataloaders['test'])
    if len(cfg.gpu) ==1:
        L2_norm = utils.Normalizer(cfg.fc_norm)
    else:
        L2_norm = utils.Normalizer_multi_GPU(cfg.fc_norm)
    pseudo_teacher.eval()

    med_tail_classes = cal_med_tail_classes(labelList)
    med_tail_classes = torch.from_numpy(med_tail_classes).to(device)

    count_epoch = 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        # Reduce learning rate
        adjust_learning_rate(optimizer, epoch-1, cfg)
        if cfg.fc_norm > 0:
            L2_norm.apply_on(pseudo_teacher)

        print('\nEpoch {}/{}'.format(epoch, cfg.epochs))
        print('-' * 10)
        if epoch <= 1:
            fn = open(log_filename, 'w')
        else:
            fn = open(log_filename, 'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch, cfg.epochs))
        fn.write('--' * 5 + '\n')
        fn.close()

        #---------------------------------------------training ----------------------------------------------------
        progress = progressbar.ProgressBar(max_value=num_training_steps).start()
        start_time = time.time()
        model.train()
        predList = np.array([])
        grndList = np.array([])
        running_loss_CE = 0.0
        running_acc = 0.0

        # Iterate over data.
        iterCount, sampleCount = 0, 0
        for i in range(num_training_steps):
            progress.update(i + 1)
            imageList, labelList = next(training_batch_generator)
            imageList = imageList.to(device)
            labelList = labelList.type(torch.long).view(-1).to(device)

            # Mix data
            _input_mix, target_a, target_b, lam = mixup_data(imageList, labelList)

            # Generate the output from pseudo-teacher
            logits_1 = pseudo_teacher(_input_mix).detach()
            logits_2 = pseudo_teacher(_input_mix.flip(3)).detach()
            logits_teacher = (logits_1 + logits_2) / 2
            softmax_teacher = F.softmax(logits_teacher / cfg.tau, dim=1)

            # Generate probability and pseudo-labels from pseudo-teacher
            max_probs_teacher, targets_teacher = torch.max(softmax_teacher, dim=1)

            # Calculate mask
            mask = max_probs_teacher.ge(cfg.lambda_init).float() * check_in_med_tail(targets_teacher, med_tail_classes)

            # Generate the output from current model
            logits = model(imageList)
            output_mix = model(_input_mix)

            # Calculate two loss functions
            error = cfg.alpha * mixup_criterion(lossFunc, output_mix, target_a, target_b, lam).mean() + \
                    cfg.beta * (loss_SelfKD(output_mix, targets_teacher) * mask).mean()
            # Calculate accuracy
            softmaxScores = logits.softmax(dim=1)
            predLabel = softmaxScores.argmax(dim=1).detach().squeeze().type(torch.float)
            accRate = (labelList.type(torch.float).squeeze() - predLabel.squeeze().type(torch.float))
            accRate = (accRate == 0).type(torch.float).mean()

            predList = np.concatenate((predList, predLabel.cpu().numpy()))
            grndList = np.concatenate((grndList, labelList.cpu().numpy()))

            # backward + optimize only if in training phase
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

            # statistics
            iterCount += 1
            sampleCount += labelList.size(0)
            running_acc += accRate * labelList.size(0)
            running_loss_CE += error.item() * labelList.size(0)

        progress.finish()
        utils.clear_progressbar()
        end_time = time.time()
        epoch_error = running_loss_CE / sampleCount
        print2screen_avgAccRate = running_acc / sampleCount
        curPerClassAcc = print_and_savelog(grndList, predList, log_filename, epoch_error, print2screen_avgAccRate,
                                           end_time-start_time, running_type='training')

        # Write to dict
        trackRecords['acc_train'].append(curPerClassAcc)
        trackRecords['loss_train'].append(epoch_error)

        # ---------------------------------------------testing ----------------------------------------------------
        model.eval()  # Set model to eval mode

        test_running_loss_CE = 0.0
        test_running_acc = 0.0
        test_predList = np.array([])
        test_grndList = np.array([])

        # Iterate over data.
        iterCount, sampleCount = 0, 0
        start_time = time.time()
        for i in range(num_testing_steps):
            imageList, labelList = next(testing_batch_generator)
            imageList = imageList.to(device)
            labelList = labelList.type(torch.long).view(-1).to(device)

            with torch.no_grad():
                logits = model(imageList)
                error = lossFunc(logits, labelList).mean()
                softmaxScores = logits.softmax(dim=1)

            predLabel = softmaxScores.argmax(dim=1).detach().squeeze().type(torch.float)
            accRate = (labelList.type(torch.float).squeeze() - predLabel.squeeze().type(torch.float))
            accRate = (accRate == 0).type(torch.float).mean()

            test_predList = np.concatenate((test_predList, predLabel.cpu().numpy()))
            test_grndList = np.concatenate((test_grndList, labelList.cpu().numpy()))

            # statistics
            iterCount += 1
            sampleCount += labelList.size(0)
            test_running_acc += accRate * labelList.size(0)
            test_running_loss_CE += error.item() * labelList.size(0)

        epoch_error = test_running_loss_CE / sampleCount
        print2screen_avgAccRate = test_running_acc / sampleCount
        test_curPerClassAcc = print_and_savelog(test_grndList, test_predList, log_filename, epoch_error, print2screen_avgAccRate,
                                                time.time()-start_time, running_type='testing')
        # Write to dict
        trackRecords['acc_test'].append(test_curPerClassAcc)
        trackRecords['loss_test'].append(epoch_error)
        current_lr = optimizer.param_groups[0]['lr']
        trackRecords['lr'].append(current_lr)

        # Save the best model
        if test_curPerClassAcc > best_perClassAcc:  # epoch_loss<best_loss:
            print('\t-----------> New best accuracy:{:.5f}'.format(test_curPerClassAcc))
            count_epoch = 0
            best_loss = epoch_error
            best_perClassAcc = test_curPerClassAcc

            path_to_save_param = os.path.join(work_dir, model_name + '_best.pth')
            torch.save(model.state_dict(), path_to_save_param)

            file_to_note_bestModel = os.path.join(work_dir, model_name + '_note_bestModel.log')
            if epoch==1:
                fn = open(file_to_note_bestModel, 'w')
            else:
                fn = open(file_to_note_bestModel, 'a')
            fn.write('The best model is achieved at epoch-{}: loss{:.5f}, acc-all:{:.5f}, acc-avg-cls:{:.5f}.\n'.format(epoch, best_loss, print2screen_avgAccRate, best_perClassAcc))
            fn.close()
            pseudo_teacher.load_state_dict(torch.load(path_to_save_param, map_location=device))
            print('\t-----------> Load new weights for the pseudo teacher')
        else:
            print('\tCurrent best accuracy:{:.5f}'.format(best_perClassAcc))
            count_epoch +=1
            print('\tCurrent count_epoch:{:3d}'.format(count_epoch))

        # Save model checkpoint at every epoch
        checkpoints_path = os.path.join(work_dir, 'checkpoints')
        jsonfile = os.path.join(work_dir, 'checkpoints', 'log.json')
        f = open(jsonfile, "w")
        f.write(json.dumps(trackRecords))
        f.close()
        torch.save(model.state_dict(), '%s/epoch_%03d_model.pth' % (checkpoints_path, epoch))
        if os.path.exists('%s/epoch_%03d_model.pth' % (checkpoints_path, epoch - 1)):
            os.remove('%s/epoch_%03d_model.pth' % (checkpoints_path, epoch - 1))

    return trackRecords

def main(cfg):
    # set device, which gpu to use.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)  # Choose GPU for training
    print('Choose GPU index: ', cfg.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.__version__)
    torch.manual_seed(0)
    np.random.seed(0)

    assert cfg.dataset in ['cifar10', 'cifar100', 'imagenet', 'inaturalist']

    if cfg.dataset == 'cifar10':
        cfg.nClasses = 10  # number of classes in CIFAR10-LT
    elif cfg.dataset == 'cifar100':
        cfg.nClasses = 100  # number of classes in CIFAR100-LT
    elif cfg.dataset == 'imagenet':
        cfg.nClasses = 1000  # number of classes in ImageNet-LT
    else:
        cfg.nClasses = 8142  # number of classes in iNaturaList 2018

    torch.cuda.empty_cache()
    curr_working_dir = os.getcwd()

    # Load dataloader
    if cfg.dataset == 'cifar100':
        dataloaders, img_num_per_cls, new_labelList, labelnames = get_cifar100(cfg)
    elif cfg.dataset == 'cifar10':
        dataloaders, img_num_per_cls, new_labelList, labelnames = get_cifar10(cfg)
    elif cfg.dataset == 'imagenet':
        dataloaders, new_labelList = get_ImageNetLT(cfg)
    else:
        dataloaders, new_labelList = get_iNaturaList(cfg)

    # Build model
    if cfg.dataset == 'cifar100' or cfg.dataset == 'cifar10':
        model = ResnetEncoder(embDimension=cfg.nClasses)
        pseudo_teacher = ResnetEncoder(embDimension=cfg.nClasses)
    elif cfg.dataset == 'imagenet':
        model = resnext50_32x4d(pretrained=False)
        pseudo_teacher = resnext50_32x4d(pretrained=False)
    else:
        model = resnet50(pretrained=False)
        pseudo_teacher = resnet50(pretrained=False)

    if len(cfg.gpu) > 1:
        print('Training with multi GPU')
        model = nn.DataParallel(model)
        pseudo_teacher = nn.DataParallel(pseudo_teacher)

    model.to(device)
    pseudo_teacher.to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs - cfg.start_epoch + 1, eta_min=1e-8, verbose=True)

    loss_CrossEntropy = nn.CrossEntropyLoss(reduction='none').to(device)
    loss_SelfKD = nn.CrossEntropyLoss(reduction='none').to(device)

    if cfg.dataset == 'cifar100' or cfg.dataset == 'cifar10':
        project_name = 'LCSL_' + cfg.dataset + '_IF_' + str(int(1 / cfg.imb_factor)) + '_data_aug_' + cfg.data_aug + '_network_' + cfg.network
    else:
        project_name = 'LCSL_' + cfg.dataset + '_data_aug_' + cfg.data_aug + '_network_' + cfg.network
    print('project_name: ', project_name)

    save_dir = os.path.join(curr_working_dir, project_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "checkpoints")):
        os.mkdir(os.path.join(save_dir, "checkpoints"))

    # Save all args
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    trackRecords = train_model(dataloaders, model, pseudo_teacher, loss_CrossEntropy, loss_SelfKD, optimizer, scheduler, cfg, labelList=new_labelList,
                               num_epochs=cfg.epochs, device=device, work_dir=save_dir, model_name=project_name)

    # load model with the best epoch accuracy
    path_to_clsnet = os.path.join(save_dir, project_name + '_best.pth')
    model.load_state_dict(torch.load(path_to_clsnet, map_location=device))

    models = {}
    model.to(device)
    model.eval()
    models[project_name] = model

    print('\n')
    print('The results on test dataset:')

    utils.print_accuracy(model, dataloaders, new_labelList, device=device)
    if cfg.dataset == 'cifar100' or cfg.dataset == 'cifar10':
        utils.plot_per_epoch_accuracy(trackRecords, save_path=save_dir, figname='train_test_curve_{}.png'.format(project_name))
        utils.plot_per_class_accuracy(models, dataloaders, labelnames, img_num_per_cls, nClasses=cfg.nClasses, device=device,
                                      save_path=save_dir, figname='visual_{}.png'.format(project_name))

    print('\n')
    print('Apply L2 at the FC layer')

    model_l2 = copy.deepcopy(model)
    model_l2.eval()
    if len(cfg.gpu) == 1:
        L2_norm = utils.Normalizer(tau=1.9)
    else:
        L2_norm = utils.Normalizer_multi_GPU(tau=1.9)
    L2_norm.apply_on(model_l2)
    models['L2 normalized {}'.format(project_name)] = model_l2
    utils.print_accuracy(model_l2, dataloaders, new_labelList, device=device)
    if cfg.dataset == 'cifar100' or cfg.dataset == 'cifar10':
        utils.plot_per_class_accuracy(models, dataloaders, labelnames, img_num_per_cls, nClasses=cfg.nClasses, device=device,
                                  save_path=save_dir, figname='visual_{}_l2.png'.format(project_name))

if __name__ == '__main__':
    cfg = get_parser()
    main(cfg)


