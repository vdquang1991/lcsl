import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Training LCSL approach')
    parser.add_argument('--image_size', type=int, default=32, help='32 for cifar and 224 for imagenet and inaturalist')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'imagenet', 'inaturalist'], default='cifar100', help='dataset to train')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--optimizer', default='sgd', help='optimizer to train with')
    parser.add_argument('--imb_type', default='exp', help='optimizer to train with')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha factor')
    parser.add_argument('--beta', default=1.0, type=float, help='beta factor')
    parser.add_argument('--tau', default=1.0, type=float, help='tau factor')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate init')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs')
    parser.add_argument('--data_aug', type=str, default='none', help='Apply data augmentation for training data')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--gpu', type=str, default='1', help='GPU id')
    parser.add_argument('--fc_norm', type=float, default='1.9', help='Apply L2 norm for the FC layer')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--lambda_init', default=0.5, type=float, help='lambda factor')

    cfg = parser.parse_args()
    if cfg.dataset == 'cifar10':
        cfg.nClasses = 10
        cfg.network = 'ResNet32'
    elif cfg.dataset == 'cifar100':
        cfg.nClasses = 100
        cfg.network = 'ResNet32'
    elif cfg.dataset == 'imagenet':
        cfg.nClasses = 1000
        cfg.network = 'ResNeXt50'
    else:
        cfg.nClasses = 8142
        cfg.network = 'ResNet50'
    return cfg
