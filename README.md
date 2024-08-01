# LCSL: Long-tailed Classification via Self-labeling

By Duc-Quang Vu, Trang T.T. Phung, Jia-Ching Wang, and Son T. Mai

## Overview
Code for the paper "LCSL: Long-tailed Classification via Self-labeling".

### Abstract
During the last decades, deep learning (DL) has been proven to be a very powerful and successful technique in many real-world applications, e.g., video surveillance or object detection. However, when class label distributions are highly skewed, DL classifiers tend to be biased towards majority classes during training phases. This leads to poor generalization of minority classes and consequently reduces the overall accuracy. How to effectively deal with this long-tailed class distribution in DL, i.e., deep long-tailed classification (DLC), remains a challenging problem despite many research efforts. Among various approaches, data augmentation, which aims at generating more samples for reducing label imbalance, is the most common and practical one. However, simply relying on existing class-agnostic augmentation strategies without properly considering the label differences would worsen the problem since more head-class samples can be inevitably augmented than tail-class ones. Moreover, none of the existing works consider the quality and suitability of augmented samples during the training process. Our proposed approach, called Long-tailed Classification via Self-Labeling (LCSL), is specifically designed to address these limitations. LCSL fundamentally differs from existing works by the way it iteratively exploits the preceding network during the training process to re-label the labeled augmented samples and uses the output confidence to decide whether new samples belong to minority classes before adding them to the data. Not only does this help to reduce imbalance ratios among classes, but this also helps to reduce the uncertainty of class prediction problems for minority classes by selecting more confident samples to the data. This incremental learning and generating scheme thus provide a new robust approach for decreasing model over-fitting, thus enhancing the overall accuracy, especially for minority classes. Extensive experiments have demonstrated that LCSL acquires better performance than state-of-the-art long-tailed learning techniques on various standard benchmark datasets.  More specifically, our LCSL obtains 85.8\%, 54.4\%, and 56.2\% in terms of accuracy on CIFAR10-LT, CIFAR100-LT, and ImageNet-LT (with moderate to extreme imbalance ratios), respectively.

<p align="center">
  <img width="800" alt="fig_method" src="model.png">
</p>

## Running the code

### Requirements
- Python3
- Pytorch
- numpy 
- Pillow
- opencv
- ...

### Training

In this code, you can reproduce the experiment results of the LCSL approach for Long-tailed Classification.
The datasets are all open-sourced, so it is easy to download.
Detailed hyperparameter settings are enumerated in the paper.

- Training with LCSL
~~~
python train_LCSL.py --dataset='cifar100' --batch_size=64 --imb_factor=0.01 --alpha=1. --beta=1. --lr=0.01 \
--gpu=0 --weight_decay=5e-3 --epochs=300 --lambda_init=0.5
~~~
In which,

`--gpu=0` denotes which GPU we use to train the network, if you want to use multi-GPUs, you can set `--gpu=0,1,2,3`.

`--dataset=cifar100` denotes the dataset used for training.

`--imb_factor=0.01` is imbalance factor which only apply for cifar10 and cifar100 datasets.

`--alpha=1` and `--beta=1` are two loss weights.

`--lr=0.01` is the learning rate init.

`--epochs=300` denotes the number of epochs for training

`--lambda_init=0.5` is the hyperparameter `conf`

All datasets should be saved in the folder `"datasets"` 

## Citation
Coming soon.
~~~
@article{vu2024lcsl,
  title={LCSL: Long-tailed Classification via Self-labeling},
  author={Vu, Duc-Quang and Phung, Trang TT and Wang, Jia-Ching and Mai, Son T},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
~~~

