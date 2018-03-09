from __future__ import print_function
import os, sys
import os
import dataset_loader
import torch
import util
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

ann_path = '/data/xiaobing.wang/xiangyu.zhu/FashionAI/data/warm_up_train/Annotations/annotations.csv'
img_dir = '/data/xiaobing.wang/xiangyu.zhu/FashionAI/data/warm_up_train/'

train_loader = torch.utils.data.DataLoader(
		dataset_loader.dataset_loader(img_dir, ann_path, 8,
		                              transforms.ToTensor()),
		batch_size=4, shuffle=True,
		num_workers=2, pin_memory=True)

for i, (input, heatmap) in enumerate(train_loader):
    imgs = input.numpy()
    heats = heatmap.numpy()
    break