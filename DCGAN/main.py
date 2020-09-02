from __future__ import print_function

import argparse
import os
import random
import torch 
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset 
import torch vision.transforms as transforms
import torchvision.utils as vutils
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999

print("Random seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "./data/celeba"
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64 # Size of feature maps in G
ndg = 64 # Size of feature maps in D
num_epochs = 5

lr = 0.0002

beta1 = 0.5
ngpu = 1

#Dataset
transform = transforms.Compose([
				transforms.Resize(image_size),
				transforms.CenterCrop(image_size), 
				transforms.ToTensor(), 
				transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)
					)])

train_dataset = dset.CelebA(dataroot, split='train', transform=transform, download=True)
valid_dataset = dset.CelebA(dataroot, split='val', download=True)
