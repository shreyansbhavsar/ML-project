import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import utilitys
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import datetime
import json

# Font: https://pymotw.com/3/argparse/
arg = argparse.ArgumentParser(description='Tranning Network')
arg.add_argument('--save_dir',  action='store', default='./checkpoint.pth')
arg.add_argument('--data_dir', action='store', default='./flowers/')
arg.add_argument('--arch', action='store', default='vgg16')
arg.add_argument('--learning_rate', action='store', default=0.01, type=float)
arg.add_argument('--hidden_units', action='store', default=25088, type=int)
arg.add_argument('--epochs', action='store', default=20, type=int)
arg.add_argument('--gpu', action='store', default='gpu')

parseArgs = arg.parse_args()
saveDir = parseArgs.save_dir
dataDir = parseArgs.data_dir
architeture = parseArgs.arch
learningRate = parseArgs.learning_rate
hiddenUnits = parseArgs.hidden_units
epochs = parseArgs.epochs
processing = parseArgs.gpu

device = 'cuda' if processing ==  'gpu' else 'cpu'

dataloaders_train, dataloaders_test, dataloaders_valid, image_datasets_train, image_datasets_test, image_datasets_valid = utilitys.load_data(dataDir)

arch = models.vgg16(pretrained=True)

criterion, optimizer, model, classifier = utilitys.build_train(device, arch, hiddenUnits)

print_every = 10

utilitys.testing_network(epochs, model, optimizer, criterion, print_every, dataloaders_train, device, dataloaders_test)
utilitys.save_checkpoint(hiddenUnits, architeture, learningRate, epochs, classifier, model, optimizer, saveDir, image_datasets_train)
utilitys.load_checkpoint(saveDir)

print('Tranning Complete')