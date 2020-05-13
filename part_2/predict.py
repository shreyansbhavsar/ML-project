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
arg = argparse.ArgumentParser(description='Predict')
arg.add_argument('--top_k',  action='store', default=5, type=int)
arg.add_argument('--category_names', action='store', default='./cat_to_name.json')
arg.add_argument('--gpu', action='store', default='gpu') #default necessário?
arg.add_argument('--input_img', action='store', default='./flowers/test/1/image_06760.jpg', type=str)
arg.add_argument('--checkpoint', action='store', default='./checkpoint.pth', type=str)

parseArgs = arg.parse_args()
image_dir = parseArgs.input_img
output = parseArgs.top_k
processing = parseArgs.gpu
check = parseArgs.checkpoint

device = 'cuda' if processing ==  'gpu' else 'cpu'

dataloaders_train, dataloaders_test, dataloaders_valid, image_datasets_train, image_datasets_test, image_datasets_valid = utilitys.load_data('./flowers')

model = utilitys.load_checkpoint(check)

#Font: Udacity Deep Learning with PyToorch: Part 8 - Transfer Learning
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

plt.rcdefaults()

index = 1
topk=5

ps = utilitys.predict(image_dir, model, topk, device)
image = utilitys.image_processing(image_dir)

a = np.array(ps[0][0])
b = [cat_to_name[str(index+1)] for index in np.array(ps[1][0])]

print("Probability: {}".format(a))
print("Labels: {}".format(b))

x = 0
bigger_prob = 0.0
bigger_label = ""
while x < len(b):
    print("For label {}, the probability is {}".format(b[x], a[x]))
    if bigger_prob <= a[x]:
        bigger_prob = a[x]
        bigger_label = b[x]
    x += 1

print("The bigger proability is {} for label {}".format(bigger_prob, bigger_label))
    