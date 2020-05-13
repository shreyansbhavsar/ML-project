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

#Font: based to Udacity Deep Learning with PyToorch: Part 8 - Transfer Learning
def load_data(dir):
    data_dir = dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    transform_train = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    transform_valid = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    image_datasets_train = datasets.ImageFolder(train_dir, transform=transform_train)
    image_datasets_test = datasets.ImageFolder(test_dir, transform=transform_test)
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform=transform_valid)

    dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=64)
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size=64)
    
    return dataloaders_train, dataloaders_test, dataloaders_valid, image_datasets_train, image_datasets_test, image_datasets_valid

#Font:based to  Udacity Deep Learning with PyToorch: Part 8 - Transfer Learning
def build_train(device, model, hiddenUnits):
    # TODO: Build and train your network
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 2048)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(2048, 256)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(256, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.to(device);
    
    return criterion, optimizer, model, classifier

#Font: based to Udacity Deep Learning with PyToorch: Part 8 - Transfer Learning
def testing_network(epochs, model, optimizer, criterion, print_every, dataloaders_train, device, dataloaders_test):
    date_time = datetime.datetime.now()
    date_and_hour_start = date_time.strftime("%Y/%m/%d %H:%M")
    print('Running training start: '+date_and_hour_start)
    start = time.time()

    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in dataloaders_train:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in dataloaders_test:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {test_loss/len(dataloaders_test):.3f}.. "
                      f"Accuracy: {accuracy/len(dataloaders_test):.3f}")
                running_loss = 0
                model.train()

    time_elapsed = time.time() - start
    date_and_hour_end = date_time.strftime('%Y/%m/%d %H:%M')
    print('\nRunning training end: '+date_and_hour_end)
    print('\nTime of tranning: {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))
    
#Font: based to Udacity Deep Learning with PyToorch: Part 6 - Saving and Loading Models-cn, https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch and https://pytorch.org/tutorials/beginner/saving_loading_models.html
def save_checkpoint(hiddenUnits, architeture, learningRate, epochs, classifier, model, optimizer, checkpoint, image_datasets_train):
    model.class_to_idx = image_datasets_train.class_to_idx

    torch.save({'input_size': hiddenUnits,
                  'output_size': 102,
                  'arch': architeture,
                  'classifier' : classifier,
                  'learning_rate': learningRate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                 }, checkpoint)
    
# font: based to https://pytorch.org/tutorials/beginner/saving_loading_models.html and https://stackoverflow.com/questions/54677683/how-to-load-a-checkpoint-file-in-a-pytorch-model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    
    return model
    
#Font: based to Udacity Deep Learning with PyToorch: Part 3 - Training Neural Networks, https://pytorch.org/docs/stable/torchvision/transforms.html and https://discuss.pytorch.org/t/transforms-resize-the-value-of-the-resized-pil-image/35372
def image_processing(images_dir):
    # TODO: Process a PIL image for use in a PyTorch model

    img_pil = Image.open(images_dir)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    image = adjustments(img_pil)
    
    return image

#Font: based to https://medium.com/@andreluiz_4916/pytorch-neural-networks-to-predict-matches-results-in-soccer-championships-part-ii-3d02b2ddd538
def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()
    imagem = image_processing(image_path)
    imagem = imagem.numpy()
    imagem = torch.from_numpy(np.array([imagem]))

    with torch.no_grad():
        output = model.forward(imagem.cuda())
        
    prob = F.softmax(output.data,dim=1)
    
    return prob.topk(topk)
