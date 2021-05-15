import argparse
import os

import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import Variable

def args_parser():
    parser = argparse.ArgumentParser(description='trainer')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--architecture', type=str, default='vgg16', help='architecture')
    parser.add_argument('--hidden_units', type=int, default=[256,128], help='Hidden layers')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save to file')
    
    args = parser.parse_args()
    return args

def process_data(train_dir, test_dir, valid_dir): 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                      ])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                     ])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    
    return trainloader, validloader, testloader

def basic_model(architecture):
    if architecture == None or architecture == 'vgg':
        load_model = models.vgg16(pretrained=True)
        print('Use vgg16')
    else:
        print('Defaulting to vgg16.')
        load_model = models.vgg16(pretrained=True)
        
    return load_model

def set_classifier(model, hidden_units):
    if hidden_units == None:
        hidden_units = (256, 128)
        
    input = model.classifier[0].in_features
    classifier = nn.Sequential(nn.Linear(25088, 256),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(128,102),
                            nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    return model

def train_model(epochs, trainloader, validloader, gpu, model, optimizer, criterion):
    if type(epochs) == type(None):
        epochs = 5
        print("Epochs = 5")
        steps = 0
        
        model.to('cuda')
        
        running_loss = 0
        print_every = 60
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                inputs, labels = inputs.to(device), labels.to(device)
                    
                model.train()
                optimizer.zero_grad()
                
                logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {valid_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
    return model

def valid_model(model, testloader, gpu):
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for test_inputs, test_labels in testloader:
            if gpu==True:
                test_inputs, test_labels = test_inputs.cuda(), test_labels.cuda()
            else:
                test_inputs, test_labels = test_inputs.cpu(), test_labels.cpu()
        test_logps = model.forward(test_inputs)
        test_batch_loss = criterion(test_logps, labels)
                    
        test_loss += test_batch_loss.item()
        
        # calculate accuracy
        test_ps = torch.exp(test_logps)
        top_p, test_top_class = test_ps.topk(1, dim=1)
        test_equals = test_top_class == labels.view(*test_top_class.shape)
        test_accuracy += torch.mean(test_equals.type(torch.FloatTensor)).item()
        final_accuracy = (test_accuracy/len(testloader))            
    print(f"Test accuracy: {final_accuracy}")
    
    return final_accuracy

def save_checkpoint(model, train_data, save_dir):
    model.class_to_idx = train_datasets.class_to_idx
    
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layers': [256, 128],
              'state_dict': state_dict,
              'classifier': classifier,
              'image_data': model.class_to_idx,
              'epochs': epochs,
              'optimizer': optimizer,
              'optimizer_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx
             }

    return torch.save(checkpoint, save_dir)

def main():
    args = args_parser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    trainloader, testloader, validloader = process_data(train_dir, test_dir, valid_dir)
    
    model = basic_model(args.architecture)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model = set_classifier(model, args.hidden_units)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    trmodel = train_model(args.epochs, trainloader, validloader, args.gpu, model, optimizer, criterion)
    
    valid_model(trmodel, testloader, args.gpu)
    save_checkpoint(trmodel, train_data, args.save_dir)
    print('Completed!')
    
if __name__ == '__main__': main()
                



