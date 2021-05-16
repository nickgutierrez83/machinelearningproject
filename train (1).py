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
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--architecture', type=str, default='vgg16', help='architecture')
    parser.add_argument('--input_layer', type=int, default=25088, help='Input Layer')
    parser.add_argument('--hidden_one', type=int, default=256, help='Hidden layer 1')
    parser.add_argument('--hidden_two', type=int, default=128, help='Hidden layer 2')
    parser.add_argument('--output_layer', type=int, default=102, help='Output Layer')
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
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle = True)
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

def set_classifier(model, input_layer, hidden_unit_1, hidden_unit_2, output_layer):
    if hidden_unit_1 == None:
        hidden_unit_1 = 256
    if hidden_unit_2 == None:
        hidden_unit_2 = 128
    input = model.classifier[0].in_features
    classifier = nn.Sequential(nn.Linear(input_layer, hidden_unit_1),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(hidden_unit_1, hidden_unit_2),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(hidden_unit_2, output_layer),
                            nn.LogSoftmax(dim=1))

    model.classifier = classifier
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    return model

def train_model(epochs, trainloader, gpu, model, optimizer, criterion, save_dir):
    if type(epochs) == type(None):
        print("Epochs = " + epochs)
        
    steps = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    model.to(device)
        
    running_loss = 0
    print_every = 60
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            steps += 1
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
                    

                
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
            
    state_dict = model.state_dict()

    checkpoint = {'input_size': input_layer,
              'output_size': output_layer,
              'hidden_layers': [hidden_one, hidden_two],
              'state_dict': state_dict,
              'classifier': model.classifier,
              'image_data': model.class_to_idx,
              'epochs': epochs,
              'optimizer': optimizer,
              'optimizer_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx
             }
    save_dir = 'checkpoint_vgg.pth'
    model_checkpoint = torch.save(checkpoint, save_dir)
    return model


    

def main():
    args = args_parser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    trainloader, testloader, validloader = process_data(train_dir, test_dir, valid_dir)
    
    model = basic_model(args.architecture)
    model.class_to_idx = datasets.ImageFolder(train_dir).class_to_idx
    
    for param in model.parameters():
        param.requires_grad = False
        
    model = set_classifier(model, args.input_layer, args.hidden_one, args.hidden_two, args.output_layer)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    trmodel = train_model(args.epochs, trainloader, args.gpu, model, optimizer, criterion, args.save_dir)

    # save_checkpoint(trmodel, save_dir)
    print('Completed!')
    print(trmodel.state_dict())
if __name__ == '__main__': main()
                



