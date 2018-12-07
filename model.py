from collections import OrderedDict
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models
import torch.nn.functional as F

def initiate_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # create custom classifier layer on top of VGG16
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs, 1024)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.3)),
        ('fc2', nn.Linear(1024, 5)), # for the 5 possible hand categories
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # replace VGG16 output classifying layer
    model.fc = classifier

    checkpoint = torch.load('model.pth')

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model
