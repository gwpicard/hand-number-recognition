# utils.py
# support functions

import cv2
import datetime
import numpy as np
import torch
from PIL import Image

# function to save and build dataset
def create_data(key, frame):
    '''
    Captures image to save to dataset based on
    key pressed as input
    '''
    key = chr(key) # convert ASCII code back to char
    time = str(datetime.datetime.now()).replace(" ","_")
    name = 'data/'+key+'/frame' + time + '.jpg'
    cv2.imwrite(name, frame)

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # define data transforms
    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    return train_dataset, valid_dataset, test_dataset

def convert_image(image_array):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    overlay = Image.fromarray(image_array)

    # resize image
    overlay.thumbnail((224,224), Image.ANTIALIAS)

    # create base layer canvas
    image = Image.new("RGB", (224,224))

    width, height = overlay.size

    # place resized image on canvas
    image.paste(overlay, ((224-width)//2,
                    (224-height)//2))

    # convert integers to 0-1 floats
    np_image = np.array(image)
    np_image = np_image/255

    # normalise
    np_image -= [0.485, 0.456, 0.406]
    np_image /= [0.229, 0.224, 0.225]

    # convert to tensor
    np_image = np_image.transpose()
    py_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    py_image.unsqueeze_(0)

    return py_image

def overlay_labels(frame, predict):
    '''
    Takes prediction labels and overlays them on top
    of webcam image stream
    '''
    # format labels from prediction
    labels = ''.join('%d: %.3f  '%(i, p) for (i, p) in predict)

    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (10,700)
    font_size = 1
    font_colour = (255,255,255)
    line_type = 2

    # overlay labels on top of frame
    cv2.putText(frame,labels,
                pos,font,
                font_size,
                font_colour,
                line_type)

    return frame
