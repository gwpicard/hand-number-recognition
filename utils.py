# utils.py
# support functions for project

import cv2
import datetime
import numpy as np
import torch
from PIL import Image

# function to save and build dataset
def create_data(key, frame):
    key = chr(key) # convert ASCII code back to char
    time = str(datetime.datetime.now()).replace(" ","_")
    name = 'data/'+key+'/frame' + time + '.jpg'
    cv2.imwrite(name, frame)

def import_image(image_array):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    overlay = Image.fromarray(image_array)

    # resize image
    overlay.thumbnail((224,224), Image.ANTIALIAS)

    image = Image.new("RGB", (224,224))

    width, height = overlay.size

    image.paste(overlay, ((224-width)//2,
                    (224-height)//2))

    # convert integers to 0-1 floats
    np_image = np.array(image)
    np_image = np_image/255

    # normalise
    np_image -= [0.485, 0.456, 0.406]
    np_image /= [0.229, 0.224, 0.225]

    np_image = np_image.transpose()
    py_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    py_image.unsqueeze_(0)

    return py_image
