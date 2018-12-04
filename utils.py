# utils.py
# support functions for project

import cv2
import datetime

# function to save and build dataset
def create_data(key, frame):
    key = chr(key) # convert ASCII code back to char
    time = str(datetime.datetime.now()).replace(" ","_")
    name = 'data/'+key+'/frame' + time + '.jpg'
    cv2.imwrite(name, frame)
