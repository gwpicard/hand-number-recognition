import cv2
import numpy as np
from utils import create_data

# create video object
video = cv2.VideoCapture(0)

# define range of keystrokes to recognise
min = 1
max = 5
key_range = [ord(str(i)) for i in range(min, max+1)]

while(True):
    # Capture frame from video object (webcam)
    check, frame = video.read()

    # # Display the resulting frame
    cv2.imshow('frame', frame)

    # capture keystroke
    key = cv2.waitKey(1)

    # capture images to be saved to build dataset
    if key in key_range:
        create_data(key, frame)

    # exit stream command
    if key == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
