import cv2
import numpy as np
import torch
from utils import create_data, import_image
from model import initiate_model

model = initiate_model()

# create video object
video = cv2.VideoCapture(0)

# define range of keystrokes to recognise
min = 1
max = 5
key_range = [ord(str(i)) for i in range(min, max+1)]

frames = 0
while(True):
    # Capture frame from video object (webcam)
    check, frame = video.read()

    # flip frame horizontally
    frame = cv2.flip(frame, +1);

    # capture keystroke
    key = cv2.waitKey(1)

    if frames % 5 == 0 or frames == 0:
        image = import_image(frame)
        output = model.forward(image)
        ps = torch.exp(output)
        probs, idx = ps.topk(5)
        probs = probs.detach().numpy().flatten()
        idx = (idx.detach().numpy()+1).flatten()
        comb = sorted(zip(idx, probs), key=lambda x:x[0])
        update = ''.join('%d: %.3f  '%(i, p) for (i, p) in comb)
        # print(update)


    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,700)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(frame, update,
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    lineType)

    # # Display the resulting frame
    cv2.imshow('frame', frame)

    # capture images to be saved to build dataset
    if key in key_range:
        create_data(key, frame)

    # exit stream command
    if key == ord('q'):
        break

    frames+=1
# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
