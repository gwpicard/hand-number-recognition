import cv2
from utils import create_data, convert_image, overlay_labels
from model import load_model, predict_label

# load model for inference
model,_,_ = load_model('checkpoint.pth')

# create video object
video = cv2.VideoCapture(0)

# define range of keystrokes for webcam image capture
min = 1
max = 6
key_range = [ord(str(i)) for i in range(min, max+1)]

frames = 0
while(True):
    # Capture frame from video object (webcam)
    check, frame = video.read()

    # flip frame horizontally for mirror effect
    frame = cv2.flip(frame, +1);

    # capture keystroke
    key = cv2.waitKey(1)

    # exit stream command
    if key == ord('q'):
        break

    # update inference every 5 frames
    if frames % 5 == 0 or frames == 0:
        image = convert_image(frame)
        predict = predict_label(model, image)

    # frame = overlay_labels(frame, predict)

    # # Display the resulting frame
    cv2.imshow('frame', frame)

    # capture images to be saved to build dataset
    if key in key_range:
        create_data(key, frame)

    # increment frame count
    frames+=1

# Release capture upon exit
video.release()
cv2.destroyAllWindows()
