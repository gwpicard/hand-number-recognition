import argparse
import cv2
from utils import create_data, convert_image, overlay_labels, overlay_top_label
from model import load_model, predict_label

# include arg parser to take custom number of photos for dataset
parser = argparse.ArgumentParser()

parser.add_argument('-i','--infer', action="store_true", required=False, dest="infer")
parser.add_argument('-n','--n_photos', action="store", dest="n_photos", type=int)
parser.add_argument('-c','--cat', action="store", dest="cat", type=str)

# parse + read arguments
args = parser.parse_args()

n_photos = args.n_photos # number of photos to take
cat = args.cat # label of data to be captured
infer = args.infer # whether to infer labels

# both n_photos and category label should be chosen
if (n_photos and cat is None) or (n_photos is None and cat):
    raise Exception('Both number of photos and category label should be specified')

# print video capture mode
if args.n_photos:
    print("Capturing data..")
    cat = ord(cat)
    photos = 1
if args.infer:
    print("Inferring from model..")

if infer:
    # load model for inference if infer flag
    model,_,_ = load_model('checkpoint.pth')

# create video object
video = cv2.VideoCapture(0)

frames = 0
while(True):
    # Capture frame from video object (webcam)
    check, frame = video.read()

    # flip frame horizontally for mirror effect
    frame = cv2.flip(frame, +1);

    # capture keystroke
    key = cv2.waitKey(1)

    # exit stream key command
    if key == ord('q'):
        break

    # capture images automatically if photos arg was provided
    if n_photos and (frames % 2 == 0 or frames == 0):
        print(photos)
        create_data(cat, frame)
        photos+=1

    # break loop once phots have been captured
    if n_photos and photos > n_photos:
        print("Data capture finished.")
        break

    # update inference every 5 frames
    if infer and (frames % 8 == 0 or frames == 0):
        image = convert_image(frame)
        predict = predict_label(model, image)

    # overlay labels if infer flag is select
    if infer:
        frame = overlay_labels(frame, predict)
        # frame = overlay_top_label(frame, top_label)

    # display the image
    cv2.imshow('frame', frame)

    # increment frame count
    frames+=1

# Release capture upon exit
video.release()
cv2.destroyAllWindows()
