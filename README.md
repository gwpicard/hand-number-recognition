# Hand Digit Recognition

A simple app that recognises digits from hand motions from a live webcam feed.

![test_gif.gif](test_gif.gif)

This app was trained with a custom dataset built using the `video_stream.py` file which contains video capture features using OpenCV.

## Jupyter Notebook Pre-reading

Navigate to the Jupyter Notebook first to get a full rundown of how the system works, create the correct directories to build your own dataset, and a full explanation of how the system is trained.

## Capture data

First, ensure you have created the proper folders to store the data using code from the top of the Jupyter Notebook.

To capture data:
`python3 video_stream.py --cat [cat] --n_photos [n_photos]`

Specify which category you want to capture data for (1 to 6, 6 being no digits shown) and the number of photos you want to take

## Train the model with Captured data

Once you have captured the data, return to the Jupyter notebook to split the data out into training, validation and testing data.

To train a new model:

`python3 train.py --data-dir images --epochs [epochs]`

`--data-dir` specificies the directory of the training data, and `--epochs` specifies the number of epochs you want to train your model for. The training function will automatically save a checkpoint after it completes so you can leave it training without needing to worry about the progress.

## Using the model for Inference with Live Webcam Feed
`python3 video_stream.py -i [-t]`

`-i` sets the program to inference mode, and the optional `[-t]` flag can be selected if you would like to the see the label of the top predicted category as opposed to the output probability for every label.
