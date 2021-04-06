# Author - Saffan Ahmed

# Training Script:

"""

Purpose of training script will be used to train the CNN Model for the TSDR_Dataset.

Goals for this script:
1 - Load the training and testing split from the TSDR_Dataset.
2 - Preprocessing the images to improve clarity and avoid class skew.
3 - Training the CNN model with TSDR_Dataset.
4 - Evaluate the accuracy of the CNN Model / Output.
5 - Store the CNN model to personal computer disk-space to make predictions for new Traffic Sign Inputs.

"""

# Import all the required packages.
from TrafficSignSearch.tsdr_cnn_model import TSDR_CNN_Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report

from skimage import exposure
from skimage import io
from skimage import transform

import matplotlib.pyplot as plplt
import random as rd
import numpy as np
import argparse as ap
import os

# Set matplotlib Backend so numerical values can be stored on local disk.
import matplotlib
matplotlib.use("Agg")


# Loading data from disk
def load_tsdr_data(baseDir, csvDir):
    ts_data = []
    ts_labels = []

    data_rows = open(csvDir).read().strip().split("\n")[1:]
    rd.shuffle(data_rows)

    # loop over the rows of the CSV file
    for (x, data_row) in enumerate(data_rows):
        # check to see if we should show a status update
        if x > 0 and x % 1000 == 0:
            print("[INFO] processed {} total images".format(x))
        # split the row into components and then grab the class ID
        # and image path
        (ts_label, imageDir) = data_row.strip().split(",")[-2:]
        # Derive the full directory of the image file and load it
        imageDir = os.path.sep.join([baseDir, imageDir])
        trafficSignImage = io.imread(imageDir)

        # resize the image to be 32x32 pixels, ignoring aspect ratio,
        # and then perform Contrast Limited Adaptive Histogram
        # Equalization (CLAHE)
        trafficSignImage = transform.resize(trafficSignImage, (32, 32))
        trafficSignImage = exposure.equalize_adapthist(
            trafficSignImage, clip_limit=0.1)
        # update the list of data and labels, respectively
        ts_data.append(trafficSignImage)
        ts_labels.append(int(ts_label))

    # Convert the Image Data (Pixels) and Image Labels into Numpy Array's
    ts_data = np.array(ts_data)
    ts_labels = np.array(ts_labels)

    # Return a tuple of the Image Data (Pixels) and Image Labels
    return (ts_data, ts_label)


# Construct the argument parser and parse the arguments
argp = ap.ArgumentParser()
argp.add_argument("-d", "--dataset", required=True,
                  help="path to input TSDR Dataset")
argp.add_argument("-m", "--model", required=True,
                  help="path to Output Model")
argp.add_argument("-p", "--plot", type=str, default="plot.png",
                  help="path to Training History Plot")
parseArg = vars(argp.parse_args())

# initialize the number of epochs to train for, base learning rate,
# and batch size
No_Of_Epochs = 30
Initial_LR = 1e-3
Batch_Size = 64

# load the label names
TrafficSignLabelNames = open(
    "TSDR_System/TrafficSignNames.csv").read().strip().split("\n")[1:]
TrafficSignLabelNames = [l.split(",")[1] for l in TrafficSignLabelNames]

# Load and Pre-Processing of the Traffic Sign Data:
# Return the path to the training and testing CSV files
train_path_dir = os.path.sep.join(
    [parseArg["dataset"], "Train.csv"])
test_path_dir = os.path.sep.join([parseArg["dataset"], "Test.csv"])

# Load the training and testing data
print("[INFO] Loading training and testing data...")
(train_x_value, train_y_value) = load_tsdr_data(
    parseArg["dataset"], train_path_dir)
(test_x_value, test_y_value) = load_tsdr_data(
    parseArg["dataset"], test_path_dir)

# Scale data to the range of [0, 1]
train_x_value = train_x_value.astype("float32") / 255.0
test_x_value = test_x_value.astype("float32") / 255.0

# one-hot encode the training and testing labels
num_labels = len(np.unique(train_y_value))
train_y_value = to_categorical(train_y_value, num_labels)
test_y_value = to_categorical(test_y_value, num_labels)

# account for skew in the labeled data
TSDR_Class_Sum = train_y_value.sum(axis=0)
TSDR_Class_Weight = TSDR_Class_Sum.max() / TSDR_Class_Sum

# Constructing Image Generator For Data Augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

# Initialize The Optimizer and Compile the Model
print("[INFO] Compiling Model...")
opt_model = Adam(lr=Initial_LR, decay=Initial_LR / (No_Of_Epochs * 0.5))
compile_model = TSDR_CNN_Model.build(width=32, height=32, depth=3,
                                     classes=num_labels)
compile_model.compile(loss="categorical_crossentropy", optimizer=opt_model,
                      metrics=["accuracy"])

# Train the Network
print("[INFO] training network...")
H = compile_model.fit_generator(
    data_augmentation.flow(train_x_value, train_y_value, batch_size=BS),
    validation_data=(test_x_value, test_y_value),
    steps_per_epoch=train_x_value.shape[0] // BS,
    epochs=No_Of_Epochs,
    class_weight=TSDR_Class_Weight,
    verbose=1)

# Evaluate the Network
print("[INFO] evaluating network...")
CNN_Predictions = compile_model.predict(test_x_value, batch_size=BS)
print(classification_report(test_y_value.argmax(axis=1),
                            CNN_Predictions.argmax(axis=1), target_names=TrafficSignLabelNames))

# Save the CNN Model to Local Machine
print("[INFO] serializing network to '{}'...".format(args["model"]))
compile_model.save(args["model"])

# Plot the Training Loss and Accuracy
N = np.arange(0, No_Of_Epochs)
plpt.style.use("ggplot")
plpt.figure()
plpt.plot(N, H.history["loss"], label="train_loss")
plpt.plot(N, H.history["val_loss"], label="val_loss")
plpt.plot(N, H.history["accuracy"], label="train_acc")
plpt.plot(N, H.history["val_accuracy"], label="val_acc")
plpt.title("Training Loss and Accuracy on Dataset")
plpt.xlabel("Epoch #")
plpt.ylabel("Loss/Accuracy")
plpt.legend(loc="lower left")
plpt.savefig(args["plot"])
