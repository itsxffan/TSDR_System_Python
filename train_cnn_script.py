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
    ts_label = []

    data_rows = open(csvDir).read().strip().split("\n")[1:]
    random.shuffle(data_rows)

    # loop over the rows of the CSV file
    for (x, data_row) in enumerate(data_row):
          # check to see if we should show a status update
          if x > 0 and x % 1000 == 0:
              print("[INFO] processed {} total images".format(i))
          # split the row into components and then grab the class ID
          # and image path
          (ts_label, imageDir) = data_row.strip().split(",")[-2:]
          # Derive the full directory of the image file and load it
          imageDir = os.path.sep.join([baseDir, imageDir])
          trafficSignImage = io.imread(imagePath)

          # resize the image to be 32x32 pixels, ignoring aspect ratio,
          # and then perform Contrast Limited Adaptive Histogram
          # Equalization (CLAHE)
          trafficSignImage = transform.resize(trafficSignImage, (32, 32))
          trafficSignImage = exposure.equalize_adapthist(
              trafficSignImage, clip_limit=0.1)
          # update the list of data and labels, respectively
          ts_data.append(trafficSignImage)
          ts_label.append(int(ts_label))

      # Convert the Image Data (Pixels) and Image Labels into Numpy Array's
      ts_data = np.array(ts_data)
      ts_label = np.array(ts_label)

      # Return a tuple of the Image Data (Pixels) and Image Labels
      return (ts_data, ts_label)

      # construct the argument parser and parse the arguments
      argp = ap.ArgumentParser()
      argp.add_argument("-d", "--dataset", required=True,
        help="path to input TSDR Dataset")
      argp.add_argument("-m", "--model", required=True,
        help="path to Output Model")
      argp.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to Training History Plot")
      args = vars(argp.parse_args())
      

