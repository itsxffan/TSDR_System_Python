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

    print(data_rows[:3])


print("Saffan")
