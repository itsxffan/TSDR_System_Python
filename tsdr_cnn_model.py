# Author - Saffan Ahmed

# Import all the required packages.

# CNN model will be based upon the Sequential API.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D

# TSDR CNN Model Class


class TSDR_Model:
    @staticmethod
    def build_model(width, height, depth, classes):
        cnnModel = Sequential()
        inputShape = (height, width, depth)
        channelDim = -1
