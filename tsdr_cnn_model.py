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
        shapeInput = (height, width, depth)
        channelDimension = -1

        # CNN Layer 1: CONV >> RELU >> BN >> POOL
        model.add(Conv2D(8, (5, 5), padding="same", input_shape=shapeInput))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(MaxPooling2D(pool_size=2, 2))

        # First Set: ((CONV >> RELU >> CONV >> RELU)) * 2 >> POOL
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second Set: ((CONV >> RELU >> CONV >> RELU)) * 2 >> POOL
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
