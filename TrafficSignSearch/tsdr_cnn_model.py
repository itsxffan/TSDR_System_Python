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
class TSDR_CNN_Model:
    @staticmethod
    def build_model(width, height, depth, classes):
        shapeInput = (height, width, depth)
        cnnModel = Sequential()
        channelDimension = -1

        # CNN Layer 1: CONV >> RELU (default acitivation function that shoots the neuron) >> BN >> POOL
        cnnModel.add(Conv2D(8, (5, 5), padding="same", input_shape=shapeInput))
        cnnModel.add(Activation("relu"))
        cnnModel.add(BatchNormalization(axis=channelDimension))
        cnnModel.add(MaxPooling2D(pool_size=(2, 2)))

        # First Set: ((CONV >> RELU (default acitivation function that shoots the neuron) >> CONV >> RELU)) * 2 >> POOL
        cnnModel.add(Conv2D(16, (3, 3), padding="same"))
        cnnModel.add(Activation("relu"))
        cnnModel.add(BatchNormalization(axis=chanDim))
        cnnModel.add(Conv2D(16, (3, 3), padding="same"))
        cnnModel.add(Activation("relu"))
        cnnModel.add(BatchNormalization(axis=chanDim))
        cnnModel.add(MaxPooling2D(pool_size=(2, 2)))

        # Second Set: ((CONV >> RELU (default acitivation function that shoots the neuron) >> CONV >> RELU)) * 2 >> POOL
        cnnModel.add(Conv2D(32, (3, 3), padding="same"))
        cnnModel.add(Activation("relu"))
        cnnModel.add(BatchNormalization(axis=chanDim))
        cnnModel.add(Conv2D(32, (3, 3), padding="same"))
        cnnModel.add(Activation("relu"))
        cnnModel.add(BatchNormalization(axis=chanDim))
        cnnModel.add(MaxPooling2D(pool_size=(2, 2)))

        # First Set of Fully Connected >> RELU (default acitivation function that shoots the neuron) layers
        cnnModel.add(Flatten())
        cnnModel.add(Dense(128))
        cnnModel.add(Activation("relu"))
        cnnModel.add(BatchNormalization())
        cnnModel.add(Dropout(0.5))

        # Second Set of Fully Connected >> RELU (default acitivation function that shoots the neuron) layers
        cnnModel.add(Flatten())
        cnnModel.add(Dense(128))
        cnnModel.add(Activation("relu"))
        cnnModel.add(BatchNormalization())
        cnnModel.add(Dropout(0.5))

        # SoftMax Classfifiers
        cnnModel.add(Dense(classes))
        cnnModel.add(Activation("softmax"))

        # Return the Model: CNN Network Architecture that has been constructed
        return cnnModel
