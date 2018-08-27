# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:  # INPUT => CONV => RELU => CONV => RELU => POOL => CONV => RELU => CONV => RELU => POOL => FC => RELU => FC => SMAX
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation="relu", weightsPath=None):
        # initialize the model
        model = Sequential()

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)
        else:  # "channels last"
            inputShape = (imgRows, imgCols, numChannels)

        # define the first set of CONV => ACTIVATION => CONV => ACTIVATION => POOL layers with Dropout
        model.add(Conv2D(32, 3, padding="same", input_shape=inputShape))
        model.add(Activation(activation))
        model.add(Conv2D(32, 3, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # define the second set of CONV => ACTIVATION => CONV => ACTIVATION => POOL layers with Dropout
        model.add(Conv2D(64, 3, padding="same"))
        model.add(Activation(activation))
        model.add(Conv2D(64, 3, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # define the first FC => ACTIVATION layers with Dropout
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(activation))
        model.add(Dropout(0.5))

        # define the second FC layer
        model.add(Dense(numClasses))

        # lastly, define the soft-max classifier
        model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model
