# Import necessary components to build AlexNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


class AlexNet:
    # INPUT => (CONV => RELU => POOL) X 3 => (CONV => RELU) X 2 => POOL => (FC => RELU) X 2 => FC => SMAX
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, l2_reg=0., weightsPath=None):
        # img_shape=(224, 224, 3), n_classes=10, l2_reg=0.,weights=None

        # initialize model
        model = Sequential()

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)
        else:  # "channels last"
            inputShape = (imgRows, imgCols, numChannels)

        # layer 1, convolution units
        model.add(Conv2D(96, (11, 11), padding='same', input_shape=inputShape,
                         kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # layer 2
        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # layer 5
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(1024, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # layer 6, flatten the conv network
        model.add(Flatten())
        model.add(Dense(3072))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # layer 7
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # layer 8, define the soft-max classifier
        model.add(Dense(numClasses))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model
