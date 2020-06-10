from tensorflow import keras
from tensorflow.keras.layers import (Input, Dense, Activation,
                                     BatchNormalization, Flatten,
                                     MaxPool3D, Conv3D)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model
from keras import Sequential

def create_discriminator_model():


    # not sure about the axis in batch norm
    # do we also add dropout after batchnorm/pooling?

    # Convolutional Layers
    # changed the no of filters
    model= Sequential()
    model.add(Conv3D(filters=48, kernel_size=(2, 2, 2), padding="same",input_shape=(16, 128, 128, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(filters=64, kernel_size=(2, 2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(filters=128, kernel_size=(2, 2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(filters=128, kernel_size=(2, 2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    # to add the 5th layer change the cap to 32 frames

    # X=Conv3D(filters=256,kernel_size=(2,2,2),padding="same")(X)
    # X=BatchNormalization()(X)
    # X=Activation('relu')(X)
    # X=MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(X)

    # Fully connected layers

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    # add batch norm to dense layer
    model.add(BatchNormalization())
    # activation done with loss fn
    # for numerical stability
    model.add(Dense(1, activation='sigmoid'))

    return model

if __name__ == "__main__":
    discriminator = create_discriminator_model()
    opt = keras.optimizers.Adam(lr=0.001)
    loss = BinaryCrossentropy()
    discriminator.compile(loss=loss,
                          optimizer=opt,
                          metrics=['accuracy'])
    print(discriminator.summary())
