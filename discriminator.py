from tensorflow import keras
from tensorflow.keras.layers import (Input, Dense, Activation,
                                     BatchNormalization, Flatten,
                                     MaxPool3D, Conv3D)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model


def create_discriminator_model():

    X_input = Input((16, 128, 128, 3))

    # not sure about the axis in batch norm
    # do we also add dropout after batchnorm/pooling?

    # Convolutional Layers
    # changed the no of filters
    X = Conv3D(filters=32, kernel_size=(2, 2, 2), padding="same")(X_input)
    X = BatchNormalization(axis=1)(X)
    X = Activation('relu')(X)
    X = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X)

    X = Conv3D(filters=64, kernel_size=(2, 2, 2), padding="same")(X)
    X = BatchNormalization(axis=1)(X)
    X = Activation('relu')(X)
    X = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X)

    X = Conv3D(filters=128, kernel_size=(2, 2, 2), padding="same")(X)
    X = BatchNormalization(axis=1)(X)
    X = Activation('relu')(X)
    X = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X)

    X = Conv3D(filters=128, kernel_size=(2, 2, 2), padding="same")(X)
    X = BatchNormalization(axis=1)(X)
    X = Activation('relu')(X)
    X = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(X)

    # to add the 5th layer change the cap to 32 frames

    # X=Conv3D(filters=256,kernel_size=(2,2,2),padding="same")(X)
    # X=BatchNormalization(axis=1)(X)
    # X=Activation('relu')(X)
    # X=MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(X)

    # Fully connected layers

    X = Flatten()(X)

    X = Dense(256, activation='relu')(X)
    # add batch norm to dense layer
    X = BatchNormalization()(X)
    # activation done with loss fn
    # for numerical stability
    X = Dense(1, activation=None)(X)

    model = Model(inputs=X_input, outputs=X, name="Discriminator")

    return model


if __name__ == "__main__":
    discriminator = create_discriminator_model()
    opt = keras.optimizers.Adam(lr=0.001)
    loss = BinaryCrossentropy(from_logits=True)
    discriminator.compile(loss=loss,
                          optimizer=opt,
                          metrics=['accuracy'])
    print(discriminator.summary())
