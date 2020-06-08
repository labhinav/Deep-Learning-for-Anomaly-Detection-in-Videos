import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv3D, LeakyReLU, Conv3DTranspose
from tensorflow.keras.layers import AveragePooling2D, MaxPool3D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def AutoEncoderModel():
    #encoder
    X_input=Input((16,128,128,3))

    X=Conv3D(32,3,padding='same')(X_input)
    X=BatchNormalization()(X)
    X=LeakyReLU()(X)
    X=MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),padding='valid')(X)
    #current shape is 8x64x64x32
    X=Conv3D(48,3,padding='same')(X)
    X=BatchNormalization()(X)
    X=LeakyReLU()(X)
    X=MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),padding='valid')(X)
    #current shape is 4x32x32x48
    X=Conv3D(64,3,padding='same')(X)
    X=BatchNormalization()(X)
    X=LeakyReLU()(X)
    X=MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),padding='valid')(X)
    #current shape is 2x16x16x64
    X=Conv3D(64,3,padding='same')(X)
    X=BatchNormalization()(X)
    X=LeakyReLU()(X)
    X=MaxPool3D(pool_size=(2,2,2),strides=(1,1,1),padding='same')(X)
    #current shape is 2x16x16x64
    #decoder

    X=Conv3DTranspose(48,2,strides=(2,2,2),padding='valid')(X)
    X=BatchNormalization()(X)
    X=LeakyReLU()(X)
    #current shape is 4x32x32x48
    X=Conv3DTranspose(32,2,strides=(2,2,2),padding='valid')(X)
    X=BatchNormalization()(X)
    X=LeakyReLU()(X)
    #current shape is 8x64x64x32
    X=Conv3DTranspose(32,2,strides=(2,2,2),padding='valid')(X)
    X=BatchNormalization()(X)
    X=LeakyReLU()(X)
    #current shape is 16x128x128x32
    X=Conv3D(3,3,strides=(1,1,1),padding='same')(X)
    X=Activation('sigmoid')(X)
    #current shape is 16x128x128x3

    model = Model(inputs=X_input,outputs=X,name='AutoEncoderModel')
    return model

def custom_loss(new, original):
  reconstruction_error =K.mean(K.square(new-original))
  return reconstruction_error

if __name__=="__main__":
    autoEncoderModel=AutoEncoderModel()
    opt = keras.optimizers.Adam(lr=0.001)
    autoEncoderModel.compile(loss=custom_loss,optimizer=opt,metrics=['accuracy'])
    print(autoEncoderModel.summary())
