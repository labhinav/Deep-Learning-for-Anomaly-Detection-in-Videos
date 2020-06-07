from autoencoder import AutoEncoderModel,custom_loss
from discriminator import create_discriminator_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv3D, LeakyReLU, Conv3DTranspose
from tensorflow.keras.layers import AveragePooling2D, MaxPool3D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class GAN():
    def __init__(self):
        self.image_shape=(16,128,128,3)
        opt=keras.optimizers.Adam(lr=0.001)
        #Build and compile the discriminator
        self.discriminator=create_discriminator_model()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        #Build and compile the generator
        self.generator=AutoEncoderModel()
        self.generator.compile(loss=custom_loss,optimizer=opt,metrics=['accuracy'])

        #the generator takes a video as input and generates a modified video
        z=Input(shape=self.image_shape)
        modified_vid=self.generator(z)
        # For the combined model we will only train the generator
        self.discriminator.trainable=False
         # The valid takes generated images as input and determines validity
        valid = self.discriminator(modified_vid)
        # The combined model  (stacked generator and discriminator) takes
        # video segment as input => generates modified video => determines validity
        self.combined = Model(z, [valid,modified_vid])
        # we need multiple losses as we need the normal loss function+reconstruction error
        # not sure if this is the right way to implement it
        lossWeights = {"valid": 1.0, "modified_vid": 1.0} #lossweights can be changed later
        self.combined.compile(loss={'valid':'binary_crossentropy','modified_vid':custom_loss}, optimizer=opt,loss_weights=lossWeights)

    def train(self,epochs,mini_batch_size,input_videos):
        #this function will need to be added later
        minibatches=construct_minibatches(input_videos,mini_batch_size)
        for epoch in range(epochs):
            for minibatch in minibatches:
                # ---------------------
                #  Train Discriminator
                # ---------------------
                gen_vids=self.generator.predict(minibatch)
                d_loss_real=self.discriminator.train_on_batch(minibatch,np.ones((mini_batch_size,1)))
                d_loss_fake=self.discriminator.train_on_batch(gen_vids,np.zeros((mini_batch_size,1)))
                d_loss=0.5*np.add(d_loss_real,d_loss_fake)
                # ---------------------
                #  Train Generator
                # ---------------------
                # The generator wants the discriminator to label the generated samples as valid (ones)
                valid_y = np.array([1] * mini_batch_size)
                # Train the generator
                g_loss = self.combined.train_on_batch(minibatch, valid_y)

                 # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

if __name__ == '__main__':
    gan = GAN()
    print(gan.combined.summary())
    print(gan.discriminator.summary())
    print(gan.generator.summary())


