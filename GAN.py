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
from tf2_multi_preprocessing import build_dataset
from tf2_test_data_loader import build_test_dataset
from cv2 import VideoWriter, VideoWriter_fourcc
class GAN():
    def __init__(self, mini_batch_size):
        self.image_shape=(16,128,128,3)
        learning_rate=0.03
        opt=keras.optimizers.Adam(lr=learning_rate)
        opt1=keras.optimizers.Adam(lr=learning_rate)
        opt_slow=keras.optimizers.Adam(lr=1)
        #Build and compile the discriminator
        self.discriminator=create_discriminator_model()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy',tf.keras.metrics.TruePositives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalseNegatives()])
        #Build and compile the generator
        self.generator=AutoEncoderModel()
        self.generator.compile(loss='mse',optimizer=opt_slow)

        #the generator takes a video as input and generates a modified video
        z = Input(shape=(self.image_shape))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=opt1,metrics=['accuracy'])
        self.dir_path = '/kaggle/input/ucf-crime-training-subset/tfrecords2/'
        self.ds = build_dataset(self.dir_path, batch_size=mini_batch_size,file_buffer=512*1024)

    def train(self,epochs,mini_batch_size):
        #this function will need to be added later
        tf.summary.trace_off()
        for epoch in range(epochs):
            d_loss_sum=tf.zeros(6)
            reconstruct_error_sum=0
            g_loss_sum=tf.zeros(2)
            no_of_minibatches=0
            for minibatch,labels in self.ds:
                # ---------------------
                #  Train Discriminator
                # ---------------------
                #normalize inputs
                no_of_minibatches+=1
                minibatch=tf.cast(tf.math.divide(minibatch,255), tf.float32)
                gen_vids=self.generator.predict(minibatch)
                #might have to combine these to improve batch norm
                d_loss_real=self.discriminator.train_on_batch(minibatch,tf.ones((mini_batch_size,1)))
                d_loss_fake=self.discriminator.train_on_batch(gen_vids,tf.zeros((mini_batch_size,1)))
                d_loss=0.5*tf.math.add(d_loss_real,d_loss_fake)
                # ---------------------
                #  Train Generator
                # ---------------------
                # The generator wants the discriminator to label the generated samples as valid (ones)
                valid_y = tf.ones((mini_batch_size,1))
                # Train the generator
                g_loss = self.combined.train_on_batch(minibatch,valid_y)
                reconstruct_error=self.generator.train_on_batch(minibatch,minibatch)
                d_loss_sum+=d_loss
                g_loss_sum+=g_loss
                reconstruct_error_sum+=reconstruct_error
            print(no_of_minibatches)
            self.combined.save_weights('/kaggle/working/weights_epoch%d' %(epoch))
            g_loss=g_loss_sum/no_of_minibatches
            d_loss=d_loss_sum/no_of_minibatches
            reconstruct_error=reconstruct_error_sum/no_of_minibatches
            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, accuracy %.2f%% from which %f is combined loss and %f is reconstruction loss]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]+reconstruct_error,g_loss[1]*100,g_loss[0],reconstruct_error))
        tf.summary.trace_on()

    def test(self,dev_set_path,mini_batch_size):
        dev_set=build_test_dataset(dev_set_path,batch_size=mini_batch_size,file_buffer=500*1024)
        no_of_minibatches=0
        ans_final=tf.zeros(6)
        for minibatch,labels in dev_set:
            no_of_minibatches+=1
            ans=self.discriminator.test_on_batch(minibatch,(labels==0),reset_metrics=False)
            ans_final=ans
        print(no_of_minibatches,ans_final[0],ans_final[1],ans_final[2],ans_final[3],ans_final[4],ans_final[5])
    
    def test_real_vs_fake(self,dev_set_path,mini_batch_size):
        dev_set=build_dataset(dev_set_path,batch_size=mini_batch_size,file_buffer=500*1024)
        ans_final=tf.zeros(6)
        no_of_minibatches=0
        for minibatch,labels in dev_set:
            no_of_minibatches+=1
            ans=self.discriminator.test_on_batch(minibatch,labels,reset_metrics=False)
            fake_vals=np.random.random((mini_batch_size,16,128,128,3))
            ans=self.discriminator.test_on_batch(fake_vals,tf.zeros((mini_batch_size,1)),reset_metrics=False)
            ans_final=ans
        print(no_of_minibatches,ans_final[0],ans_final[1],ans_final[2],ans_final[3],ans_final[4],ans_final[5])
    
    def visualise_autoencoder_outputs(self,no_of_minibatches):
        fourcc = VideoWriter_fourcc(*'MP42') #some code required for VideoWriter
        video = VideoWriter('/kaggle/working/reconstructed_video.avi', fourcc, float(24), (128, 128)) #creates video to store 1st segment
        for i in range(no_of_minibatches):
            inp=np.load("../input/tom-and-jerry-clips/minibatches/minibatch%d.npz" % (i))
            inp=inp['arr_0']
            gen_vids=self.generator.predict(inp)
            gen_vids*=255
            for j in range(16):
                for k in range(16):
                    frame = np.uint8(gen_vids[j][k])
                    video.write(frame)

if __name__ == '__main__':
    gan = GAN()
    gan.combined.load_weights('../input/saved-models/weights_epoch35.h5')
    print(gan.combined.summary())
    print(gan.discriminator.summary())
    print(gan.generator.summary())


