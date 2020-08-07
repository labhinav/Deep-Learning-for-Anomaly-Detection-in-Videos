# Deep-Learning-for-Anomaly-Detection-in-Videos
This repository belongs to a team of 5 students from BITS-Pilani. The project was done as a part of the PS1 program, at CEERI-Pilani

This project aims to use unsupervised deep learning methods , to detect anomalies in videos. Our model architecture is based on Variational Autoencoder Generative Adversarial Networks (VAE-GANs). It consists of 2 parts, an autoencoder, and a discriminator. The model is trained adversarially, the autoencoder is trained to reconstruct videos well, and the discriminator is trained to classify real videos, and regenerated ones (from the autoencoder) , accurately.

As the autoencoder is trained on normal videos, it is unable to reconstruct videos containing anomalies accurately, and thus, the discriminator will be able to classify the video containing anomalies  as a regenerated video.

We trained and tested the model using the UCF-Crime dataset.

Link for Kaggle Notebook:
https://www.kaggle.com/abhinavlalwani/deep-learning-for-anomaly-detection-in-videos

