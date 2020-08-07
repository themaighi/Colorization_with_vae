import pandas as pd
import numpy as np
import random
import cv2
from keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import pickle
from tensorflow import keras


class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_var))
        return z_mean + tf.exp(z_var) * epsilon


def vae_model(img_shape = (224,224,3), latent_dim=10):
    model_colorization = tf.keras.models.load_model('models/my_model_colorization.h5')
    # model_colorization.summary()
    model_colorization.trainable = False

    model_ = tf.keras.models.Model(model_colorization.input, model_colorization.layers[-14].output)
    input_img = tf.keras.layers.Input(shape=img_shape)
    model = model_(input_img)
    x = tf.keras.layers.Dense(120, activation='sigmoid')(model)
    x = tf.keras.layers.Dense(20, activation='sigmoid')(x)
    x = tf.keras.layers.Flatten()(x)
    z_mean = tf.keras.layers.Dense(latent_dim, activation='linear')(x)
    z_var = tf.keras.layers.Dense(latent_dim, activation='linear')(x)
    z = Sampling()([z_mean, z_var])

    x = tf.keras.layers.Dense(56 * 56 * 12, activation="relu")(z)
    x = tf.keras.layers.Reshape((56, 56, 12))(x)
    x = tf.keras.layers.Conv2DTranspose(18, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(24, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(2, 3, activation="sigmoid", padding="same")(x)
    vae = tf.keras.Model(input_img, decoder_outputs, name="vae")
    vae.summary()

    return vae


def load_data(prop = 0.2):
    list_files = os.listdir('data/fruit/Test/colorization')

    x_test = []
    y_test = []
    for i in list_files: # i = list_files[0]
        with open('data/fruit/Test/colorization/' + i, 'rb') as f:
            image_dict = pickle.load(f)
        x_test.extend(random.sample(image_dict['X'], int(np.ceil(len(image_dict['X']) * prop))))
        y_test.extend(random.sample(image_dict['y'], int(np.ceil(len(image_dict['y']) * prop))))

    x_test = np.asarray(x_test)
    y_test = np.concatenate([x_test, y_test], axis=3)

    list_files = os.listdir('data/fruit/Training/colorization')
    x_train_all = []
    y_train_all = []
    for i in list_files: # i = list_files[0]
        with open('data/fruit/Training/colorization/' + i, 'rb') as f:
            image_dict = pickle.load(f)
        x_train_all.extend(random.sample(image_dict['X'], int(np.ceil(len(image_dict['X']) * prop))))
        y_train_all.extend(random.sample(image_dict['y'], int(np.ceil(len(image_dict['y']) * prop))))


    x_train = np.asarray(x_train_all)
    y_train = np.concatenate([x_train, y_train_all], axis=3)

    x_train, y_train, x_val, y_val = train_test_split(x_train, y_train, test_size=0.2)

    return x_train, y_train, x_val, y_val, x_test, y_test

def train(model, x_train, y_train):
    # Q(z|X) -- encoder

    def vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mae(y_true, y_pred)
        )
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(z_var) + K.square(z_mean) - 1. - z_var, axis=1)

        return reconstruction_loss + kl


    model.compile(optimizer='adam', loss=vae_loss)
    history = model.fit(np.tile(x_train, [1,1,1,3])/255, y_train/255, batch_size=30, epochs=10)

    ## Nice things to show
    ## - Training error in training set
    ## - Validation Error
    ## - Test error after every epoch?
    ## - Show how does the prediction changes across iterations
    ## - Show for different latent variable what the prediction would be (by setting all the other to the mean


    return model, history
#
# def predict(model, x_test, y_test):
#     prediction_model = model.predict(x_test)



if __name__ == '__main__':
    import os
    os.chdir('..')
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    model = vae_model()
    train(model=model, x_train=x_train, x_test=x_test)


