import pandas as pd
import numpy as np
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

def encoder(img_shape = (224,224,3)):

    latent_dim = 2
    model_colorization = tf.keras.models.load_model('models/my_model_colorization.h5')
    # model_colorization.summary()
    model_colorization.trainable = False


    model_ = tf.keras.models.Model(model_colorization.input,model_colorization.layers[-14].output)
    input_img = tf.keras.layers.Input(shape=img_shape)
    model = model_(input_img)
    x = tf.keras.layers.Dense(120, activation='sigmoid')(model)

    z_mean = tf.keras.layers.Dense(latent_dim, activation='linear')(x)
    z_var = tf.keras.layers.Dense(latent_dim, activation='linear')(x)
    z = Sampling()([z_mean, z_var])
    encoder = tf.keras.Model(input_img, [z_mean, z_var, z], name="encoder")

    encoder.summary()

    return encoder

def decoder():
    latent_dim = 2
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(61 * 61 * 12, activation="relu")(latent_inputs)
    x = tf.keras.layers.Reshape((61, 61, 12))(x)
    x = tf.keras.layers.Conv2DTranspose(18, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(24, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
def train(encoder, decoder):
    with open('data/processed_fruit_images_colorizationTest.pkl', 'rb') as f:
        output_dict_colorization = pickle.load(f)
    with open('data/processed_fruit_images_realTest.pkl', 'rb') as f:
        output_real = pickle.load(f)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    train_x = np.asarray(output_dict_colorization['X'])
    train_y = np.asarray(output_real['real_img'])
    vae.fit(train_x, train_y, epochs=30, batch_size=128)


if __name__ == '__main__':
    import os
    os.chdir('..')
    encoder_model = encoder()
    decoder = decoder()
    train(encoder, decoder)
