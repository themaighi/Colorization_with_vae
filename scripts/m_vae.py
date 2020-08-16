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
import matplotlib.pyplot as plt
# Add the labels in such a way that we are able to create the plot for the different scatter plots


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
    decoder_outputs = tf.keras.layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    vae = tf.keras.Model(input_img, decoder_outputs, name="vae")
    vae.summary()

    def vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mae(y_true, y_pred)
        )
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = 0.5 * K.sum(K.exp(z_var) + K.square(z_mean) - 1. - z_var, axis=1)

        return reconstruction_loss + kl

    vae.compile(optimizer='adam', loss=vae_loss)
    encoder = tf.keras.Model(input_img, z_mean, name = 'encoder')
    # generator = tf.keras.Model(z, decoder_outputs, name = 'decoder')

    return vae, encoder


def load_data(prop = 0.05):
    list_files = os.listdir('data/fruit/Test/colorization')

    x_test = []
    y_test = []
    for i in list_files: # i = list_files[-2]
        with open('data/fruit/Test/colorization/' + i, 'rb') as f:
            image_dict = pickle.load(f)
        mask = random.sample(range(len(image_dict['X'])), int(np.ceil(len(image_dict['X']) * prop)))
        try:
            x_test.extend([image_dict['X'][j] for j in mask])
            y_test.extend([image_dict['y'][j] for j in mask])
        except:
            print(i)
            print(len(image_dict['X']))
            print(mask)

    x_test = np.asarray(x_test)
    y_test = np.concatenate([x_test, y_test], axis=3)

    list_files = os.listdir('data/fruit/Training/colorization')
    x_train_all = []
    y_train_all = []
    for i in list_files: # i = list_files[0]
        with open('data/fruit/Training/colorization/' + i, 'rb') as f:
            image_dict = pickle.load(f)

        mask = random.sample(range(len(image_dict['X'])), int(np.ceil(len(image_dict['X']) * prop)))
        try:
            x_train_all.extend([image_dict['X'][j] for j in mask])
            y_train_all.extend([image_dict['y'][j] for j in mask])
        except:
            print(i)
            print(len(image_dict['X']))
            print(mask)


    x_train = np.asarray(x_train_all)
    y_train = np.concatenate([np.asarray(x_train_all), y_train_all], axis=3)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    return x_train, y_train, x_val, y_val, x_test, y_test

def train(model, x_train, y_train, x_val, y_val, encoder):
    # Q(z|X) -- encoder


    class PredictCallback(tf.keras.callbacks.Callback):

        def __init__(self, x_dt):
            self.x_dt = x_dt

        def on_epoch_end(self, epoch, logs=None):
            # if not hasattr(logs, 'epoch_prediction'):
            #     print('creating the epoch prediction log')
            #     logs['epoch_prediction'] = list()
            logs['epoch_prediction'] = self.model.predict(np.tile(self.x_dt, [1,1,1,3])/255)

    epochs_num = 10
    batch_size_val = 30

    mask = random.sample(range(len(x_val)), 100)
    x_val_selected = [x_val[i] for i in mask]
    y_val_selected = [y_val[i] for i in mask]

    history = model.fit(np.tile(x_train, [1,1,1,3])/255, y_train/255, batch_size=batch_size_val, epochs=epochs_num,
                        validation_data=(np.tile(x_val, [1,1,1,3])/255, y_val/255), callbacks=[PredictCallback(x_dt=x_val_selected)])

    # img_real = np.concatenate([np.asarray(x_val_selected), np.asarray(y_val_selected)], axis=3)
    img_real = y_val_selected
    os.makedirs('models/fruit/vae/results/' + 'real' + '/', exist_ok=True)
    for img_n in range(len(img_real)):
        cv2.imwrite('models/fruit/vae/results/' + 'real' + '/real_img_' + str(img_n) + '.png',
                    img_real[img_n])

    ## Reconpose the images
    for ep in range(len(history.history['epoch_prediction'])):
        os.makedirs('models/fruit/vae/results/' + str(ep) + '/', exist_ok=True)
        for img_n in range(len(history.history['epoch_prediction'][ep])):
            img_save = np.concatenate([np.asarray(x_val_selected), np.asarray(history.history['epoch_prediction'][ep][:,:,:,1:]) * 255], axis=3)
            cv2.imwrite('models/fruit/vae/results/' + str(ep) +'/reconstructed_img_' + str(img_n) + '.png', img_save[img_n])
            cv2.imwrite('models/fruit/vae/results/' + str(ep) +'/prediction_img_' + str(img_n) + '.png', history.history['epoch_prediction'][ep][img_n] * 255)


    ## Printing loss

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.savefig('models/fruit/vae/results/loss.png')
    ## Nice things to show
    ## - Show for different latent variable what the prediction would be (by setting all the other to the mean
    ## - Pass the labels of the actual imageso that we can understand if the latent variable are expressing something



    # model.save('models/fruit/vae/model_10_epochs.h5')
    ## Function to get the real images
    n_rows = 5
    n_cols = 10
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):

            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            image = cv2.cvtColor(img_real[index], cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.axis('off')
    plt.savefig('example_image_dataset.png')

    x_test_encoded = encoder.predict(np.tile(x_val_selected, [1,1,1,3])/255)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 8], x_test_encoded[:, 9])
    # plt.colorbar()
    plt.show()

    return model, history, x_test_encoded, labels_encoder
#
# def predict(model, x_test, y_test):
#     prediction_model = model.predict(x_test)

def create_plots_paper():
    list_directory = os.listdir('models/fruit/vae/results')
    list_directory = [l for l in list_directory if '.' not in l]

    img_select = [3, 20, 40]
    img_print = list()
    for i in img_select:
        for l in list_directory:
            images = os.listdir('models/fruit/vae/results/' + l)
            images = [img for img in images if ('reconstructed' in img) | ('real' in img)]
            read_images = cv2.imread('models/fruit/vae/results/' + l + '/' + images[i])
            img_print.append(read_images)

    n_rows = 3
    n_cols = 11
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            image = cv2.cvtColor(img_print[index], cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.axis('off')

    plt.savefig('models/fruit/vae/image_different_epochs.png')

    return 5



if __name__ == '__main__':
    import os
    os.chdir('..')
    # x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    # model, encoder = vae_model(latent_dim=30)
    # train(model=model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, encoder=encoder)
    create_plots_paper()

