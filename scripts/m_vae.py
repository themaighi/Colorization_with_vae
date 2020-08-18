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
import datetime
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

    decoder_layer1 = tf.keras.layers.Dense(56 * 56 * 12, activation="relu")
    decoder_layer2 = tf.keras.layers.Reshape((56, 56, 12))
    decoder_layer3 = tf.keras.layers.Conv2DTranspose(18, 3, activation="relu", strides=2, padding="same")
    decoder_layer4 = tf.keras.layers.Conv2DTranspose(24, 3, activation="relu", strides=2, padding="same")
    decoder_outputs = tf.keras.layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")

    layer1_decoder = decoder_layer1(z)
    layer2_decoder = decoder_layer2(layer1_decoder)
    layer3_decoder = decoder_layer3(layer2_decoder)
    layer4_decoder = decoder_layer4(layer3_decoder)
    output_decoder = decoder_outputs(layer4_decoder)

    vae = tf.keras.Model(input_img, output_decoder, name="vae")
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

    encoder = tf.keras.Model(input_img, z_mean, name = 'encoder')
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
    layer1_generator = decoder_layer1(decoder_input)
    layer2_generator = decoder_layer2(layer1_generator)
    layer3_generator = decoder_layer3(layer2_generator)
    layer4_generator = decoder_layer4(layer3_generator)
    output_generator = decoder_outputs(layer4_generator)

    generator = tf.keras.Model(decoder_input, output_generator, name = 'decoder')
    vae.compile(optimizer='adam', loss=vae_loss)

    return vae, encoder, generator, model_colorization


def load_data(prop = 0.3):
    list_files = os.listdir('data/fruit/Test/colorization')

    x_test = []
    y_test = []
    labels_test = []
    for i in list_files: # i = list_files[-2]
        with open('data/fruit/Test/colorization/' + i, 'rb') as f:
            image_dict = pickle.load(f)
        mask = random.sample(range(len(image_dict['X'])), int(np.ceil(len(image_dict['X']) * prop)))
        try:
            x_test.extend([image_dict['X'][j] for j in mask])
            y_test.extend([image_dict['y'][j] for j in mask])
            labels_test.extend([i.replace('.pkl', '').split('_')[-1].split(' ')[0] for j in mask])
        except:
            print(i)
            print(len(image_dict['X']))
            print(mask)

    x_test = np.asarray(x_test)
    y_test = np.concatenate([x_test, y_test], axis=3)
    y_test = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_Lab2BGR) for img in y_test]
    x_test = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_Lab2BGR)[0,:,:] for img in y_test]

    list_files = os.listdir('data/fruit/Training/colorization')
    x_train_all = []
    y_train_all = []
    labels_train = []
    for i in list_files: # i = list_files[0]
        with open('data/fruit/Training/colorization/' + i, 'rb') as f:
            image_dict = pickle.load(f)

        mask = random.sample(range(len(image_dict['X'])), int(np.ceil(len(image_dict['X']) * prop)))
        try:
            x_train_all.extend([image_dict['X'][j] for j in mask])
            y_train_all.extend([image_dict['y'][j] for j in mask])
            labels_train.extend([i.replace('.pkl', '').split('_')[-1].split(' ')[0] for j in mask])
        except:
            print(i)
            print(len(image_dict['X']))
            print(mask)


    x_train = np.asarray(x_train_all)
    y_train = np.concatenate([np.asarray(x_train_all), y_train_all], axis=3)
    y_train = np.asarray([cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2Lab) for img in y_train])
    x_train = np.asarray([img[:,:,0:1] for img in y_train])

    x_train, x_val, y_train, y_val, labels_train, labels_val = train_test_split(x_train, y_train,labels_train, test_size=0.2)

    return x_train, y_train, x_val, y_val, x_test, y_test, labels_train, labels_val

def train(model, x_train, y_train, x_val, y_val):
    # Q(z|X) -- encoder


    class PredictCallback(tf.keras.callbacks.Callback):

        def __init__(self, x_dt):
            self.x_dt = x_dt

        def on_epoch_end(self, epoch, logs=None):
            # if not hasattr(logs, 'epoch_prediction'):
            #     print('creating the epoch prediction log')
            #     logs['epoch_prediction'] = list()
            logs['epoch_prediction'] = self.model.predict(np.tile(self.x_dt, [1,1,1,3])/255)

    epochs_num = 1
    batch_size_val = 30
    time_experiment = str(datetime.datetime.today()).replace(':', '').replace(' ', '_').split('.')[0]


    mask = random.sample(range(len(x_val)), 100)
    x_val_selected = [x_val[i] for i in mask]
    y_val_selected = [y_val[i] for i in mask]

    history = model.fit(np.tile(x_train, [1,1,1,3])/255, y_train/255, batch_size=batch_size_val, epochs=epochs_num,
                        validation_data=(np.tile(x_val, [1,1,1,3])/255, y_val/255), callbacks=[PredictCallback(x_dt=x_val_selected)])

    # img_real = np.concatenate([np.asarray(x_val_selected), np.asarray(y_val_selected)], axis=3)
    img_real = y_val_selected
    os.makedirs('models/fruit/vae/results/' + time_experiment + '/real' + '/', exist_ok=True)
    for img_n in range(len(img_real)):
        cv2.imwrite('models/fruit/vae/results/' + time_experiment + '/real' + '/real_img_' + str(img_n) + '.png',
                    cv2.cvtColor(img_real[img_n], cv2.COLOR_Lab2BGR))

    ## Reconpose the images
    for ep in range(len(history.history['epoch_prediction'])):
        os.makedirs('models/fruit/vae/results/' + time_experiment + '/' + str(ep) + '/', exist_ok=True)
        for img_n in range(len(history.history['epoch_prediction'][ep])):
            img_save = np.concatenate([np.asarray(y_val_selected)[:,:,:,0:1], np.asarray(history.history['epoch_prediction'][ep][:,:,:,1:]) * 255], axis=3)
            cv2.imwrite('models/fruit/vae/results/'+ time_experiment + '/' + str(ep) +'/reconstructed_img_' + str(img_n) + '.png',
                        cv2.cvtColor(img_save[img_n].astype(np.uint8), cv2.COLOR_Lab2BGR))
            cv2.imwrite('models/fruit/vae/results/'+ time_experiment + '/' + str(ep) +'/prediction_img_' + str(img_n) + '.png', cv2.cvtColor((history.history['epoch_prediction'][ep][img_n] * 255).astype(np.uint8), cv2.COLOR_Lab2BGR))


    ## Printing loss

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.savefig('models/fruit/vae/results/'+ time_experiment +'/loss.png')
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


    return model, history, time_experiment
#
# def predict(model, x_test, y_test):
#     prediction_model = model.predict(x_test)

def create_plots_paper(x_val, label_val, model_colorization, time_experiment, encoder, generator):
    list_directory = os.listdir('models/fruit/vae/results/' + time_experiment)
    list_directory = [l for l in list_directory if '.' not in l]

    img_select = [3, 20, 40, 70]
    img_print = list()
    for i in img_select:
        for l in list_directory:
            images = os.listdir('models/fruit/vae/results/'+ time_experiment + '/' + l)
            images = [img for img in images if ('reconstructed' in img) | ('real' in img)]
            read_images = cv2.imread('models/fruit/vae/results/' + time_experiment + '/' + l + '/' + images[i])
            if 'real' in images[0]:
                img_real = read_images
                read_images = cv2.cvtColor(read_images.astype(np.uint8), cv2.COLOR_BGR2RGB)
            else:
                read_images = read_images.astype(np.uint8)
            img_print.append(read_images)


        prediction_img = model_colorization.predict(np.tile(np.reshape(img_real[:,:,0],(1,img_real[:,:,0].shape[0],img_real[:,:,0].shape[1], 1)), [1, 1, 1, 3])/255)
        prediction_img[0] = prediction_img[0] * 255
        prediction_img[0] = (np.where(prediction_img[0] > 255, 255, prediction_img[0])).astype(np.uint8)

        img_real_lab = cv2.cvtColor(img_real, cv2.COLOR_BGR2Lab)

        img_print.append(cv2.cvtColor(np.concatenate([np.reshape(img_real_lab[:,:,0],(1, img_real_lab[:,:,0].shape[0],img_real_lab[:,:,0].shape[1], 1)),
                                         prediction_img[0]], axis=3)[0,:,:,:].astype(np.uint8), cv2.COLOR_Lab2BGR))


    n_rows = len(img_select)
    n_cols = len(list_directory) + 1
    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            # image = cv2.cvtColor(img_print[index], cv2.COLOR_Lab2BGR)
            plt.imshow(img_print[index])
            plt.axis('off')

    plt.savefig('models/fruit/vae/results/' + time_experiment +'/image_different_epochs.png')

    color_dictionary = dict(zip(set(label_val), range(len(set(label_val)))))

    # Understand which one is the dimention with the highest variance and the second highest variance
    x_test_encoded = encoder.predict(np.tile(x_val, [1,1,1,3])/255)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, np.argsort(np.var(x_test_encoded, axis=0))[-1]],
                x_test_encoded[:, np.argsort(np.var(x_test_encoded, axis=0))[-2]], c=[color_dictionary[i] for i in label_val])
    plt.colorbar()
    plt.show()
    plt.savefig('models/fruit/vae/results/' + time_experiment + '/image_scatterplot.png')
    import yaml
    with open('models/fruit/vae/results/' + time_experiment + '/image_labels.yaml', 'w') as f:
        yaml.dump(color_dictionary, f)

    # Let's see the figure for different values of the latent variables
    n = 15  # figure with 15x15 digits
    digit_size = 224
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    # we will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    vector_x = np.array([0] * x_test_encoded.shape[1])

    x_num = np.argsort(np.var(x_test_encoded, axis=0))[-1]
    y_num = np.argsort(np.var(x_test_encoded, axis=0))[-2]


    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            epsilon_std = np.random.normal(size=len(vector_x))
            vector_x[x_num] = xi
            vector_x[y_num] = yi

            z_sample = np.array([vector_x]) * epsilon_std
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, 3) * 255
            digit[digit>255] = 255
            digit[digit < 0] = 0
            digit = digit.astype(np.uint8)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(figure.astype(np.uint8), cv2.COLOR_Lab2BGR))

    plt.savefig('models/fruit/vae/results/' + time_experiment + '/image_generator.png')

    return 5



if __name__ == '__main__':
    import os
    os.chdir('..')
    x_train, y_train, x_val, y_val, x_test, y_test, labels_train, labels_val = load_data(0.01)
    model, encoder, generator, model_colorization = vae_model(latent_dim=64)
    model, history, time_experiment = train(model=model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    create_plots_paper(x_val=x_val, label_val=labels_val, model_colorization=model_colorization,
                       time_experiment=time_experiment, encoder= encoder, generator = generator)

