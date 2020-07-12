import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle



def split_test(folder_path, proportion):
    list_img = os.listdir(folder_path)
    train_img, test_img = train_test_split(list_img, test_size=proportion)
    return train_img, test_img

def read_img(filename, img_size):
    '''Reading images and reshape them depending on the NN needed shape'''
    img = cv2.imread(filename, 3)
    height, width, channels = img.shape
    min_hw = int(min(height, width) / 2)
    img = img[int(height / 2) - min_hw:int(height / 2) + min_hw, int(width / 2) - min_hw:int(width / 2) + min_hw, :]
    labimg = cv2.cvtColor(cv2.resize(img, (img_size, img_size)), cv2.COLOR_BGR2Lab)
    return np.reshape(labimg[:, :, 0], (img_size, img_size, 1)), labimg[:, :, 1:], img


def import_images(list_img, folder_path):
    '''Steps needed to import the images
    - Batch = is the list of images Black and white (first layer)
    - Labels = Are the color of the image
    - Filelist = list of files'''


    n_img = len(list_img)
    # selection_mask = np.random.binomial(1,proportion,size=n_img)
    # img_to_import = [list_img[i] for i in range(n_img) if selection_mask[i] == 1]
    img_to_import = list_img
    batch = []
    actual_image = []
    labels = []
    filelist = []
    for img in img_to_import:
        filename = os.path.join(folder_path, img)
        filelist.append(img)
        greyimg, colorimg, img = read_img(filename, img_size=224)
        batch.append(greyimg)
        labels.append(colorimg)
        actual_image.append(img)

    return batch, labels, filelist, actual_image


def load_model(model_location):
    '''Load the colorization model and take the needed steps'''
    model_colorization = tf.keras.models.load_model(model_location)
    model_colorization.summary()
    model_classification = tf.keras.models.Model(model_colorization.input,model_colorization.layers[-1].output)

    return model_classification

def transform_image(batch_images, model_classification):
    '''Use the model to make predictions on the images'''
    transformed_images = []
    for img in batch_images:
        transformed_images.append(model_classification.predict(np.tile(img / 255, [1, 1, 1, 3])))
        print(len(transformed_images)/len(batch_images)*100)
    # np.asarray(batch_images)/255
    # transformed_images = model_classification.predict(np.tile(np.asarray(batch_images) / 255, [1, 1, 1, 3]))
    return transformed_images

def process_data(path):
    labels_images = os.listdir(path)
    model_classification = load_model('models/my_model_colorization.h5')
    actual_image_list = list()
    classification_labels_list = list()
    black_with_image = list()
    color_image = list()
    transformed_images = list()

    for i in labels_images: # i = labels_images[0]
        print(i)
        image_list = os.listdir(path + '/' + i)
        batch, labels, filelist, actual_image = import_images(image_list, path + '/' + i)
        tranform_images = transform_image(batch, model_classification)
        actual_image_list.append(actual_image)
        black_with_image.append(batch)
        color_image.append(labels)
        classification_labels_list.append(i)
        transformed_images.append(tranform_images)

    output_dict = {'X': black_with_image, 'y': classification_labels_list, 'filename':image_list,
                   'real_img': actual_image_list, 'color_image': color_image, 'knn_input': transformed_images}

    train_or_test = os.path.basename(path)
    with open('data/processed_fruit_images_' + train_or_test + '.pkl', 'wb') as f:
        pickle.dump(output_dict, f)




if __name__ == '__main__':
    import os
    os.chdir('..')
    process_data('data/5857_1166105_bundle_archive/fruits-360/Test')
    process_data('data/5857_1166105_bundle_archive/fruits-360/Training')