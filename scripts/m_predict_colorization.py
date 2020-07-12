import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
import pickle
from keras.applications.vgg16 import decode_predictions
import subprocess

def read_img(filename, img_size):
    '''Reading images and reshape them depending on the NN needed shape'''
    img = cv2.imread(filename, 3)
    height, width, channels = img.shape
    min_hw = int(min(height, width) / 2)
    img = img[int(height / 2) - min_hw:int(height / 2) + min_hw, int(width / 2) - min_hw:int(width / 2) + min_hw, :]
    labimg = cv2.cvtColor(cv2.resize(img, (img_size, img_size)), cv2.COLOR_BGR2Lab)
    return np.reshape(labimg[:, :, 0], (img_size, img_size, 1)), labimg[:, :, 1:]

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
    labels = []
    filelist = []
    for img in img_to_import:
        filename = os.path.join(folder_path, img)
        filelist.append(img)
        greyimg, colorimg = read_img(filename, img_size=224)
        batch.append(greyimg)
        labels.append(colorimg)

    return batch, labels, filelist


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)

def predict_colorization():

    list_img = os.listdir('data/val_256')
    img_to_select = np.random.choice(list_img, 10, replace=False)
    model_classification = load_model('models/my_model_colorization.h5')
    batch, labels, filelist = import_images(img_to_select, 'data/val_256')

    os.makedirs('data/reconstructed/', exist_ok=True)
    labels_predicted = list()
    for i in range(len(batch)): #i = 0
        pred_ab, pred_obj = model_classification.predict(np.tile(batch[i]/255,[1,1,1,3]))
        label = decode_predictions(pred_obj)
        labels_predicted.append(label)
        result = np.concatenate((batch[i], deprocess(pred_ab[0])), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join('data/reconstructed/', filelist[i] + "_reconstructed.jpg")
        subprocess.call('cp data/val_256/' + filelist[i] + ' data/reconstructed/' + filelist[i] + '_original.jpg')

        cv2.imwrite(save_path, result)

    print(labels_predicted)





if __name__ == '__main__':
    import os
    os.chdir('..')
    predict_colorization()
