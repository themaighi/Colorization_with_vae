import pandas as pd
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle


## Create the KNN model from simple picture
## Create KNN model from the semantic distribution
## Predict with the compex model

def process_data_for_knn_on_pictures(path):

    ## Train images
    list_images = os.listdir(path + 'Training/real')
    label_general_train = []
    label_specific_train = []
    images_train = []
    for img_label in list_images: #img_label = list_images[0]
        file_read = pd.read_pickle(path + 'Training/real/' + img_label)
        images_train.extend([i/255 for i in file_read['real_img']])
        label_general_train.extend([lab.split(' ')[0] for lab in file_read['y']])
        label_specific_train.extend(file_read['y'])

    ## Test images

    list_images = os.listdir(path + 'Test/real')
    label_general_test = []
    label_specific_test = []
    images_test = []
    for img_label in list_images: #img_label = list_images[0]
        file_read = pd.read_pickle(path + 'Test/real/' + img_label)
        images_test.extend([i/255 for i in file_read['real_img']])
        label_general_test.extend([lab.split(' ')[0] for lab in file_read['y']])
        label_specific_test.extend(file_read['y'])


    return images_train, label_general_train, label_specific_train, images_test, label_general_test, label_specific_test


def process_data_for_knn_on_semantic_distribution(path):
    ## Train images
    list_images = os.listdir(path + 'Training/knn')
    label_general_train = []
    label_specific_train = []
    images_train = []
    for img_label in list_images:  # img_label = list_images[0]
        file_read = pd.read_pickle(path + 'Training/knn/' + img_label)
        images_train.extend([i / 255 for i in file_read['X']])
        label_general_train.extend([lab.split(' ')[0] for lab in file_read['y']])
        label_specific_train.extend(file_read['y'])

    ## Test images

    list_images = os.listdir(path + 'Test/real')
    label_general_test = []
    label_specific_test = []
    images_test = []
    for img_label in list_images:  # img_label = list_images[0]
        file_read = pd.read_pickle(path + 'Test/real/' + img_label)
        images_test.extend([i / 255 for i in file_read['X']])
        label_general_test.extend([lab.split(' ')[0] for lab in file_read['y']])
        label_specific_test.extend(file_read['y'])

    return images_train, label_general_train, label_specific_train, images_test, label_general_test, label_specific_test

def estimate_knn_models():
    return 5

def predict_subset_of_test():
    return 5

def process_data_model_layer1():
    with open('data/processed_fruit_images_knnTraining.pkl', 'rb') as f:
        dict_var = pickle.load(f)
    ## Train
    tranform_images = np.asarray(dict_var['X'])
    tranform_images = tranform_images.reshape((tranform_images.shape[0],tranform_images.shape[2]))
    trans = MinMaxScaler()
    trans.fit(tranform_images)
    tranform_images_train = trans.transform(tranform_images)
    # list_of_lists = [[i.split(' ')[0]] * 10 for i in dict_var['y']]
    labels = dict_var['y']
    # labels = sum(list_of_lists, [])
    lab_trans = LabelEncoder()
    lab_trans.fit(labels)
    labels_train = lab_trans.transform(labels)

    ## Test

    with open('data/processed_fruit_images_knnTest.pkl', 'rb') as f:
        dict_var = pickle.load(f)

    tranform_images = np.asarray(dict_var['X'])
    tranform_images =  tranform_images.reshape((tranform_images.shape[0],tranform_images.shape[2]))
    tranform_images_test = trans.transform(tranform_images)
    # list_of_lists = [[i.split(' ')[0]] * 10 for i in dict_var['y']]
    # list_of_lists = [[i] * 10 for i in dict_var['y']]
    labels = dict_var['y']
    labels_test = lab_trans.transform(labels)

    return tranform_images_train, tranform_images_test, labels_train, labels_test, lab_trans

def train_knn_layer1(train_img, labels_img, *args, **kwargs):

    neigh = KNeighborsClassifier(*args, **kwargs)
    image_train, valid_img,labels_train, valid_labels= train_test_split(train_img, labels_img, test_size=0.2)
    neigh.fit(image_train, labels_train)
    neigh.score(valid_img, valid_labels)

    return neigh

def knn_predict(test_img, labels_test, knn_model):
    '''Make predictions'''
    knn_model.score(test_img, labels_test)

    return 5
if __name__ == '__main__':
    import os
    os.chdir('..')

    images_train_real, label_general_train_real, label_specific_train_real,\
    images_test_real, label_general_test_real, label_specific_test_real = process_data_for_knn_on_pictures('data/fruit/')
    images_train_semantic, label_general_train_semantic, label_specific_train_semantic,\
    images_test_semantic, label_general_test_semantic, label_specific_test_semantic = process_data_for_knn_on_semantic_distribution('data/fruit/')

    tranform_images_train, tranform_images_test, labels_train, labels_test, lab_trans = process_data_model_layer1()
    knn_model = train_knn_layer1(tranform_images_train,labels_train, n_neighbors=2)
    knn_predict(tranform_images_test, labels_test, knn_model)
    # knn_predict_layer1(model_knn_layer1)