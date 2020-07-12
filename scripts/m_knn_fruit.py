import pandas as pd
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle




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
    tranform_images_train, tranform_images_test, labels_train, labels_test, lab_trans = process_data_model_layer1()
    knn_model = train_knn_layer1(tranform_images_train,labels_train, n_neighbors=2)
    knn_predict(tranform_images_test, labels_test, knn_model)
    # knn_predict_layer1(model_knn_layer1)