import pandas as pd
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
#
# def knn_fit(dt_dict, *args, **kwargs):
#     '''Fit a Knn model'''
#     neigh = KNeighborsClassifier(*args, **kwargs)
#     tranform_images = np.asarray(dt_dict['X'])
#     tranform_images = tranform_images.reshape((tranform_images.shape[0],tranform_images.shape[2]))
#     neigh.fit(tranform_images, dt_dict['y'])
#
#     return neigh
#
# def knn_fit_output_layer(dt_dict, *args, **kwargs):
#     '''Fit a Knn model'''
#     neigh = KNeighborsClassifier(*args, **kwargs)
#     tranform_images = np.asarray(dt_dict['layer_1'])
#     tranform_images = tranform_images.reshape((tranform_images.shape[0],tranform_images.shape[2]))
#     trans = MinMaxScaler()
#     trans.fit(tranform_images)
#     tranform_images = trans.transform(tranform_images)
#
#     lab_trans = LabelEncoder()
#     lab_trans.fit(dt_dict['y'])
#     labels = lab_trans.transform(dt_dict['y'])
#     neigh.fit(tranform_images, labels)
#
#     pred = neigh.predict(tranform_images[50:55])
#
#     return neigh
#
# def run_knn(*args, **kwargs):
#     with open('data/d_image_processed_knn_train_additonal_info.pkl', 'rb') as f:
#         dict_var = pickle.load(f)
#     # model_knn = knn_fit(dict_var, *args, **kwargs)
#     model_knn_layer1 = knn_fit_output_layer(dict_var, *args, **kwargs)
#     return model_knn, model_knn_layer1



# def knn_predict_layer1(knn_model):
#     '''Make predictions'''
#     with open('data/d_image_processed_knn_test_additonal_info.pkl', 'rb') as f:
#         dt_dict = pickle.load(f)
#     tranform_images = np.asarray(dt_dict['layer_1'])
#     tranform_images = tranform_images.reshape((tranform_images.shape[0],tranform_images.shape[2]))
#     predictions = knn_model.predict(tranform_images[0:100])
#     knn_model.score(tranform_images[300:305,], dt_dict['y'][300:305])

def process_data_model_layer1():
    with open('data/d_image_processed_knn_train_additonal_info.pkl', 'rb') as f:
        dict_var = pickle.load(f)
    ## Train
    tranform_images = np.asarray(dict_var['layer_1'])
    tranform_images = tranform_images.reshape((tranform_images.shape[0],tranform_images.shape[2]))
    trans = MinMaxScaler()
    trans.fit(tranform_images)
    tranform_images_train = trans.transform(tranform_images)

    lab_trans = LabelEncoder()
    lab_trans.fit(dict_var['y'])
    labels_train = lab_trans.transform(dict_var['y'])

    ## Test

    with open('data/d_image_processed_knn_test_additonal_info.pkl', 'rb') as f:
        dict_var = pickle.load(f)

    tranform_images = np.asarray(dict_var['layer_1'])
    tranform_images = tranform_images.reshape((tranform_images.shape[0],tranform_images.shape[2]))
    tranform_images_test = trans.transform(tranform_images)

    labels_test = lab_trans.transform(dict_var['y'])

    return tranform_images_train, tranform_images_test, labels_train, labels_test, lab_trans

def train_knn_layer1(train_img, labels_img, *args, **kwargs):

    neigh = KNeighborsClassifier(*args, **kwargs)
    neigh.fit(train_img, labels_img)

    return neigh

def knn_predict(test_img, labels_test, knn_model):
    '''Make predictions'''
    knn_model.score(test_img, labels_test)

    return 5
if __name__ == '__main__':
    import os
    os.chdir('..')
    tranform_images_train, tranform_images_test, labels_train, labels_test, lab_trans = process_data_model_layer1()
    knn_model = train_knn_layer1(tranform_images_train,labels_train, n_neighbors=1, p=1)
    knn_predict(tranform_images_test, labels_test, knn_model)
    # knn_predict_layer1(model_knn_layer1)
