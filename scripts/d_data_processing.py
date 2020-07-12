import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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


def label_images(filelist):
    '''For each image assign the class that it belongs to'''
    labels_txt = pd.read_table('data/file_list-standard/places365_val.txt', sep=' ')
    labels_txt.columns = ['filename', 'category']
    dt = pd.DataFrame({'filename':filelist})
    dt = dt.merge(labels_txt, on='filename', how='left')

    category_label = pd.read_table('data/file_list-standard/categories_places365.txt', sep=' ')
    category_label.columns = ['category_labels', 'category']
    dt = dt.merge(category_label, on='category', how='left')

    return dt

def load_model(model_location):
    '''Load the colorization model and take the needed steps'''
    model_colorization = tf.keras.models.load_model(model_location)
    model_colorization.summary()
    model_classification = tf.keras.models.Model(model_colorization.input,model_colorization.layers[-5].output)

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



def process_data():
    train_img, test_img = split_test('data/val_256', 0.2)
    model_classification = load_model('models/my_model_colorization.h5')

    ## Train dataset
    chunk_size = 100
    transformed_image_list = list()
    classification_labels_list = list()
    filename_list = list()
    stop_index = len(train_img)
    # stop_index = 1000
    for i in range(0,stop_index , chunk_size):
        chunk = train_img[i:i + chunk_size]
        batch, labels, filelist = import_images(chunk, 'data/val_256')
        classification_labels = label_images(filelist)
        tranform_images = transform_image(batch, model_classification)
        classification_labels_list.extend(classification_labels.category_labels.values)
        transformed_image_list.extend(tranform_images)
        filename_list.extend(filelist)
    output_dict = {'X': transformed_image_list, 'y': classification_labels_list, 'train_img':filename_list}
    with open('data/d_image_processed_knn_train.pkl', 'wb') as f:
        pickle.dump(output_dict, f)


    ## Test dataset

    transformed_image_list = list()
    classification_labels_list = list()
    filename_list = list()
    stop_index = len(test_img)
    # stop_index = 200

    for i in range(0,stop_index , chunk_size):
        chunk = test_img[i:i + chunk_size]
        batch, labels, filelist = import_images(chunk, 'data/val_256')
        classification_labels = label_images(filelist)
        tranform_images = transform_image(batch, model_classification)
        classification_labels_list.extend(classification_labels.category_labels.values)
        transformed_image_list.extend(tranform_images)
        filename_list.extend(filelist)
    output_dict = {'X': transformed_image_list, 'y': classification_labels_list, 'test_img':filename_list}
    with open('data/d_image_processed_knn_test.pkl', 'wb') as f:
        pickle.dump(output_dict, f)

def get_KGG_output():
    with open('data/d_image_processed_knn_train.pkl', 'rb') as f:
        dict_var = pickle.load(f)
    model_colorization = tf.keras.models.load_model('models/my_model_colorization.h5')
    # model_colorization.summary()
    weight_hidded_3 = model_colorization.layers[-3].get_weights()
    weight_hidded_1 = model_colorization.layers[-1].get_weights()

    def softmax(X):
        exp_x = np.exp(X - np.max(X))
        return exp_x / np.sum(exp_x)

    layer_3=list()
    layer_1 = list()
    i=1
    for img in dict_var['X']: #img = dict_var['X'][18995]
        print(i)
        result_layer_3 = np.dot(img, weight_hidded_3[0])
        result_layer_3 = result_layer_3 + weight_hidded_3[1]
        layer_3.append(result_layer_3)
        result_layer_1 = np.dot(result_layer_3, weight_hidded_1[0])
        result_layer_1 = result_layer_1 + weight_hidded_1[1]
        output_layer = softmax(result_layer_1)
        layer_1.append(output_layer)
        i+=1

    dict_var['layer_3'] = layer_3
    dict_var['layer_1'] = layer_1

    with open('data/d_image_processed_knn_train_additonal_info.pkl', 'wb') as f:
        pickle.dump(dict_var,f)


    with open('data/d_image_processed_knn_test.pkl', 'rb') as f:
        dict_var = pickle.load(f)

    layer_3 = list()
    layer_1 = list()
    i = 1
    for img in dict_var['X']:  # img = dict_var['X'][18995]
        print(i)
        result_layer_3 = np.dot(img, weight_hidded_3[0])
        result_layer_3 = result_layer_3 + weight_hidded_3[1]
        layer_3.append(result_layer_3)
        result_layer_1 = np.dot(result_layer_3, weight_hidded_1[0])
        result_layer_1 = result_layer_1 + weight_hidded_1[1]
        output_layer = softmax(result_layer_1)
        layer_1.append(output_layer)
        i += 1

    dict_var['layer_3'] = layer_3
    dict_var['layer_1'] = layer_1

    with open('data/d_image_processed_knn_test_additonal_info.pkl', 'wb') as f:
        pickle.dump(dict_var, f)


if __name__ == '__main__':
    import os
    os.chdir('..')
    process_data()
    get_KGG_output()
    # knn_model, model_knn_layer1 = run_knn(n_neighbors=5)
    # # knn_predict(knn_model)
    # knn_predict_layer1(model_knn_layer1)
