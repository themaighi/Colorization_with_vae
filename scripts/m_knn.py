import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier


def read_img(filename, img_size):
    '''Reading images and reshape them depending on the NN needed shape'''
    img = cv2.imread(filename, 3)
    height, width, channels = img.shape
    min_hw = int(min(height, width) / 2)
    img = img[int(height / 2) - min_hw:int(height / 2) + min_hw, int(width / 2) - min_hw:int(width / 2) + min_hw, :]
    labimg = cv2.cvtColor(cv2.resize(img, (img_size, img_size)), cv2.COLOR_BGR2Lab)
    return np.reshape(labimg[:, :, 0], (img_size, img_size, 1)), labimg[:, :, 1:]


def import_images(folder_path, proportion):
    '''Steps needed to import the images
    - Batch = is the list of images Black and white (first layer)
    - Labels = Are the color of the image
    - Filelist = list of files'''
    list_img = os.listdir(folder_path)
    n_img = len(list_img)
    selection_mask = np.random.binomial(1,proportion,size=n_img)
    img_to_import = [list_img[i] for i in range(n_img) if selection_mask[i] == 1]
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

def knn_fit(tranform_images, classification_labels, *args, **kwargs):
    '''Fit a Knn model'''
    neigh = KNeighborsClassifier(n_neighbors=3)
    tranform_images = np.asarray(tranform_images)
    tranform_images = tranform_images.reshape((tranform_images.shape[0],tranform_images.shape[2]))
    neigh.fit(tranform_images, classification_labels['category_labels'].values)

    neigh.predict([tranform_images[0],tranform_images[1]])
    return neigh
def knn_predict(img, knn_model):
    '''Make predictions'''

    return knn_model.predict(img)


def run():
    batch, labels, filelist = import_images('data/val_256', 0.001)
    classification_labels = label_images(filelist)
    model_classification = load_model('models/my_model_colorization.h5')
    tranform_images = transform_image(batch, model_classification)
    knn_fit(tranform_images, classification_labels)

    return 5



if __name__ == '__main__':
    import os
    os.chdir('..')
    run()