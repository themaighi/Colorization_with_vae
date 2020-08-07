import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

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
        greyimg, colorimg = read_img(filename, img_size=50)
        batch.append(greyimg/255)
        labels.append(colorimg)

    return batch, labels, filelist


def label_images(filelist):
    '''For each image assign the class that it belongs to'''
    labels_txt = pd.read_table('data/file_list-standard/places365_val.txt', sep=' ')
    labels_txt.columns = ['filename', 'category']
    dt = pd.DataFrame({'filename':filelist})
    dt = dt.merge(labels_txt, on='filename', how='left')

    category_label = pd.read_table('data/file_list-standard/categories_places365.txt', sep=' ', header=None)
    category_label.columns = ['category_labels', 'category']
    dt = dt.merge(category_label, on='category', how='left')

    return dt

def run_knn():

    list_img = os.listdir('data/val_256')
    img_to_select = np.random.choice(list_img, 5000, replace=False)
    train_img, test_img = train_test_split(img_to_select, test_size=0.2)

    batch_train, labels_train, filelist_train = import_images(train_img, 'data/val_256')
    classification_labels_train = label_images(filelist_train)['category_labels']

    # lab_trans = LabelEncoder()
    # lab_trans.fit(labels)
    # classification_labels_train = lab_trans.transform(labels)

    batch_train = np.array(batch_train)
    batch_train = np.reshape(batch_train, (batch_train.shape[0], -1))

    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(batch_train, classification_labels_train)

    batch_test, labels_test, filelist_test = import_images(test_img, 'data/val_256')
    classification_labels_test = label_images(filelist_test)['category_labels']
    # classification_labels_test = lab_trans.transform(labels)

    batch_test = np.array(batch_test)
    batch_test = np.reshape(batch_test, (batch_test.shape[0], -1))

    score_val = neigh.score(batch_test, classification_labels_test)
    score_val = neigh.predict(batch_test)


if __name__ == '__main__':
    import os
    os.chdir('..')
    run_knn()