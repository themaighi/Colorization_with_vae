import pandas as pd
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
import time
import random
from keras.applications.vgg16 import decode_predictions
import matplotlib.pyplot as plt
## Try to run the model on real images without scaling resizing the images
## Try to run the model and test it on the whole test

def process_data_for_knn_on_pictures(path):

    ## Train images
    list_images = os.listdir(path + 'Training/real')
    label_general_train = []
    label_specific_train = []
    images_train = []
    for img_label in list_images: #img_label = list_images[0]
        file_read = pd.read_pickle(path + 'Training/real/' + img_label)
        images_train.extend([cv2.resize(i/255, (32, 32))[:,:,0].flatten() for i in file_read['real_img']])
        label_general_train.extend([lab.split(' ')[0] for lab in file_read['y']])
        label_specific_train.extend(file_read['y'])

    ## Test images

    list_images = os.listdir(path + 'Test/real')
    label_general_test = []
    label_specific_test = []
    images_test = []
    for img_label in list_images: #img_label = list_images[0]
        file_read = pd.read_pickle(path + 'Test/real/' + img_label)
        images_test.extend([cv2.resize(i/255, (32, 32))[:,:,0].flatten() for i in file_read['real_img']])
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
        images_train.extend([i.flatten() for i in file_read['X']])
        label_general_train.extend([lab.split(' ')[0] for lab in file_read['y']])
        label_specific_train.extend(file_read['y'])

    ## Test images

    list_images = os.listdir(path + 'Test/knn')
    label_general_test = []
    label_specific_test = []
    images_test = []
    for img_label in list_images:  # img_label = list_images[0]
        file_read = pd.read_pickle(path + 'Test/knn/' + img_label)
        images_test.extend([i.flatten() for i in file_read['X']])
        label_general_test.extend([lab.split(' ')[0] for lab in file_read['y']])
        label_specific_test.extend(file_read['y'])

    return images_train, label_general_train, label_specific_train, images_test, label_general_test, label_specific_test

def knn_model(X,y, *args, **kwargs):
    model = KNeighborsClassifier(*args, **kwargs)
    return model.fit(X, y)

def estimate_knn_models(train_real, label_real_detailed, label_real_general,
                        train_semantic, label_semantic_detailed, label_smantic_general,
                        *args, **kwargs):


    time_before = time.time()
    neigh_real_general = knn_model(train_real, label_real_general, *args, **kwargs)
    time_after = time.time()
    print('----- Estimation process taken Real general: ', str(time_after - time_before))
    time_before = time.time()
    neigh_real_detailed = knn_model(train_real, label_real_detailed, *args, **kwargs)
    time_after = time.time()
    print('----- Estimation process taken Real detailed: ', str(time_after - time_before))
    time_before = time.time()
    neigh_semantic_general = knn_model(train_semantic, label_smantic_general, *args, **kwargs)
    time_after = time.time()
    print('----- Estimation process taken Semantic General: ', str(time_after - time_before))
    time_before = time.time()
    neigh_semantic_detailed = knn_model(train_semantic, label_semantic_detailed, *args, **kwargs)
    time_after = time.time()
    print('----- Estimation process taken Semantic Real: ', str(time_after - time_before))
    return neigh_real_general, neigh_real_detailed, neigh_semantic_general, neigh_semantic_detailed
    # return neigh_real_general

def metrics_calculation(model, X_test, y_test):

    time_before = time.time()
    score_prediction = model.score(X_test, y_test)
    time_after = time.time()
    print('----- Estimation process time prediction: ', str(time_after - time_before))

    return score_prediction

def print_predicted_label(model, X_test):
    time_before = time.time()
    predicted_labels = model.predict(X_test)
    time_after = time.time()
    print('----- Estimation process time prediction: ', str(time_after - time_before))
    return predicted_labels

if __name__ == '__main__':
    import os
    os.chdir('..')

    images_train_real, label_general_train_real, label_specific_train_real,\
    images_test_real, label_general_test_real, label_specific_test_real = process_data_for_knn_on_pictures('data/fruit/')
    images_train_semantic, label_general_train_semantic, label_specific_train_semantic,\
    images_test_semantic, label_general_test_semantic, label_specific_test_semantic = process_data_for_knn_on_semantic_distribution('data/fruit/')

    list_images = os.listdir('data/fruit/' + 'Test/real')
    images_real = []
    for img_label in list_images:  # img_label = list_images[0]
        file_read = pd.read_pickle('data/fruit/' + 'Test/real/' + img_label)
        images_real.extend([i for i in file_read['real_img']])

    prop = 0.1
    mask_score = random.sample(range(len(images_test_real)), int(np.ceil(len(images_test_real) * prop)))
    mask_prediction = random.sample(range(len(images_test_real)), 15)
    metrics_total = pd.DataFrame()

    for k in range(1,10):
        print('-------- Running model with ', k, 'neighbours -------')
        neigh_real_general, neigh_real_detailed,\
        neigh_semantic_general, neigh_semantic_detailed = estimate_knn_models(images_train_real,label_specific_train_real, label_general_train_real,
                            images_train_semantic, label_specific_train_semantic, label_general_train_semantic, n_neighbors=k)
        # neigh_real_general = estimate_knn_models(images_train_real, label_specific_train_real, label_general_train_real,
        #                                           images_train_semantic, label_specific_train_semantic, label_general_train_semantic)

        ## Calculate score KNN

        #
        score_real_general = metrics_calculation(neigh_real_general, [images_test_real[i] for i in mask_score],
                                                 [label_general_test_real[i] for i in mask_score])
        score_real_detailed = metrics_calculation(neigh_real_detailed, [images_test_real[i] for i in mask_score],
                                                 [label_specific_test_real[i] for i in mask_score])
        score_semantic_general = metrics_calculation(neigh_semantic_general, [images_test_semantic[i] for i in mask_score],
                                                 [label_general_test_semantic[i] for i in mask_score])
        score_semantic_detailed = metrics_calculation(neigh_semantic_detailed, [images_test_semantic[i] for i in mask_score],
                                                 [label_specific_test_semantic[i] for i in mask_score])
        metrics_total = metrics_total.append(pd.DataFrame({'k':k, 'score_model1': score_real_general,
                                           'score_model2': score_real_detailed, 'score_model3': score_semantic_general,
                                           'score_model4': score_semantic_detailed}, index=[0])).reset_index(drop=True)
        print('List of scores \n', metrics_total)

        ## Make predictions


        prediction_real_general = print_predicted_label(neigh_real_general, [images_test_real[i] for i in mask_prediction])
        prediction_real_detailed = print_predicted_label(neigh_real_detailed, [images_test_real[i] for i in mask_prediction])
        prediction_semantic_general = print_predicted_label(neigh_semantic_general, [images_test_semantic[i] for i in mask_prediction])
        prediction_semantic_detailed = print_predicted_label(neigh_semantic_detailed, [images_test_semantic[i] for i in mask_prediction])

        ## Prediction with colorization model

        chroma_gan_predictons = decode_predictions(np.asarray([images_test_semantic[i] for i in mask_prediction]))
        chroma_gan_predictons = [i[0][1] for i in chroma_gan_predictons]
        ## Create a plot with predicted labels, real labels and real image


        img_real_plot = [images_real[i] for i in mask_prediction]
        labels_img = [label_specific_test_real[i] for i in mask_prediction]
        n_rows = 3
        n_cols = 5
        plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6), )
        for row in range(n_rows):
            for col in range(n_cols):
                index = n_cols * row + col
                plt.subplot(n_rows, n_cols, index + 1)
                image = cv2.cvtColor(img_real_plot[index], cv2.COLOR_BGR2RGB)
                plt.imshow(image)
                plt.axis('off')
                plt.text(0.5,120, 'Real Label: ' + labels_img[index], ha='center', size=6)
                plt.text(0.5, 135, 'Model 1: ' + prediction_real_general[index], ha='center', size=6)
                plt.text(0.5, 150, 'Model 2: ' + prediction_real_detailed[index], ha='center', size=6)
                plt.text(0.5, 165, 'Model 3: ' + prediction_semantic_general[index], ha='center', size=6)
                plt.text(0.5, 180, 'Model 4: ' + prediction_semantic_detailed[index], ha='center', size=6)
                plt.text(0.5, 195, 'ChromaGan: ' + chroma_gan_predictons[index], ha='center', size=6)

        plt.tight_layout(pad=0.5)
        os.makedirs('results/fruit/knn/', exist_ok=True)
        plt.savefig('results/fruit/knn/' + 'prediction_k' + str(k) + '.png')

    metrics_total.to_csv('results/fruit/knn/score_dt.csv', index=False)





