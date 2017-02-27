# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:25:03 2016

@author: chris
"""

import numpy as np
import pandas as pd
from skimage.filters import gaussian_filter
from skimage.filters import sobel
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

#function to carry out feature engineering methods on input data and return these features
def feature_engineering(data, flag, pca_fit_data = None, pca_test_flag = False,  n_pca = 150, std_blur = 1.0, pixels_per_cell_hog = (4,4), orientations_hog = 5):
    X = data.copy() 
    #compute the number of pixel rows/columns in the image
    image_size = int(np.sqrt(X.shape[1]))
    
    #method to replace each image in the dataset with the gaussian blurred version of itself
    if (flag == 'gaussian blur'):
        for i in X.index:
            #retrive the ith observation
            obs_unfiltered = np.array(X.iloc[i], dtype=np.float64)
            #resize the current observation into a matrix
            obs_unfiltered = np.resize(obs_unfiltered, new_shape = (image_size,image_size))
            #compute the filtered image
            obs_filtered = gaussian_filter(obs_unfiltered, std_blur) 
            #reshape the filtered image into an array
            obs_filtered = np.resize(obs_filtered, new_shape = (image_size*image_size))
            #store the filtered image back into the data frame
            X.iloc[i] = obs_filtered
            
    elif (flag == 'sobel edge detection'):
        for i in X.index:
            #retrive the ith observation
            obs_unfiltered = np.array(X.iloc[i], dtype=np.float64)
            #resize the current observation into a matrix
            obs_unfiltered = np.resize(obs_unfiltered, new_shape = (image_size,image_size))
            #compute the filtered image
            obs_filtered = sobel(obs_unfiltered) 
            #reshape the filtered image into an array
            obs_filtered = np.resize(obs_filtered, new_shape = (image_size*image_size))
            #store the filtered image back into the data frame
            X.iloc[i] = obs_filtered
    elif (flag == 'principal components'):
        #if applying pca to training data
        if (pca_test_flag == False):
            #initialise a pca model
            pca = PCA(n_components=n_pca)
            #fit the model and compute PCs
            X = pd.DataFrame(pca.fit_transform(X))
        #if applying pca to test data, need to fit model on training data
        else:
            #initialise a pca model
            pca = PCA(n_components=n_pca)
            #fit the model on training data and compute PCs on test data
            pca.fit(pca_fit_data)
            X = pd.DataFrame(pca.transform(X))
            
    elif (flag == 'hog features'):
        for i in X.index:
            #retrive the ith observation
            obs_unfiltered = np.array(X.iloc[i], dtype=np.float64)
            #resize the current observation into a matrix
            obs_unfiltered = np.resize(obs_unfiltered, new_shape = (image_size,image_size))
            #compute the filtered image
            obs_filtered_features, obs_filtered_image = hog(obs_unfiltered, orientations = orientations_hog, pixels_per_cell=pixels_per_cell_hog, visualise = True)
            #reshape the filtered image into an array
            obs_filtered = np.resize(obs_filtered_image, new_shape = (image_size*image_size))
            #store the filtered image back into the data frame
            X.iloc[i] = obs_filtered

    else:
        return 'Error: feature engineering method not found'
    
    return X
    

###############################################################################
###############################################################################

############################# PREPROCESSING ###################################
print('Preprocessing training data... \n')
#Read in MNIST digits data
Digits_train_data = pd.read_csv("input/train.csv",header = 0)
Digits_test_data = pd.read_csv("input/test.csv",header = 0)
#take a copy of the training data
X_all = Digits_train_data.copy()
#take a sample of the training data to use for CV
X_sample = X_all.sample(n=20000, replace = True, random_state = 1)
X = X_sample.copy()
#reset index for the sample
X.index = range(len(X))
#separate labels from data in training set
t = X['label']
del X['label']
#Transform data using one of the feature engineering methods
#X = feature_engineering(X, flag = 'gaussian blur')
#X = feature_engineering(X, flag = 'sobel edge detection')
#X = feature_engineering(X, flag = 'hog features',pixels_per_cell_hog = (4,4), orientations_hog = 4)

#Check that the sample label counts are reasonably even.
#print('sample label counts: \n', X_sample['label'].value_counts())

############################# CROSS-VALIDATION ################################
print('Starting 10 fold cross-validation on {} samples'.format(len(X_sample)))
#initialise folds for 10 fold CV
folds = KFold(len(X), n_folds=10)
#create array to store accuracy scores
accuracy_scores = [0]*10
#initialise counter for iterating in the scores array
i = 0
#fit and score model for each CV fold
for train_index, test_index in folds:
    #for the currect fold, retrieve training data and preprocess it
    X_train_unprocessed = X.iloc[train_index]
    #X_train = feature_engineering(X_train_unprocessed, flag = 'principal components',n_pca=40)
    X_train = X_train_unprocessed    
    #retrieve training target
    t_train = t.iloc[train_index]
    #retrive testing data and preprocess it
    X_test_unprocessed = X.iloc[test_index]
    #X_test = feature_engineering(X.iloc[test_index], flag = 'principal components', pca_fit_data = X_train_unprocessed, pca_test_flag = True,n_pca=40) 
    X_test = X_test_unprocessed    
    #retrive test target
    t_test = t.iloc[test_index]
    
    #initialise model
    knn = KNeighborsClassifier(n_neighbors=3)
    #fit model
    knn.fit(X_train, t_train)
    #predict on test data
    t_predict = knn.predict(X_test)
    #compute accuracy of current fold and store in array
    accuracy_scores[i] = accuracy_score(t_test,t_predict)
    #increment counter
    i += 1
    print('Fold {}, dev accuracy = {}'.format(i-1,accuracy_scores[i-1]))
    
#compute mean and std of accuracy score
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
print('10-fold accuracy mean: {}'.format(mean_accuracy))
print('10-fold accuracy std: {} \n'.format(std_accuracy))


############################# FITTING MODEL ###################################

print('Preprocessing test data \n')
#take a copy of the testing data
X_testing_data = Digits_test_data.copy()
#reset the training data as the whole dataset
X = X_all
#separate labels from data in training set
t = X['label']
del X['label']

#preprocess submission data
X_processed = X
X_testing_data_processed = X_testing_data

#X_processed = feature_engineering(X, flag = 'gaussian blur')
#X_testing_data_processed = feature_engineering(X_testing_data, flag = 'gaussian blur') 

#X_processed = feature_engineering(X, flag = 'sobel edge detection')
#X_testing_data_processed = feature_engineering(X_testing_data, flag = 'sobel edge detection')

#X_processed = feature_engineering(X, flag = 'hog features',pixels_per_cell_hog = (4,4), orientations_hog = 4)
#X_testing_data_processed = feature_engineering(X_testing_data, flag = 'hog features',pixels_per_cell_hog = (4,4), orientations_hog = 4)

#X_processed = feature_engineering(X, flag = 'principal components',n_pca=40)
#X_testing_data_processed = feature_engineering(X_testing_data, flag = 'principal components', pca_fit_data = X, pca_test_flag = True,n_pca=40) 

print('Fitting model \n')
#initialise a model for submission
knn_sub = KNeighborsClassifier(n_neighbors=3)
#fit model on training dataset
knn_sub.fit(X_processed,t)
#compute predictions on submission data
t_predict_sub = pd.DataFrame(knn_sub.predict(X_testing_data_processed))

print('Saving test predictions to csv...')
#format and write the predictions to a csv file
t_predict_sub.index += 1
t_predict_sub.index.names = ['ImageId']
t_predict_sub.columns = ['Label']
t_predict_sub.to_csv(path_or_buf = 'output/results_KNN.csv')














