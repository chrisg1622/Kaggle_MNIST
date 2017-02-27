# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 13:37:38 2016

@author: chris
"""

import numpy as np
import pandas as pd
import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet
from skimage.filters import gaussian_filter
from skimage.filters import sobel
from sklearn.decomposition import PCA
from skimage.feature import hog


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

#define the CNN
def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below
        ('pool1', layers.MaxPool2DLayer),   # Like downsampling, for execution speed
        ('conv2', layers.Conv2DLayer),
        ('hidden3', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),  #added dropout layer to reduce overfitting
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, 28, 28),
    conv1_num_filters=7, 
    conv1_filter_size=(3, 3), 
    conv1_nonlinearity=lasagne.nonlinearities.rectify,
        
    pool1_pool_size=(2, 2),
        
    conv2_num_filters=12, 
    conv2_filter_size=(2, 2),    
    conv2_nonlinearity=lasagne.nonlinearities.rectify,
    
    hidden3_num_units=100,
    dropout1_p=0.5,
    output_num_units=10, 
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.001,
    update_momentum=0.9,

    max_epochs=n_epochs,
    verbose=1,
    )
    return net1

###############################################################################
###############################################################################

############################# PREPROCESSING ###################################
print('Preprocessing training data... \n')
#Read in MNIST digits data
Digits_train_data = pd.read_csv("input/train.csv",header = 0)
Digits_test_data = pd.read_csv("input/test.csv",header = 0)

#take a copy of the training data
X = Digits_train_data.copy()
#separate labels from data in training set
t = X['label']
del X['label']
#take a copy of the testing data
X_test_unprocessed = Digits_test_data.copy()

#Transform training and testing data using one of the feature engineering methods
X_processed = X
X_test_processed = X
#X_processed = feature_engineering(X, flag = 'gaussian blur')
#X_test_processed = feature_engineering(X_test_unprocessed, flag = 'gaussian blur')
#X_processed = feature_engineering(X, flag = 'sobel edge detection')
#X_test_processed = feature_engineering(X_test_unprocessed, flag = 'sobel edge detection')
#X_processed = feature_engineering(X, flag = 'principal components', n_pca = 784)
#X_test_processed = feature_engineering(X_test_unprocessed, flag = 'principal components', n_pca = 784)
#X_processed = feature_engineering(X, flag = 'hog features')
#X_test_processed = feature_engineering(X_test_unprocessed, flag = 'hog features')

# Read competition data files:
t_train = t.values.ravel()
X_train = X_processed.values
X_test = X_test_processed.values

# convert to array, specify data type, and reshape
t_train = t_train.astype(np.uint8)
X_train = np.array(X_train).reshape((-1, 1, 28, 28)).astype(np.uint8)
X_test = np.array(X_test).reshape((-1, 1, 28, 28)).astype(np.uint8)

print('Fitting model \n')
# train the CNN model for 10 epochs
cnn = CNN(10).fit(X_train,t_train)

print('Saving test predictions to csv...')
# use the NN model to classify test data
t_predict_sub = pd.DataFrame(cnn.predict(X_test))
#format and write the predictions to a csv file
t_predict_sub.index += 1
t_predict_sub.index.names = ['ImageId']
t_predict_sub.columns = ['Label']
t_predict_sub.to_csv(path_or_buf = 'output/results_CNN_Without_Rotations.csv')