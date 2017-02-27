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
from skimage.transform import rotate
    
#function to bootstrap sample n images from the data, rotate them by the given 
#angle, and add them to the data
def add_rotated_images(data, n_images, angle):
    X = data.copy()
    #take a sample of the training data to rotate and add to data
    X_sample = X.sample(n=n_images, replace = True).copy()
    #reset index for the sample
    X_sample.index = range(len(X_sample))
    #separate labels from data in training set
    t_sample = X_sample['label']
    del X_sample['label']
    #compute the number of pixel rows/columns in the image
    image_size = int(np.sqrt(X.shape[1]))
    #rotate each image in the sample by the required angle
    for i in X_sample.index:
        #retrive the ith observation
        obs_unfiltered = np.array(X_sample.iloc[i], dtype=np.float64)
        #resize the current observation into a matrix
        obs_unfiltered = np.resize(obs_unfiltered, new_shape = (image_size,image_size))
        #compute the rotated image
        obs_filtered = rotate(image = obs_unfiltered, angle = angle) 
        #reshape the rotated image into an array
        obs_filtered = np.resize(obs_filtered, new_shape = (image_size*image_size))
        #store the rotated image back into the data frame
        X_sample.iloc[i] = obs_filtered
    X_sample = pd.concat([t_sample,X_sample],axis=1)     
    #make the index of the sample continue from that of X
    X_sample.index += len(X)
    #join the sample onto the dataset
    X2 = pd.concat([X,X_sample], axis=0)
    return X2    

#define the CNN
def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),       #input layer
        ('conv1', layers.Conv2DLayer),      #Convolutional layer
        ('pool1', layers.MaxPool2DLayer),   #max pooling layer for sub-sampling, reduces overfitting
        ('conv2', layers.Conv2DLayer),      #another convolutional layer
        ('hidden3', layers.DenseLayer),     #fully connected layer
        ('dropout1', layers.DropoutLayer),  #dropout layer to reduce overfitting
        ('output', layers.DenseLayer),      #fully connected output layer
        ],

    input_shape=(None, 1, 28, 28),
    conv1_num_filters=10, 
    conv1_filter_size=(3, 3), 
    conv1_nonlinearity=lasagne.nonlinearities.rectify,
        
    pool1_pool_size=(2, 2),
        
    conv2_num_filters=15, 
    conv2_filter_size=(2, 2),    
    conv2_nonlinearity=lasagne.nonlinearities.rectify,
    
    hidden3_num_units=750,
    
    dropout1_p=0.5,
    
    output_num_units=30, 
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.0005,
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

X = Digits_train_data.copy()

#take bootsrap samples of the data, rotate the images at different angles,
#and add the samples back onto the dataset
X_processed = add_rotated_images(data = X, n_images = 10000, angle = 5)
X_processed = add_rotated_images(data = X_processed, n_images = 10000, angle = 10)
X_processed = add_rotated_images(data = X_processed, n_images = 10000, angle = 15)
X_processed = add_rotated_images(data = X_processed, n_images = 10000, angle = 20)
X_processed = add_rotated_images(data = X_processed, n_images = 10000, angle = -5)
X_processed = add_rotated_images(data = X_processed, n_images = 10000, angle = -10)
X_processed = add_rotated_images(data = X_processed, n_images = 10000, angle = -15)
X_processed = add_rotated_images(data = X_processed, n_images = 10000, angle = -20)
#separate labels from data in training set
t = X_processed['label']
del X_processed['label']

# Read competition data files:
t_train = t.values.ravel()
X_train = X_processed.values
X_test = Digits_test_data.copy().values

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
t_predict_sub.to_csv(path_or_buf = 'output/results_CNN_With_Rotations.csv')