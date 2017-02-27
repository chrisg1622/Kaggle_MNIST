# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 13:37:38 2016

@author: chris 
"""

import numpy as np
import pandas as pd
from scipy.stats import mode

#Read in the results from the first 16 submissions
t_RF_NoFE = pd.read_csv("Ensemble_input/RF - No FE.csv",header = 0)['Label']
t_RF_Blurring = pd.read_csv("Ensemble_input/RF - Blurring.csv",header = 0)['Label']
t_SVM_PCA_std = pd.read_csv("Ensemble_input/SVM - PCA std.csv",header = 0)['Label']
t_SVM_HOG = pd.read_csv("Ensemble_input/SVM - HOG.csv",header = 0)['Label']
t_KNN_Blurring = pd.read_csv("Ensemble_input/KNN - Blurring.csv",header = 0)['Label']
t_KNN_Edges = pd.read_csv("Ensemble_input/KNN - Edges.csv",header = 0)['Label']
t_KNN_PCA = pd.read_csv("Ensemble_input/KNN - PCA.csv",header = 0)['Label']
t_KNN_HOG = pd.read_csv("Ensemble_input/KNN - HOG.csv",header = 0)['Label']
t_CNN_extended_data = pd.read_csv("Ensemble_input/CNN - Extended Dataset.csv",header = 0)['Label']
t_CNN_extended_data_2 = pd.read_csv("Ensemble_input/CNN - Extended Dataset v2.csv",header = 0)['Label']


#concatenate each submission
t_predict_concat = pd.concat([t_RF_NoFE,t_RF_Blurring,t_SVM_HOG,
                              t_KNN_Blurring,t_KNN_Edges,t_KNN_PCA,t_KNN_HOG,
                              t_CNN_extended_data,t_CNN_extended_data_2],axis=1)
                              
 
#compute the final predictions as the modes of predictions from each classifier
target_modes = [0]*len(t_predict_concat)
for i in range(len(t_predict_concat)):
    target_modes[i] = mode(np.asarray(t_predict_concat.iloc[i]))[0][0]

#format and write the predictions to a csv file
t_predict_sub = pd.DataFrame(data={'Label': target_modes})
t_predict_sub.index += 1
t_predict_sub.index.names = ['ImageId']
t_predict_sub.to_csv(path_or_buf = 'output/ensemble_predictions.csv')