# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:24:53 2017

@author: Deepti
"""

import cv2;
import os;
import numpy as np;
from sklearn.externals import joblib;

#Predicting the class of images based on the training model

images = [];
color_histograms = [];

#For each image to be classified
for every_image in os.listdir(r'A:\Fall 2017\VD\Projectl\Prediction'):
        img = cv2.imread("A:\Fall 2017\VD\Projectl\Prediction/"+every_image ,0);
        histogram = cv2.calcHist([img],[0],None,[32],[0,256]);
        color_histograms.append([histogram[0][0],histogram[1][0],histogram[2][0],histogram[3][0],histogram[4][0],histogram[5][0],histogram[6][0],histogram[7][0],histogram[8][0],histogram[9][0],histogram[10][0],histogram[11][0],histogram[12][0],histogram[13][0],histogram[14][0],histogram[15][0],histogram[16][0],histogram[17][0],histogram[18][0],histogram[19][0],histogram[20][0],histogram[21][0],histogram[22][0],histogram[23][0],histogram[24][0],histogram[25][0],histogram[26][0],histogram[27][0],histogram[28][0],histogram[29][0],histogram[30][0],histogram[31][0]]);
        images.append(every_image);
        
color_histograms=np.asmatrix(color_histograms);

#Load the model created from training
clf = joblib.load(r'A:\Fall 2017\VD\Project1\TrainingImages\Model.pkl');

#Predict using loaded model
class_prediction = clf.predict(color_histograms);

for sample,class_num in zip(images,class_prediction):
    print(sample+" --> "+str(class_num));