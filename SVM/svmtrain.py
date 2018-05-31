# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:14:17 2017

@author: Deepti
"""

import cv2;
import os;
import numpy as np;
from sklearn import svm;
from sklearn.externals import joblib;

# Creating Color Histograms and assigning class number to each class
class_cnt = 0;
color_histograms = [];
class_num = [];

#For each dataset in the training samples
for dataset in os.listdir(r'A:\Fall 2017\VD\Project\TrainingImages'):
    num_images = 0;
    class_cnt = class_cnt +1; 
    #For each image in current dataset
    for every_image in os.listdir("A:\Fall 2017\VD\Project\TrainingImages/"+dataset):
        print ("Image: "+every_image+" Label: "+str(class_cnt)+" Dataset: "+dataset);
        num_images = num_images +1;
        #Limit number of images selected to 1000 
        if(num_images > 1000):
            break;
        img = cv2.imread("A:\Fall 2017\VD\Project\TrainingImages/"+dataset+"/"+every_image ,0);
        histogram = cv2.calcHist([img],[0],None,[32],[0,256]);
        color_histograms.append([histogram[0][0],histogram[1][0],histogram[2][0],histogram[3][0],histogram[4][0],histogram[5][0],histogram[6][0],histogram[7][0],histogram[8][0],histogram[9][0],histogram[10][0],histogram[11][0],histogram[12][0],histogram[13][0],histogram[14][0],histogram[15][0],histogram[16][0],histogram[17][0],histogram[18][0],histogram[19][0],histogram[20][0],histogram[21][0],histogram[22][0],histogram[23][0],histogram[24][0],histogram[25][0],histogram[26][0],histogram[27][0],histogram[28][0],histogram[29][0],histogram[30][0],histogram[31][0]]);
        class_num.append(class_cnt);
  
color_histograms=np.asmatrix(color_histograms);
print "Histograms formed: ", color_histograms

#Perform SMV classification
print "Calling sklearn svm.linearSVC"

#Do LinearSVC for doing one to many type of classification
clf = svm.LinearSVC();
clf.fit(color_histograms,class_num);

#Create a Model file to use for prediction
joblib.dump(clf, r'A:\Fall 2017\VD\Project\TrainingImages\Model.pkl')