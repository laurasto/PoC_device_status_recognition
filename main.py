#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:21:49 2017

This program has 3 modes: 'train', 'test' and 'classify'.

train: read data from 3 folders in the train directory.Each folder contains 
data that belong to a certain class. Train a classifier 'model' and save it to disk.

test: load the trained classifier 'model' apply it to the data in 3 folders in the test 
directory and report results.

classify: load an image from the modem foto folder, classify it and print results.


This version is extended from program created by martin.gullaksen by laura.astola.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import os

from datetime import datetime

from skimage import exposure
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import SGD, Adam, rmsprop
from keras import metrics
from DataProcessing import PreProcessor
from NetworkGenerator import NetworkGenerator
from sklearn.model_selection import train_test_split


# Set to 'TRAIN' or 'TEST'
mode = 'TRAIN'  
#mode = 'TEST'
#mode = ´CLASSIFY

im_length=320
im_width=180
maximum=300# MAXIMUM NUMBER OF IMAGES PER CLASS

train_data_folder='./processed_data'
test_data_folder='./processed_data_test'
model_folder='./learned_models'
modem_foto_folder='./upload_folder'

dataset = PreProcessor() 

# VIDEO TO IMAGE
#  extracts raw images from video, e.g. by capturing every 24th frame
#  does not have to be run over and over again
#dataset.video_to_images(image_path='./raw_pictures_test', video_path='./trimmed_videos',\
#                        size_pics=(im_length,im_width), frame_spacing=18) 
# IMAGES TO AUGMENTED IMAGE DATA SETS
#  augments the (manually cleaned) raw images to form a richer training set
#  does not have to be run over and over again
#dataset.augment_data(image_path='./raw_pictures_test', augment_path='./processed_data_test', \
#                     batch_size=5)

# BALANCE THE IMAGE DATA SETS
#  balances the training sets to have equal number of samples
#  does not have to be run over and over again
dataset.balance_image_classes(maximum, augment_path='./processed_data')

#  gives the size of the images to be classified 
size_pics = dataset.size_pics 
   
net_gen = NetworkGenerator()

#  choose one of the different network architectures
model, net_name = net_gen.get_vgg_net(pic_size=size_pics)
#network, net_name = net_gen.get_convnet(pic_size=size_pics)
#network, net_name = net_gen.get_alex_net(pic_size=size_pics)
#network, net_name = net_gen.get_highway(pic_size=size_pics)

#  choose the parameters for the network model
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.01, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt,\
              metrics=[metrics.categorical_accuracy])

#********************************training
if mode == 'TRAIN':
    
#  loads the training data as well as their labels
    X, y = dataset.load_data(train_data_folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)
    X_train=X_train.reshape(X_train.shape[0],im_width,im_length,3)
    X_test=X_test.reshape(X_test.shape[0],im_width,im_length,3)
#    inputshape=(256,256,3)

    X_train/=255. # comment these out when preprocessing
    X_test/=255.

# optional image processing    
# process the images before training
#    for index, array in enumerate(X_train):
#        X_train[index]=dataset.process_array(array)
#        
#    for index, array in enumerate(X_test):
#        X_test[index]=dataset.process_array(array)  
##        
        

#   fit a model    
    history=model.fit(X_train, y_train, batch_size=16, epochs=10)
    scores=model.evaluate(X_test, y_test, batch_size=16,verbose=2)


    #  check if destination folder present
    if not os.path.exists(model_folder):
        os.makedirs(model_folder) 
    model_number= len(os.listdir(model_folder))+1
    model.save(model_folder+'/model_'+str(model_number)+'.h5')

    file = open('/home/laura/Documents/PoC_Modems-master/project_reports/deep_learning_results.txt','a')
    file.write('time:'+str(datetime.now())+'categorical accuracies'+\
               str(history.history['categorical_accuracy'])+\
               ', scores:' + str(scores) + ',  **comments here** model number'+ str(model_number) )
    file.close()
    
#*****************************testing
elif mode == 'TEST':
    
# load the model fitted with training data
    
    model=load_model(model_folder+'/model_'+str(model_number)+'.h5')


# loads the testing data as well as their labels
    X, Y = dataset.load_data(test_data_folder)
    X=X.reshape(X.shape[0],im_width,im_length,3)
    X/=255.
#    for index, array in enumerate(X):
#        X[index]=dataset.process_array(array)    

    folders = [folder for folder in os.listdir(test_data_folder)]
    prediction_accuracies=[]
    folder_start_index=0
    col_number=0

    
    for folder in folders:
    
        number_of_predictions=len(os.listdir(test_data_folder+'/'+folder)) 
        Y_pred=model.predict(X[folder_start_index:folder_start_index+number_of_predictions])
        
# to compute binarized accuracy: set the class with highest prob. to one, rest to zero       
        Y_class=np.zeros_like(Y_pred)
        Y_class[np.arange(len(Y_pred)),Y_pred.argmax(1)]=1

        accuracy=(1-sum(abs(Y[folder_start_index:folder_start_index+number_of_predictions,col_number]
                            -Y_class[:,col_number]))/number_of_predictions)*100
        
        print("real statuses = " + str(np.sum(Y[folder_start_index:folder_start_index+number_of_predictions],axis=0)))
        print("predicted statuses = " + str(np.sum(Y_class,axis=0)))
        print('accurately predicted class %s in %.2f percent of the cases' %(folder,accuracy))
        
        prediction_accuracies.append(accuracy)
        folder_start_index+=number_of_predictions
        col_number+=1

        
#********************************training
if mode == 'CLASSIFY':        

    
    model=load_model(model_folder+'/model.h5')    
    image_to_be_classified=os.listdir(modem_foto_folder)[0]
    array = dataset.load_image(modem_foto_folder + '/' + image_to_be_classified)
    predicted_status = model.predict([array])
    idx = predicted_status[predicted_status!=0]
    print('modem status is ' + os.listdir(test_data_folder)[idx])
    