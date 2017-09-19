#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 08:15:44 2017
usage: cf=Classyfire()
       cf.class_predict(image_file_path, keras_model_path)
        
@author: laura.astola
This script takes as input 'an image' and 'a pretrained keras-NN-classification-model'
and plots this image with title: 'status' on/off/error' and the 'confidence' related to the particular classificaTION

"""

class Classyfire:

    def class_predict(self, image_file='./new_augmented/Error/sample34.png', keras_model_file= './learned_models/model_vgg16_aws_pt_4.h5'):
        import numpy as np
        import matplotlib.pyplot as plt
        from keras.models import load_model
        import skimage
        from skimage import data
    
        image=skimage.io.imread(image_file)
        image=image/255.
        array=np.expand_dims(image, axis=0)
        
    # load the neural trained network
        model=load_model(keras_model_file)    

    # predict the class of the image
        Y_pred=model.predict(array) 
        Y_class=np.zeros_like(Y_pred)
        Y_class[np.arange(len(Y_pred)),Y_pred.argmax(1)]=1    
    
        status=['Error', 'On', 'Off'][np.where(Y_class==1)[1][0]]
        probs=sorted(Y_pred[0])
        if probs[2]<1.:
            confidence=np.round((probs[2]-probs[1])/(probs[2]-probs[0])*100.0, decimals=2)
        else: # this should not occur as we scale the image values to [0,1], but just in case
            confidence=[90, 86, 88][np.where(Y_class==1)[1][0]]#temporary solution
        
        title_string = 'with '+ str(confidence) + '% confidence, the status is ' + str(status)  
        fig = plt.figure(figsize=(8,8))
        ax1=fig.add_subplot(1,1,1)
        ax1.imshow(image)
        ax1.set_title(title_string)
        plt.show()

        return 

