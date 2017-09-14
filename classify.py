#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 08:15:44 2017
usage: classify.py -i <input.json> -o <output.json> 
        
@author: laura.astola
This script takes as input 'an image as a json-file' and 'a keras neural network model'
and writes the output: 'the classification status: on/off/error' and
'the confidence of the classification' into a file
"""
import sys
import numpy as np
import urllib.request
import base64
import json
import requests
from keras.models import load_model
import skimage
from skimage import data

#string = urllib.request.urlopen('https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png').read()
#data = base64.b64encode(string)
#json_string=json.dumps({'picture': data})

imf='/home/laura/Documents/PoC_Modems-master/test_set/On/sample36.png'

def main(argv):
    if len(sys.argv) < 2:
        print('Usage: classify.py -i <image> ')
        sys.exit()
    
    # load the json string-data to python
    #json_data = json.loads(sys.argv[1])  

    # read in the base64 encoded image from the data 
    #image = base64.decodestring(json.dumps(json_data)['image'])
    image=skimage.io.imread(sys.argv[1])
    # here comes the model number with best performance so far
    image=np.expand_dims(image, axis=0)
    model_number=6
    # load the neural trained network
    model=load_model('./learned_models/model_'+str(model_number)+'.h5')    

    # predict the class of the image
    Y_pred=model.predict(image) # if image is a nice nparray!
    Y_class=np.zeros_like(Y_pred)
    Y_class[np.arange(len(Y_pred)),Y_pred.argmax(1)]=1    
    
    status=['On', 'Error', 'Off'][np.where(Y_class==1)[0][0]]
    sorted_probs = sorted(Y_pred[0]) 
    confidence = (sorted_probs[2]-sorted_probs[1])/(sorted_probs[2]-sorted_probs[0])*100
     
    data = {'status' : status,
            'confidence' : str(confidence) + '%'
            }

    with open('classified.txt', 'w') as outfile:
            json.dump(data, outfile)

if __name__ == "__main__":
    main(sys.argv[1:])  
            
