import numpy as np 
import pandas as pd
import math
import os
from keras_facenet import FaceNet

embedder = FaceNet()

from tensorflow import keras
model = keras.models.load_model('./Age_Gender_Estimation/data-huber.h5')

def check (image):
    # image = image_alignment(image)
    sample =  np.reshape(image, (1,image.shape[0], -1,3))
    sample = embedder.embeddings(sample)
    # plt.figure()
    # plt.imshow(image)
    res = model.predict(sample)
    
    age = math.ceil(res[0]*116)
    if res[-1] > 0.5:
      gender = "F"
    else:
      gender = "M"
    return age, gender


