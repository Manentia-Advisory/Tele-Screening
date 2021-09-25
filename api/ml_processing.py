# Machine Learning Libraries
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

from tensorflow import keras
import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from tensorflow.keras import Model
from keras import optimizers
from classification_models.tfkeras import Classifiers
# from .gradCAM2 import *

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import IPython
import keras
import cv2 
# from demo.HetMap import *

# identify image is xray or not

def image_xray_or_not(img_path):
    classifier = keras.models.load_model("./model/xray_not.h5")
    img_pred = keras.preprocessing.image.load_img(img_path, target_size = (64, 64))
    img_pred = keras.preprocessing.image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis = 0)
    rslt = classifier.predict(img_pred)
    if rslt[0][0] == 1:
        return False # image is not an xray image
    return True # it is an xray image