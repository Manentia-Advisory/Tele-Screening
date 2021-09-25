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

img_size = (224, 224)

preprocess_input = keras.applications.inception_v3.preprocess_input
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap
    
def gradCam(img_path, model):
    last_conv_layer_name = "conv2d_250"
    classifier_layer_names = [
        "activation_246",

        "conv2d_251",
        "activation_247",

        "conv2d_252",

        "conv2d_253",
        "activation_248",

        "conv2d_254",
        "activation_249",

        # "multiply_49",
        # "add_49",
        "activation_250",
        "global_average_pooling2d_50",
        "dropout_1",
        "dense"
    ]
    image = load_img(img_path, target_size=(256, 256,3))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(image, model, last_conv_layer_name, classifier_layer_names)

    # Display heatmap
    # plt.matshow(heatmap)
    # plt.show()
    # ## Create a superimposed visualization

    # We load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    save_path = "media/XRay_GradCam/grad_cam1.jpg"
    print("Over")

    superimposed_img.save(save_path)
    return superimposed_img