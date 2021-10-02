# Machine Learning Libraries
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
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

def gradCam(img_path, model, ID_millis):
    # last_conv_layer_name = "input_1"
    # classifier_layer_names = [
    #     "model",
    #     "global_average_pooling2d_50",
    #     "dropout_1",
    # ]
    # last_conv_layer_name = "conv2d_251"
    # classifier_layer_names = [
    #     # "activation_247",
    #     "conv2d_252",
    #     "conv2d_253",
    #     "activation_248",

    #     "conv2d_254",
    #     "activation_249",

    #     # "multiply_49",
    #     # "add_49",
    #     "activation_250",
    #     # "dropout",

    #     "global_average_pooling2d_50",
    #     "dropout_1",
    #     "dense"
    # ]
    last_conv_layer_name = "conv2d_250"
    classifier_layer_names = [
        # "activation_244",
        # "conv2d_249",
        # "activation_244",
        # "multiply_48",
        # "add_48",
        # "activation_245",

        # "conv2d_250",
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
        # "dropout",

        "global_average_pooling2d_50",
        "dropout_1",
        "dense"
    ]
    # Prepare image
    # orig = cv2.imread(img_path)
    # resized = cv2.resize(orig, (224, 224))
    image = load_img(img_path, target_size=(256, 256,3))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # img_array = preprocess_input(get_img_array(resized, size=(299,299,3)))
    # image = imagenet_utils.preprocess_input(image)
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
    save_path = "media/Image/{0}/grad_cam1.jpg".format(ID_millis)
    print("Over")

    superimposed_img.save(save_path)
    return superimposed_img

def xray_gradcam(temp2, ImageName, ID_millis):
    nb_classes = 5   # number of classes
    img_width, img_height = 256, 256  # change based on the shape/structure of your images
    img_size = 256
    learn_rate = 0.0001  # sgd learning rate
    seresnet152, _ = Classifiers.get('seresnet152')
    base = seresnet152(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    x = base.output
    x = layers.GlobalAveragePooling2D()(layers.Dropout(0.16)(x))
    x = layers.Dropout(0.3)(x)
    preds = layers.Dense(nb_classes, 'sigmoid')(x)
    multiClassModel=Model(inputs=base.input,outputs=preds)
    loss= tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0)
    multiClassModel.compile(optimizers.Adam(lr=learn_rate),loss=loss,metrics=[tf.keras.metrics.AUC(multi_label=True)])
    multiClassModel.load_weights('./model/customTLWeights_NEW_WITH_TB.h5')
    resultImg = gradCam(temp2, multiClassModel, ID_millis) 
    res_path = "/media/Image/{0}/".format(ID_millis) + ImageName
    
    return resultImg, res_path, multiClassModel

def predict_xray_for_5_diseases(testimage, multiClassModel):
    x = load_img(testimage, target_size=(256,256))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    return multiClassModel.predict(x)

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def lung_segment(img_path, ID_millis, django_generated_image_name):
    lung_segment_model = load_model('./model/cxr_reg_model.h5', custom_objects={'dice_coef_loss':                   
    dice_coef_loss, 'dice_coef': dice_coef})
    # img_path = "./media/"+path
    print(img_path)
    print("^^^^^^^^^^^^")
    x_im = cv2.resize(cv2.imread(img_path),(512, 512))[:,:,0]
    # a = model.predict((im.reshape(1, 512, 512, 1)-127.0)/127.0)
    # img = np.squeeze(a)*255
    op = lung_segment_model.predict((x_im.reshape(1, 512, 512, 1)-127.0)/127.0)
    plt.imshow(x_im, cmap="bone")
    plt.imshow(op.reshape(512, 512), alpha=0.5, cmap="jet")
    plt.axis('off')
    save_path = "/media/Image/{0}/".format(ID_millis)+ "lung_segment_" + django_generated_image_name
    plt.savefig("."+save_path, bbox_inches='tight')
    return save_path

def image_ct_or_not(img_path):
    classifier = keras.models.load_model("./model/ctscan_not.h5")
    img_pred = keras.preprocessing.image.load_img(img_path, target_size = (64, 64))
    img_pred = keras.preprocessing.image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis = 0)

def CTget_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def CTmake_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
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

def CTgradCam(img_path, model):
    last_conv_layer_name = "block4_conv1"
    classifier_layer_names = [
        "block4_conv2",
        "block4_conv3",

        "block4_pool",
        
        "block5_conv1",
        "block5_conv2",
        "block5_conv3",
        "block5_pool",
        "flatten",
        "dense",
    ]
    # Prepare image
    preprocess_input = keras.applications.xception.preprocess_input
    img_array = preprocess_input(CTget_img_array(img_path, size=(224,224,3)))

    # Generate class activation heatmap
    heatmap = CTmake_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)

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
    save_path = "media/CTScan_GradCam/grad_cam1.jpg"

    superimposed_img.save(save_path)
    

    # Display Grad CAM
    # IPython.display.display(Image(superimposed_img))
    # plt.matshow(superimposed_img)
    # plt.show()
    return superimposed_img
    print("Over")

def predict_ct_scan(testimage, ImageName, ID_millis):
    CTcovidNormal = load_model('model/ctscan_VGG16.h5')

    image = cv2.imread(testimage) # read file
    
    resultImg = CTgradCam(testimage, CTcovidNormal)
    res_path = "media/Image/{0}/GradCam_".format(ID_millis)
    res_path += ImageName
    print(res_path)
    resultImg.save(res_path)
    res_path = "/" + res_path

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
    image = cv2.resize(image,(224,224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)

    CTcovidNormal = CTcovidNormal.predict(image)
    print(CTcovidNormal)
    probabilityCTcovidNormal = CTcovidNormal[0]
    return res_path, probabilityCTcovidNormal





