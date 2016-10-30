# -*- coding: utf-8 -*-
'''VGG19 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import AtrousConv2D, Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from imagenet_utils import decode_predictions, preprocess_input


TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG19Dense(weights='imagenet', input_tensor=None):
    '''Instantiate the VGG19 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    # Block 1
    x = AtrousConv2D(64, 3, 3, atrous_rate=(1,1), activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = AtrousConv2D(64, 3, 3, atrous_rate=(1,1), activation='relu', border_mode='same', name='block1_conv2')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = AtrousConv2D(128, 3, 3, atrous_rate=(2,2), activation='relu', border_mode='same', name='block2_conv1')(x)
    x = AtrousConv2D(128, 3, 3, atrous_rate=(2,2), activation='relu', border_mode='same', name='block2_conv2')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = AtrousConv2D(256, 3, 3, atrous_rate=(4,4), activation='relu', border_mode='same', name='block3_conv1')(x)
    x = AtrousConv2D(256, 3, 3, atrous_rate=(4,4), activation='relu', border_mode='same', name='block3_conv2')(x)
    x = AtrousConv2D(256, 3, 3, atrous_rate=(4,4), activation='relu', border_mode='same', name='block3_conv3')(x)
    x = AtrousConv2D(256, 3, 3, atrous_rate=(4,4), activation='relu', border_mode='same', name='block3_conv4')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = AtrousConv2D(512, 3, 3, atrous_rate=(8,8), activation='relu', border_mode='same', name='block4_conv1')(x)
    x = AtrousConv2D(512, 3, 3, atrous_rate=(8,8), activation='relu', border_mode='same', name='block4_conv2')(x)
    x = AtrousConv2D(512, 3, 3, atrous_rate=(8,8), activation='relu', border_mode='same', name='block4_conv3')(x)
    x = AtrousConv2D(512, 3, 3, atrous_rate=(8,8), activation='relu', border_mode='same', name='block4_conv4')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = AtrousConv2D(512, 3, 3, atrous_rate=(16,16), activation='relu', border_mode='same', name='block5_conv1')(x)
    x = AtrousConv2D(512, 3, 3, atrous_rate=(16,16), activation='relu', border_mode='same', name='block5_conv2')(x)
    x = AtrousConv2D(512, 3, 3, atrous_rate=(16,16), activation='relu', border_mode='same', name='block5_conv3')(x)
    x = AtrousConv2D(512, 3, 3, atrous_rate=(16,16), activation='relu', border_mode='same', name='block5_conv4')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Create model
    model = Model(img_input, x)

    # load weights
    if weights == 'imagenet':
        print('K.image_dim_ordering:', K.image_dim_ordering())
        if K.image_dim_ordering() == 'th':
            weights_path = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5',
                                    TH_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model


if __name__ == '__main__':
    model = VGG19Dense(weights='imagenet')

    img_path = 'cat.jpg'
    img = image.load_img(img_path, target_size=(512, 512))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)

    print(preds.shape)

