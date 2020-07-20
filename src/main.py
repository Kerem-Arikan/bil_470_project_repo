import cv2 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from math import ceil
from math import exp
from ConvolutionalLayer import ConvolutionalLayer as conv
from MaxPoolingLayer import MaxPooling as mpool
from FullyConnectedLayer import FullyConnectedLayer as fc
from Data import Data as dt

def flatten(feature_map):
    (fd, fl, fw) = feature_map.shape
    return feature_map.reshape(fd*fl*fw)
def make_tensor(feature_map, shape):
    return feature_map.reshape(shape)





cancer = dt(csv_file="../Data/train.csv",path="../Data/train/",size=100,target=[1])
healty = dt(csv_file="../Data/train.csv",path="../Data/train/",size=500,target=[0])


rate = 0.01
arr = []
conv_layer = conv(4, 3, rate)
conv_layer2 = conv(4, 3, rate)
mpool_layer =  mpool(2)
mpool_layer2 =  mpool(2)
fc_layer = fc(258064, 2,rate)

while cancer.hasSample():

    feature_map = cancer.nextSample()
    init_image = feature_map
    (fl, fw) = feature_map.shape
    feature_map = np.array(feature_map).reshape((1, fl, fw))
    feature_map = feature_map / 255


    feature_map = conv_layer.forward_prop(feature_map)
    feature_map = mpool_layer.forward_prop(feature_map)
    feature_map = conv_layer2.forward_prop(feature_map)
    feature_map = mpool_layer2.forward_prop(feature_map)

    tensor_shape = feature_map.shape

    prediction = fc_layer.forward_prop(flatten(feature_map))


    loss_gradient = fc_layer.backward_prop(np.array([0,1]))
    loss_gradient = mpool_layer2.backward_prop(make_tensor(loss_gradient, tensor_shape))

    loss_gradient = conv_layer2.backward_prop(loss_gradient)  
    loss_gradient = mpool_layer.backward_prop(loss_gradient)
    loss_gradient = conv_layer.backward_prop(loss_gradient)

while healty.hasSample():
    feature_map = healty.nextSample()
    init_image = feature_map
    (fl, fw) = feature_map.shape
    feature_map = np.array(feature_map).reshape((1, fl, fw))
    feature_map = feature_map / 255


    feature_map = conv_layer.forward_prop(feature_map)
    feature_map = mpool_layer.forward_prop(feature_map)
    feature_map = conv_layer2.forward_prop(feature_map)
    feature_map = mpool_layer2.forward_prop(feature_map)

    tensor_shape = feature_map.shape

    prediction = fc_layer.forward_prop(flatten(feature_map))
    print(prediction)

    loss_gradient = fc_layer.backward_prop(np.array([1,0]))
    loss_gradient = mpool_layer2.backward_prop(make_tensor(loss_gradient, tensor_shape))
    print("loss gradient")
    print(loss_gradient)
    loss_gradient = conv_layer2.backward_prop(loss_gradient)   
    loss_gradient = mpool_layer.backward_prop(loss_gradient)
    loss_gradient = conv_layer.backward_prop(loss_gradient)
    print(loss_gradient)

