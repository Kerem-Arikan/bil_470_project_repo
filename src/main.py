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
import random

def flatten(feature_map):
    (fd, fl, fw) = feature_map.shape
    return feature_map.reshape(fd*fl*fw)
def make_tensor(feature_map, shape):
    return feature_map.reshape(shape)


# most right one is with relu back prop

print("Prepairing Data...")
cancer = dt(csv_file="../Data/train.csv",path="../Data/train/",size=500,target=[1])
healty = dt(csv_file="../Data/train.csv",path="../Data/train/",size=500,target=[0])

samples = []
tests = []
for i in range(400):
    samples.append(cancer.nextSample())

for i in range(400):
    samples.append(healty.nextSample())



while (cancer.hasSample()):
    tests.append(cancer.nextSample())

while (healty.hasSample()):
    tests.append(healty.nextSample())


random.shuffle(samples)
random.shuffle(tests)

rate = 0.01
conv_layer_rate = 0.1
arr = []

conv_layer = conv(5, 1, 3, conv_layer_rate)
conv_layer2 = conv(5, 5, 5, conv_layer_rate)
conv_layer3 = conv(5, 5, 5, conv_layer_rate)
conv_layer4 = conv(5, 5, 5, conv_layer_rate)

mpool_layer =  mpool(2)
mpool_layer2 =  mpool(2)
mpool_layer3 =  mpool(2)
mpool_layer4 =  mpool(2)
fc_layer = fc(5*29*29, 2,rate)
epoch = 20

print("Training...")
cancer_correct_count = 0
cancer_count = 0
healty_correct_count = 0
healthy_count = 0

for i in range(epoch):
    for sample in range(len(samples)):

        feature_map, target = samples[sample]
        init_image = feature_map
        (fd, fl, fw) = feature_map.shape
        feature_map = np.array(feature_map).reshape((3, fl, fw))
        feature_map = feature_map / 255

        feature_map = conv_layer.forward_prop(feature_map)

        feature_map = mpool_layer.forward_prop(feature_map)

        feature_map = conv_layer2.forward_prop(feature_map)

        feature_map = mpool_layer2.forward_prop(feature_map)
   
        feature_map = conv_layer3.forward_prop(feature_map)

        feature_map = mpool_layer3.forward_prop(feature_map)

        feature_map = conv_layer4.forward_prop(feature_map)

        feature_map = mpool_layer4.forward_prop(feature_map)



        tensor_shape = feature_map.shape
 
        input_map = flatten(feature_map)

        prediction = fc_layer.forward_prop(input_map)

        

        if(target == [1, 0]):
            cancer_count += 1
        else:
            healthy_count += 1

        if(target == [1, 0] and prediction[0] > prediction[1]):
            cancer_correct_count += 1
        elif(target == [0, 1] and prediction[0] <= prediction[1]):
            healty_correct_count += 1


        print(prediction,"predict:",[1, 0] if prediction[0]>prediction[1] else [0, 1],"real:",target)

        loss_gradient = fc_layer.backward_prop(np.array(target))

        loss_gradient = mpool_layer4.backward_prop(make_tensor(loss_gradient, tensor_shape))
        loss_gradient = conv_layer4.backward_prop(loss_gradient)  

        loss_gradient = mpool_layer3.backward_prop(loss_gradient)
        loss_gradient = conv_layer3.backward_prop(loss_gradient)  

        loss_gradient = mpool_layer2.backward_prop(loss_gradient)
        loss_gradient = conv_layer2.backward_prop(loss_gradient)  
        loss_gradient = mpool_layer.backward_prop(loss_gradient)
        loss_gradient = conv_layer.backward_prop(loss_gradient)



print("cancer_count",cancer_count,"cancer_correct_count",cancer_correct_count)
print("healthy_count",healthy_count,"healty_correct_count",healty_correct_count)
print("train_size",epoch*8300,"correct_count",healty_correct_count+cancer_correct_count,"success_rate",(100*(healty_correct_count+cancer_correct_count))/epoch*8300)    


print("Testing Test Error...")
cancer_correct_count = 0
cancer_count = 0
healty_correct_count = 0
healthy_count = 0
for sample in range(len(tests)):

    feature_map, target = tests[sample]
    init_image = feature_map
    (fd, fl, fw) = feature_map.shape
    feature_map = np.array(feature_map).reshape((3, fl, fw))
    feature_map = feature_map / 255


    feature_map = conv_layer.forward_prop(feature_map)
    feature_map = mpool_layer.forward_prop(feature_map)
    feature_map = conv_layer2.forward_prop(feature_map)
    feature_map = mpool_layer2.forward_prop(feature_map)
    
    tensor_shape = feature_map.shape

    prediction = fc_layer.forward_prop(flatten(feature_map))

    if(target == [1, 0]):
        cancer_count += 1
    else:
        healthy_count += 1

    if(target == [1, 0] and prediction[0] > prediction[1]):
        cancer_correct_count += 1
    elif(target == [0, 1] and prediction[0] <= prediction[1]):
        healty_correct_count += 1

    

print("cancer_count",cancer_count,"cancer_correct_count",cancer_correct_count,"success_rate",(100*(cancer_correct_count))/cancer_count)
print("healthy_count",healthy_count,"healty_correct_count",healty_correct_count,"success_rate",(100*(healty_correct_count))/healthy_count)
print("test_size",200,"correct_count",healty_correct_count+cancer_correct_count,"success_rate",(100*(healty_correct_count+cancer_correct_count))/200)