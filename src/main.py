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
def ReLU(input):
    return np.maximum(0, input)


def maxpool2(input):
    (fl, fw) = input.shape
    res = np.zeros((ceil(fl/2), ceil(fw/2)))

    for i in range(0,fl-1, 2):
        for j in range(0, fw-1, 2):
            res[ceil(i/2), ceil(j/2)] = input[i:i+1, j:j+1].max()

    return res

def generate_laplacian(num):
    kernel = np.zeros((num, num))

    for i in range(0, num):
        for j in range(0, num):
            bool_list = [i==ceil(num/2)-1, j==ceil(num/2)-1]
            if(bool_list[0] & bool_list[1]):
                kernel[i, j] = 2*num
            elif(bool_list[0] | bool_list[1]):
                kernel[i, j] = -1
    return kernel

def hair_remove(image):
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))

    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)

    return final_image

def flatten(feature_map):
    (fd, fl, fw) = feature_map.shape
    return feature_map.reshape(fd*fl*fw)

jpeg_env = "../Data/sample_jpeg/"

filenames = os.listdir(jpeg_env)
filename = "ISIC_0074268.jpg"#filenames[12]

filepath = jpeg_env + filename
feature_map = cv2.imread(filepath)
print(feature_map)
feature_map = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
init_image = feature_map
(fl, fw) = feature_map.shape
feature_map = np.array(feature_map).reshape((1, fl, fw))
feature_map = feature_map / 255
'''

init_image = feature_map

ssize = 5
lowpass = np.ones((ssize, ssize))/(ssize * ssize)
#feature_map = hair_remove(feature_map)

laplacian = generate_laplacian(5)
kernel = signal.convolve2d(lowpass, laplacian)

plt.figure()
plt.imshow(np.uint8(feature_map), cmap='gray', vmin=0, vmax=255)
plt.figure()
plt.imshow(np.uint8(init_image), cmap='gray', vmin=0, vmax=255)
plt.show()
'''


rate = 0.01
conv_layer = conv(4, 3, rate)
conv_layer2 = conv(4, 3, rate)
mpool_layer =  mpool(2)
mpool_layer2 =  mpool(2)


feature_map = conv_layer.forward_prop(feature_map)

feature_map = mpool_layer.forward_prop(feature_map)

feature_map = conv_layer2.forward_prop(feature_map)

feature_map = mpool_layer2.forward_prop(feature_map)

fc_layer = fc(feature_map.size, 2,rate)

output = fc_layer.forward_prop(flatten(feature_map))


print(output)



#
# plt.figure()
# plt.imshow(np.uint8(feature_map[3]), cmap='gray', vmin=0, vmax=255)
# plt.figure()
# plt.imshow(np.uint8(feature_map[2]), cmap='gray', vmin=0, vmax=255)
# plt.figure()
# plt.imshow(np.uint8(init_image), vmin=0, vmax=255)
# plt.show()