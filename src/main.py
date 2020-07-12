import cv2 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from math import ceil

def generate_laplacian(num):
    kernel = np.zeros((num, num))

    for i in range(0, num):
        for j in range(0, num):
            bool_list = [i==ceil(num/2)-1, j==ceil(num/2)-1]
            if(bool_list[0] & bool_list[1]):
                kernel[i, j] = 2*num+1
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


jpeg_env = "../Data/sample_jpeg/"

filenames = os.listdir(jpeg_env)
filename = filenames[0]

filepath = jpeg_env + filename
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ssize = 5
lowpass = np.ones((ssize, ssize))/(ssize * ssize)
#img = hair_remove(img)
img = signal.convolve2d(img, lowpass)


kernel = generate_laplacian(9)

feature_map = signal.convolve2d(img, kernel)


plt.figure()
plt.imshow(np.uint8(img), cmap='gray', vmin=0, vmax=255)
plt.figure()
plt.imshow(np.uint8(feature_map), cmap='gray', vmin=0, vmax=255)
plt.show()
