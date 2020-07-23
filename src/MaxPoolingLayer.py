import numpy as np
from numpy import *
import scipy.ndimage
import skimage.measure
import math
class MaxPooling(object):

    def __init__(self, subset_size):
        self.feature_map = None
        self.feature_map_size = 0
        self.Coordinates = []
        self.subset_size = subset_size



    def forward_prop(self, feature_map):
        self.feature_map = feature_map
        (fd, fl, fw) = feature_map.shape
        self.feature_map_size = fd
        new_feature_map = np.zeros(((fd, round(fl/self.subset_size), round(fw/self.subset_size))))
        (nd, nl, nw) = new_feature_map.shape
        self.Coordinates = np.zeros((fd, fl, fw))

        for i in range(fd):
            new_feature_map[i] = skimage.measure.block_reduce(feature_map[i], (self.subset_size,self.subset_size), np.max)
            for j in range(nl):
                for k in range(nw):
                    little_feature_matrix = feature_map[i][k*self.subset_size:(k+1)*self.subset_size,j*self.subset_size:(j+1)*self.subset_size]
                    coor = np.where(little_feature_matrix == new_feature_map[i,k,j])
                    self.Coordinates[i,j*self.subset_size+coor[0][0],k*self.subset_size+coor[1][0]] = 1
        return new_feature_map


    def backward_prop(self, loss_graident):
        (od, ol, ow) = loss_graident.shape
        back_prop_loss = np.zeros((od, ol*self.subset_size, ow*self.subset_size))
        for i in range(od):
            enlargened_gradient = scipy.ndimage.zoom( loss_graident[i], self.subset_size, order=0)
            back_prop_loss[i] = np.multiply(enlargened_gradient,self.Coordinates[i])
        return back_prop_loss


