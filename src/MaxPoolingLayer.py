import numpy as np
from numpy import *
import scipy.ndimage


class MaxPooling(object):

    def __init__(self, subset_size):
        self.feature_map = None
        self.feature_map_size = 0
        self.Coordinates = []
        self.subset_size = subset_size

    def maxval(self, data, x, y):
        r, h = data.shape
        val = data[0, 0]
        for i in range(r):
            for j in range(h):
                val = max(data[i, j], val)
        for i in range(r):
            for j in range(h):
                if(data[i, j] == val):
                    self.Coordinates.append([x+i, y+j])

        return val

    def pool(self, data):
        r, h = data.shape
        sub_data = (data.reshape(
            h//self.subset_size, self.subset_size, -1, self.subset_size).swapaxes(1, 2).reshape(-1, self.subset_size, self.subset_size))
        result = np.zeros(
            shape=(int(r/self.subset_size), int(h/self.subset_size)))
        index = 0

        for i in range(int(r/self.subset_size)):
            for j in range(int(h/self.subset_size)):
                result.itemset((i, j), self.maxval(
                    sub_data[index], i*self.subset_size, j*self.subset_size))
                index += 1

        return result

    def forward_prop(self, feature_map):
        self.feature_map = feature_map
        (fd, fl, fw) = feature_map.shape
        self.feature_map_size = fd
        new_feature_map = np.zeros(
            (fd, int(fl/self.subset_size), int(fw/self.subset_size)))
        self.Coodinates = []
        for feature_idx in range(fd):
            new_feature_map[feature_idx] = self.pool(feature_map[feature_idx])

        return new_feature_map

    def backward_prop(self, loss_graident):
        (od, ol, ow) = loss_graident.shape
        back_prop_loss = np.zeros((od, ol*self.subset_size, ow*self.subset_size))
        for i in range(od):
            back_prop_loss[i] = scipy.ndimage.zoom(
                loss_graident[i], self.subset_size, order=0)
            for j in range(ol*self.subset_size):
                for k in range(ow*self.subset_size):
                    if([j, k] not in [l for l in self.Coordinates[i*(int(len(self.Coordinates)/od)):(i+1)*(int(len(self.Coordinates)/od))]]):
                        back_prop_loss.itemset((i, j, k), 0)
        return back_prop_loss
