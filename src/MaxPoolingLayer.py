import numpy as np
from numpy import *


class Max(object):

    def __init__(self, learning_rate):
        self.alpha = learning_rate
        self.feature_map = None
        self.feature_map_size = 0

    def maxval(self, data):
        r, h = data.shape
        val = data[0, 0]
        for i in range(r):
            for j in range(h):
                val = max(data[i, j], val)
        return val

    def pool(self, data, n):
        r, h = data.shape
        sub_data = (data.reshape(
            h//n, n, -1, n).swapaxes(1, 2).reshape(-1, n, n))
        result = np.zeros(shape=(int(r/n), int(h/n)))
        index = 0
        for i in range(int(r/n)):
            for j in range(int(h/n)):
                result.itemset((i, j), self.maxval(sub_data[index]))
                index += 1

        return result

    def forward_prop(self, feature_map, subset_size):
        self.feature_map = feature_map
        (fd, fl, fw) = feature_map.shape
        self.feature_map_size = fd
        new_feature_map = np.zeros((self.kernel_count*fd, fl, fw))
        for feature_idx in range(0, fd):
            new_feature_map[feature_idx] = self.pool(feature_map[feature_idx], subset_size)
        new_feature_map = self.pool(feature_map, subset_size)
        return new_feature_map

    def backward_prop(self, loss_graident):
        (od, ol, ow) = loss_graident.shape

        assert self.kernel_count == od
        back_prop_loss = np.zeros(self.feature_map.shape)

        for map_idx in range(0, self.feature_map_size):

        return back_prop_loss