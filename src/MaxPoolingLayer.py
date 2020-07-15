import numpy as np


class MaxPoolingLayer():
    @staticmethod
    def maxval(data):
        r, h = data.shape
        val = data[0, 0]
        for i in range(r):
            for j in range(h):
                val = max(data[i, j], val)
        return val

    @staticmethod
    def pool(data, n):
        r, h = data.shape
        sub_data = (data.reshape(
            h//n, n, -1, n).swapaxes(1, 2).reshape(-1, n, n))
        result = np.zeros(shape=(int(r/n), int(h/n)))
        index = 0
        for i in range(int(r/n)):
            for j in range(int(h/n)):
                result.itemset((i, j), Max.maxval(sub_data[index]))
                index += 1

        return result
