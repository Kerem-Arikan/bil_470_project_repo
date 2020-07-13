from scipy import signal
import numpy as np

class ConvolutionalLayer:
    def __init__(self, kernel_count, kernel_size, learning_rate):
        self.alpha = learning_rate
        
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.kernel = np.random.random(size=(kernel_count, kernel_size, kernel_size)) + np.random.randint(low=-1, high=1)
        
        self.feature_map = None

    def forward_prop(self, feature_map):
        self.feature_map = feature_map
        (fd, fl, fw) = feature_map.shape
        new_feature_map = np.zeros((self.kernel_count*fd, fl, fw))
        for feature_idx in range(0, fd):
            for kernel_idx in range(0, self.kernel_count):
                new_feature_map[self.kernel_count * feature_idx + kernel_idx] = signal.convolve2d(feature_map[feature_idx], self.kernel[kernel_idx], mode='same', boundary='fill', fillvalue=0)
        return new_feature_map

    def backward_prop(self, maxpool_output):
        new_kernel = self.kernel
        (od, ol, ow) = maxpool_output.shape

        assert self.kernel_count == od

        

        for kernel_idx in range(0, self.kernel_count):


        return 0
