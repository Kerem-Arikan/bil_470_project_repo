from scipy import signal
import numpy as np

class ConvolutionalLayer:
    def __init__(self, kernel_count, kernel_size, learning_rate):
        self.alpha = learning_rate
        
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.kernel = np.random.random(size=(kernel_count, kernel_size, kernel_size)) + np.random.randint(low=-1, high=1)
        
        self.feature_map = None
        self.feature_map_size = 0

    def forward_prop(self, feature_map):
        self.feature_map = feature_map
        (fd, fl, fw) = feature_map.shape
        self.feature_map_size = fd
        new_feature_map = np.zeros((self.kernel_count*fd, fl, fw))
        for feature_idx in range(0, fd):
            for kernel_idx in range(0, self.kernel_count):
                new_feature_map[self.kernel_count * feature_idx + kernel_idx] = signal.convolve2d(feature_map[feature_idx], self.kernel[kernel_idx], mode='same')
        return new_feature_map

    def backward_prop(self, loss_graident):
        (od, ol, ow) = loss_graident.shape

        assert self.kernel_count == od

        rotated_kernel = np.rot90(self.kernel, 2)
        back_prop_loss = np.zeros(self.feature_map.shape)
        
        for map_idx in range(0, self.feature_map_size):
            for kernel_idx in range(0, self.kernel_count):
                rotated_kernel = np.rot90(self.kernel[kernel_idx], 2)
                back_prop_loss[map_idx] += signal.convolve2d(rotated_kernel, loss_graident, mode='full')
        
                derivative = signal.convolve2d(self.feature_map[map_idx], loss_graident, mode='same')
                self.kernel[kernel_idx] = self.kernel[kernel_idx] - self.alpha * derivative

        return back_prop_loss
