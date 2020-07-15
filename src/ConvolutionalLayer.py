from scipy import signal
import numpy as np

class ConvolutionalLayer:
    def __init__(self, kernel_count, kernel_size, learning_rate):
        self.alpha = learning_rate
        
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.kernel = np.random.random(size=(kernel_count, kernel_size, kernel_size))
        
        self.feature_map = None
        self.feature_map_size = 0

    def forward_prop(self, feature_map):
        self.feature_map = feature_map
        (fd, fl, fw) = feature_map.shape
        self.feature_map_size = fd
        new_feature_map = np.zeros((self.kernel_count*fd, fl-self.kernel_size+1, fw-self.kernel_size+1))
        for feature_idx in range(0, fd):
            for kernel_idx in range(0, self.kernel_count):
                new_feature_map[self.kernel_count * feature_idx + kernel_idx] = signal.convolve2d(feature_map[feature_idx], self.kernel[kernel_idx], mode='valid')
        print("foward_prop bitti")
        return self.ReLU(new_feature_map)

    def backward_prop(self, loss_graident):
        (od, ol, ow) = loss_graident.shape

        assert self.kernel_count == od
        back_prop_loss = np.zeros(self.feature_map.shape)
        
        for map_idx in range(0, self.feature_map_size):
            for kernel_idx in range(0, self.kernel_count):
                rotated_kernel = np.rot90(self.kernel[kernel_idx], 2)

                print("test -> ", rotated_kernel.shape, loss_graident[map_idx].shape)
                back_prop_loss[map_idx] += signal.convolve2d(loss_graident[map_idx], rotated_kernel, mode='full')
                derivative = signal.convolve2d(self.feature_map[map_idx], loss_graident[map_idx], mode='valid')
                self.kernel[kernel_idx] = self.kernel[kernel_idx] - self.alpha * derivative

            back_prop_loss[map_idx] /= self.kernel_count
                
        return back_prop_loss

    def ReLU(self, feature_map):
        return np.maximum(0, feature_map)