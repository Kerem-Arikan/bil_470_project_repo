from scipy import signal
import numpy as np

class ConvolutionalLayer:
    def __init__(self, filter_count, kernel_count, kernel_size, learning_rate):
        self.alpha = learning_rate
        self.feature_map_size = kernel_count
        self.filter_count = filter_count
        self.kernel_size = kernel_size
        self.kernel = np.random.uniform(low=-4, high=4, size=(self.filter_count, self.feature_map_size, kernel_size, kernel_size))
        
        self.feature_map = None
        self.output_map = None

    def forward_prop(self, feature_map):
        self.feature_map = feature_map
        (fd, fl, fw) = self.feature_map.shape
        assert self.feature_map_size == fd
        
        new_feature_map = np.zeros((self.filter_count, fl-self.kernel_size+1, fw-self.kernel_size+1))

        for filter_idx in range(0, self.filter_count):
            for kernel_idx in range(0, self.feature_map_size):
                new_feature_map[filter_idx] += signal.convolve2d(feature_map[kernel_idx], self.kernel[filter_idx][kernel_idx], mode='valid')
        self.output_map = self.ReLU(new_feature_map)        
        return self.ReLU(new_feature_map)

    def backward_prop(self, loss_graident):
        (od, ol, ow) = loss_graident.shape

        assert self.filter_count == od

        fm_shape = self.feature_map.shape
        back_prop_loss = np.zeros(fm_shape)
        fm_dim = (fm_shape[1], fm_shape[2])
        
        for filter_idx in range(0, self.feature_map_size):
            temp_fm_loss = np.zeros(shape=fm_dim)
            for kernel_idx in range(0, self.feature_map_size):

                rotated_kernel = np.rot90(self.kernel[filter_idx][kernel_idx], 2)

                temp_fm_loss += signal.convolve2d(loss_graident[kernel_idx], rotated_kernel, mode='full')

                derivative = signal.convolve2d(self.output_map[kernel_idx], loss_graident[kernel_idx], mode='valid')    
                #print("DERIVATIVE\n", derivative, "\nFEATURE MAP\n", self.feature_map[kernel_idx], "\nLOSS GRADIENT\n", loss_graident[kernel_idx])
                self.kernel[filter_idx][kernel_idx] += self.alpha * derivative 
                
            #print(self.kernel[filter_idx][0])
            back_prop_loss[filter_idx] = temp_fm_loss
        return back_prop_loss

    def ReLU(self, feature_map):
        return np.maximum(0, feature_map)
