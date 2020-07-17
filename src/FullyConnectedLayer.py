import numpy as np

class FullyConnectedLayer:
    def __init__(self, perc_count, target_count, learning_rate):
        self.alpha = learning_rate
        self.perc_count = perc_count
        self.target_count = target_count 
        self.neurons = []
        self.output_layer = np.zeros((target_count))
        self.final_output = np.zeros((target_count))
        for i in range(0, perc_count):
            self.neurons.append(Perceptron(target_count))


    def forward_prop(self, input_list):
        (total_pixel_count) = input_list.shape[0]
        assert total_pixel_count == self.perc_count

        for perc_idx in range(0, self.perc_count):
            self.output_layer += self.neurons[perc_idx].calc_output(input_list[perc_idx])
        
        self.final_output = self.softmax(self.output_layer)
        return self.final_output
    
    def softmax(self, input_list):
        beta = -0.00001
        a = np.exp(beta * input_list)
        b = np.sum(np.exp(beta * input_list))
        return  a/b 


    def backward_prop(self, train_targets):
        loss_gradient = np.zeros(shape=(self.perc_count))
        (train_target_count) = train_targets.shape[0]

        assert train_target_count == self.target_count

        for perc_idx in range(0, self.perc_count):
            loss_gradient_sum = 0
            for weight_idx in range(0, self.target_count):
                loss_gradient_sum += np.absolute(train_targets[weight_idx] - self.final_output[weight_idx]) * self.neurons[perc_idx].weights[weight_idx]
                weight_gradient = (train_targets[weight_idx] - self.final_output[weight_idx]) * self.neurons[perc_idx].input_ 
                self.neurons[perc_idx].weights[weight_idx] -= self.alpha * weight_gradient
            loss_gradient[perc_idx] = loss_gradient_sum
                
        return loss_gradient

                

class Perceptron:
    def __init__(self, weight_count):
        self.output = 0
        self.weight_count = weight_count
        self.weights = np.random.random((weight_count))
    
    def ReLU(self, input_):
        return max([0, input_])

    def calc_output(self, input_):
        self.input_ = input_
        self.output = self.ReLU(input_)
        return self.output * self.weights
