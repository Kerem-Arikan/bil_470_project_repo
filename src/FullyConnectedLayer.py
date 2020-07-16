import numpy as np

class FullyConnectedLayer:
    def __init__(self, perc_count, target_count, val_list):
        self.perc_count = perc_count
        self.target_count = target_count 
        self.neurons = []
        self.output_layer = np.zeros((target_count))
        self.final_output = np.zeros((target_count))
        for i in range(0, perc_count):
            self.neurons.append(Perceptron(target_count))


    def forward_prop(self, input_list):
        (prev_layer_perc_count, prev_layer_weight_count) = input_list.shape

        assert prev_layer_weight_count == self.perc_count

        for perc_idx in range(0, self.perc_count):
            total_input = []
            for prev_layer_perc_idx in range(0, prev_layer_perc_count):
                total_input.append(input_list[prev_layer_perc_idx][perc_idx])
            total_input = np.array(total_input)
            self.output_layer += self.neurons[perc_idx].calc_output(total_input)
        
        self.final_output = self.softmax(self.output_layer)
        return self.final_output
    
    def softmax(self, input_list):
        return np.exp(input_list) / np.sum(np.exp(input_list))


    def backward_prop(self, train_targets):
        loss_gradient = np.zeros(shape=(self.perc_count, self.target_count))
        (train_target_count) = train_targets.shape

        assert train_target_count == self.target_count

        for perc_idx in range(0, self.perc_count):
            for weight_idx in range(0, self.target_count):
                weight_gradient = (train_targets[weight_idx] - self.final_output[weight_idx]) * self.neurons[perc_idx].input_ 
                self.neurons[perc_idx].weights[weight_idx] -= weight_gradient
                # todo continue here


                

class Perceptron:
    def __init__(self, weight_count):
        self.output = 0
        self.weight_count = weight_count
        self.weights = []
        for w_idx in range(0, self.weight_count):
            self.weights.append(0.0)
        self.weights = np.array(self.weights)
    
    def ReLU(self, input_):
        return max([0, input_])

    def calc_output(self, input_):
        self.input_ = input_
        self.output = self.ReLU(total_)
        return self.output * self.weights
