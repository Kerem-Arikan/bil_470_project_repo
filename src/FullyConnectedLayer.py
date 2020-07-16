import numpy as np

class FullyConnectedLayer:
    def __init__(self, perc_count, weight_count, val_list):
        self.perc_count = perc_count
        self.weight_count = weight_count
        self.neurons = []
        for i in range(0, perc_count):
            self.neurons.append(Perceptron(val_list[i], weight_count))
            for w_idx in range(0, weight_count):


    def forward_prop(self, input_list):
        output_list = np.zeros((self.perc_count, self.weight_count))
        (prev_layer_perc_count, prev_layer_weight_count) = input_list.shape

        assert prev_layer_weight_count == self.perc_count

        for perc_idx in range(0, self.perc_count):
            total_input = []
            for prev_layer_perc_idx in range(0, prev_layer_perc_count):
                total_input.append(input_list[prev_layer_perc_idx][perc_idx])
            total_input = np.array(total_input)
            output_list[perc_idx] = self.neurons[perc_idx].calc_output(total_input)
    


                

class Perceptron:
    def __init__(self, init_output, weight_count):
        self.output = init_output
        self.weight_count = weight_count
        self.weights = []
        for w_idx in range(0, self.weight_count):
            self.weights.append(0.0)
        self.weights = np.array(self.weight)
    
    def ReLU(self, input):
        return max([0, input])

    def calc_output(self, input_list):
        total = np.sum(input_list)
        self.output = self.ReLU(total)
        return self.output * self.weights
