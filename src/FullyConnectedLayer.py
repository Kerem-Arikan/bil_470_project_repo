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
        output = np.zeros((self.perc_count, self.weight_count))
        # todo: write a forward propogation algorithm here

class Perceptron:
    def __init__(self, init_output, weight_count):
        self.output = init_output
        self.weight_count = weight_count
        self.weight = []
        for w_idx in range(0, self.weight_count):
            self.weight.append(1)
        self.weight = np.array(self.weight)
    
    def ReLU(self, input):
        return max([0, input])

    def calc_output(self, input_list):
        total = np.sum(input_list)
        self.output = self.ReLU(total)
        return self.output * self.weight
