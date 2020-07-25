import numpy as np
from scipy.special import softmax
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
        self.output_layer = np.zeros((self.target_count))#oğuz
        for perc_idx in range(0, self.perc_count):
            self.output_layer += self.neurons[perc_idx].calc_output(input_list[perc_idx])
            print(self.output_layer)
        self.final_output = self.softmax(self.output_layer)
        return self.final_output
    
    def softmax(self, input_list):
        print("softmax_input_list - ",input_list)
        e_x = np.exp(input_list - np.max(input_list))
        return  e_x / e_x.sum()
        
    
    def backward_prop(self, train_targets):
        loss_gradient = np.zeros(shape=(self.perc_count))
        (train_target_count) = train_targets.shape[0] ######

        assert train_target_count == self.target_count

        for perc_idx in range(0, self.perc_count):
            loss_gradient_sum = 0
            for weight_idx in range(0, self.target_count):

                loss_gradient_sum += (-train_targets[weight_idx] + self.final_output[weight_idx]) * self.neurons[perc_idx].weights[weight_idx] #*(1/self.perc_count)#Oğuz 
                weight_gradient = (-train_targets[weight_idx] + self.final_output[weight_idx]) * self.neurons[perc_idx].input_
                self.neurons[perc_idx].weights[weight_idx] = self.neurons[perc_idx].weights[weight_idx] - self.alpha * weight_gradient #+ self.neurons[perc_idx].momentum#Oğuz
                #self.neurons[perc_idx].momentum = self.alpha * weight_gradient#Oğuz
                print("loss_gradient_sum  - ",loss_gradient_sum)
                print("weight_gradient  - ",weight_gradient)
                print("self.neurons[perc_idx].weights[weight_idx]  - ",self.neurons[perc_idx].weights[weight_idx])
                
        #loss_gradient[perc_idx] += (loss_gradient_sum * self.neurons[perc_idx].weights[weight_idx])   
        return loss_gradient


    def ReLU(self, i_array):
        return np.maximum(0, i_array)
                

class Perceptron:
    def __init__(self, weight_count):
        self.output = 0
        self.weight_count = weight_count
        self.weights =np.ones((weight_count))
        #self.momentum = 0#Oğuz
    def ReLU(self, i_array):
        return np.maximum(0, i_array)

    def calc_output(self, input_):
        self.input_ = input_
        self.output = input_*self.weights#Oğuz
        return self.output