import numpy as np

def ReLu_function(x):
    return np.where(x > 0, x, 0)

def ReLu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0

class Perceptron:
    def __init__(self, n_units, input_units):
        self.n_units = n_units
        self.input_units = input_units
        
        self.size = n_units * input_units
        self.weight_matrix = np.random.rand(input_units, n_units)
        self.bias = np.zeros(n_units)

        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None
    
    def forward_step(self, input_data):
        self.input_data
        self.layer_preactivation = np.dot(input_data, self.weight_matrix) + self.bias
        self.layer_activation = ReLu_function(self.layer_preactivation)
        return self.layer_activation
        
    def backward_step(self, dL_dactivation, learning_rate):
        dactivation_dpreactivation = ReLu_derivative(self.layer_preactivation)
        dL_dpreactivation = dL_dactivation * dactivation_dpreactivation

        dL_dweights = np.dot(self.input_data.T, dL_dpreactivation)
        dL_dbias = np.sum(dL_dpreactivation, axis=0)

        dL_dinput = np.dot(dL_dpreactivation, self.weight_matrix.T)
        
        self.weight_matrix -= learning_rate * dL_dweights
        self.bias -= learning_rate * dL_dbias

        return dL_dinput