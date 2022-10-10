import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layer_weights = [np.random.uniform(-0.5, 0.5, (3,2)) - 1, np.random.uniform(-0.5, 0.5, (2,1)) - 1]
        #self.biases = [np.random.uniform(-0.5, 0.5, (3,1)) - 1, np.random.uniform(-0.5, 0.5, (3,1)) - 1]
        
    def activation_func(self, x): #sigmoid activation
        return 1 / ( 1 + np.exp(-x) )

    def activation_func_derivative(self, x):
        return x * ( 1 - x )

#########
    def train(self, training_inputs, training_outputs, training_iterations = 10, learn_rate = 0.01):
        for iteration in range(training_iterations):
            for input in training_inputs:
                #forward propagation
                layers_outputs = self.forward_propagate(input)
                
                #error and adjustments:
                
                outputs_error = training_outputs - layers_outputs[-1]
                
                layer_index = len(self.layer_weights) - 1
                while(layer_index > 0):
                    
                    if(layer_index == len(self.layer_weights) - 1):
                        result_delta = outputs_error
                    else:
                        result_delta = np.transpose(self.layer_weights[layer_index]) @ outputs_error * activation_func_derivative(layers_outputs[layer_index - 1])
                     
                    #adjust:
                    self.layer_weights[layer_index] += -learn_rate * np.transpose(result_delta) @ layers_outputs[layer_index]
                    #self.biases += ...
                    
                    layer_index -= 1
 
 ########
    def forward_propagate(self, input_layer):
        outputs = input_layer.astype(float)
        ret = [outputs]
        for layer in self.layer_weights:
            #calculate layer TODO: add bias
            outputs = np.dot(outputs, layer)
            ret += [outputs]
        
        return ret #all layer values
    
 
    def predict(self, input_layer):
        outputs = input_layer.astype(float)
        
        for layer in self.layer_weights:
            #calculate layer TODO: add bias
            outputs = np.dot(outputs, layer)
        
        return self.sigmoid(outputs)
 
training_inputs = np.array([[0,0,1],
                                            [1,1,1],   
                                            [1,0,1],
                                            [0,1,1]])
                                            
training_outputs = np.array([[0,1,1,0]]).T
 
 
net = NeuralNetwork(3)
 
net.train(training_inputs, training_outputs, 200)
 
print(net.predict(training_inputs))
