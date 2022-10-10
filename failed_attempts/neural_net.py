import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = np.random.uniform(-0.5, 0.5, (3,1)) - 1
 
    def sigmoid(self, x):
        return 1 / ( 1 + np.exp(-x) )

    def sigmoid_derivative(self, x):
        return x * ( 1 - x )

    def train(self, training_inputs, training_outputs, training_iterations = 100):
        for iteration in range(training_iterations):
            outputs = self.predict(training_inputs)
            
            error = training_outputs - outputs
            
            adjustments = error * self.sigmoid_derivative(outputs)
 
            self.weights += np.dot(training_inputs.T, adjustments)
 
    def predict(self, input_layer):
        input_layer = input_layer.astype(float)
        return self.sigmoid(np.dot(input_layer, self.weights))
 
training_inputs = np.array([[0,0,1],
                                            [1,1,1],   
                                            [1,0,1],
                                            [0,1,1]])
                                            
training_outputs = np.array([[0,1,1,0]]).T
 
 
net = NeuralNetwork(3)
 
net.train(training_inputs, training_outputs, 200)
 
print(net.predict(training_inputs))
