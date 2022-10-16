#not my code, but an iteresting comparison as they used different orders

import numpy as np
# X = (hours sleeping, hours studying), y = test score of the student
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) #maximum of X array
y = y/100 # maximum test score is 100

class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # weight matrix from hidden to output layer
        
    def feedForward(self, input_layer):
        #forward propogation through the network
        self.z = np.dot(input_layer, self.W1) #input layer is a row, multiply it by weights
        self.z2 = self.sigmoid(self.z) #activation function on hidden layer
        self.z3 = np.dot(self.z2, self.W2) #output weights from hidden layer
        output = self.sigmoid(self.z3)
        return output
        
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
    
    def backward(self, input_layer, expected_output, output):
        #backward propogate through the network
        self.output_error = expected_output - output # row vector of error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        
        self.hidden_layer_error = self.output_delta @ self.W2.T #z2 error: how much our hidden layer weights contribute to output error
        self.hidden_layer_error = self.hidden_layer_error * self.sigmoid(self.z2, deriv=True) #applying derivative of sigmoid to z2 error
        
        self.W1 += input_layer.T.dot(self.hidden_layer_error) # adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights
        
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)
        
NN = NeuralNetwork()

for i in range(1000): #trains the NN 1000 times
    if (i % 100 == 0):
        print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
    NN.train(X, y)
        
print("Input: " + str(X))
print("Actual Output: " + str(y))
print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
print("\n")
print("Predicted Output: " + str(NN.feedForward(X)))