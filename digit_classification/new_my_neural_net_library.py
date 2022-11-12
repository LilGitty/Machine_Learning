import numpy as np

# This is my implementation of a Single-Hidden-Layer Neural Net using Leaky ReLU and SoftMax
# The Activation Functions are easily changable

class NeuralNet:
    def __init__(self, layers_size):
        input_layer_size, output_layer_size = layers_size[0], layers_size[-1]
        self.W = [np.random.rand(layers_size[i+1], layers_size[i]) for i in range(len(layers_size) - 1)]
        self.b = [np.random.rand(layers_size[i+1], 1) for i in range(len(layers_size) - 1)]

    def forward_propagate(self, input_vector):
        #accepts columns as input
        self.hidden_vector = [input_vector]
        for i in range(len(self.W) - 1):
            self.hidden_vector.append(relu(self.W[i] @ self.hidden_vector[i] + self.b[i]))
        self.output_vector = softmax(self.W[-1] @ self.hidden_vector[-1] + self.b[-1])
        
        #returns output as columns
        return self.output_vector
        
    def predict(self, input_vector):
        #accepts rows as input (easier to format)
        input_vector = np.asmatrix(input_vector, dtype=float).T
        #returns output as rows
        return self.forward_propagate(input_vector).T
    
    def backwards_propagate(self, training_input, expected_output, learn_rate = 0.1):
        self.forward_propagate(training_input)
        
        delta_W = [0]*len(self.W)
        delta_b = [0]*len(self.b)
        
        num_of_inputs = np.shape(training_input)[1]
        
        delta_output = self.output_vector - expected_output #delta according to cross-entropy and softmax gradient
        
        delta_W[-1] = delta_output @ self.hidden_vector[-1].T / num_of_inputs #change to hidden_last layer
        delta_b[-1] = np.sum(delta_output) / num_of_inputs
        
        delta_hidden_last_layer = delta_output
        
        for i in range(len(self.W) - 2,-1, -1):
            delta_hidden_layer = np.multiply(self.W[i+1].T @ delta_hidden_last_layer,  relu_deriv(self.hidden_vector[i+1]))
            
            delta_W[i] = delta_hidden_layer @ self.hidden_vector[i].T / num_of_inputs
            delta_b[i] = np.sum(delta_hidden_layer) / num_of_inputs
        
            delta_hidden_last_layer = delta_hidden_layer
        
        self.update_params(delta_W, delta_b, learn_rate)
    
    def update_params(self, delta_W, delta_b, learn_rate):
        for i in range(len(self.W)):
            self.W[i] -= normalize(delta_W[i]) * learn_rate
            self.b[i] -= normalize(delta_b[i]) * learn_rate
    
    def train_network(self, training_input, training_output, num_of_iterations = 1000, learn_rate = 0.1, batch_size = -1):
        if batch_size == -1:
            batch_size = training_input.shape[0]
        
        training_input = np.asmatrix(training_input, dtype=float)
        training_output = np.asmatrix(training_output, dtype=float)

        if batch_size == training_input.shape[0]:
            for iter in range(num_of_iterations):
                self.backwards_propagate(training_input.T , training_output.T, learn_rate)
                
        else:
            for iter in range(num_of_iterations):
                for i in range(np.shape(training_input)[0] // batch_size):
                    batch_input = training_input[i * batch_size : (i+1) * batch_size : ].T
                    batch_output = training_output[i * batch_size : (i+1) * batch_size : ].T
                    self.backwards_propagate(training_input , training_output, learn_rate)
                    
                    #Also possible to take random batches so it doesn't "Overfit"
                    # data_permutation = np.arange(training_input.shape[0])
                    # np.random.shuffle(data_permutation)

                    #Shuffle Training data to get Random Batches:
                    #input_layer = training_input[data_permutation]
                    #output_layer = training_output[data_permutation]
            
    def load_weights(self, filename):
        weights = np.load(filename)
        self.W = weights["W"]
        self.b = weights["b"]

    def save_weights(self, filename):
        np.savez_compressed(filename, W=self.W, b=self.b) #change if this takes too much space
    

#In the future, make these part of the class and let the user decide which to use

def relu_deriv(s, EPS = 0.01):
    return (np.multiply((s > 0), 1 - EPS) + EPS)

def relu(s, EPS = 0.01):
    return (np.multiply((s > 0), s * (1 - EPS)) + EPS*s)

def softmax(v):
    v = v - v.max(0) #shift by largest value
    new_v = np.exp(v)
    return  new_v / new_v.sum(0)

def normalize(A, EPS = 0.1):
    m = np.abs(A).max()
    if(m < EPS):
        return A
    return A / m
