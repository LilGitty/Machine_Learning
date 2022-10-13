import numpy as np

############
#Memory Setup

class NeuralNet:
    def __init__(self, layers_size):
        input_layer_size, hidden_layer_size, output_layer_size = layers_size

        self.W1 = np.random.rand(hidden_layer_size, input_layer_size)
        self.W2 = np.random.rand(output_layer_size, hidden_layer_size)

        self.b1 = np.random.rand(hidden_layer_size, 1)
        self.b2 = np.random.rand(output_layer_size, 1)

    def forward_propagate(self, input_vector):
        #accepts columns as input
        self.hidden_vector = relu(self.W1 @ input_vector + self.b1)
        self.output_vector = softmax(self.W2 @ self.hidden_vector + self.b2)
        
        #returns output as columns
        return self.output_vector
        
    def predict(self, input_vector):
        #accepts rows as input (easier to format)
        input_vector = np.asmatrix(input_vector, dtype=float).T
        #returns output as rows
        return self.forward_propagate(input_vector).T
    
    def backwards_propagate(self, training_input, expected_output, learn_rate = 0.1):
        self.forward_propagate(training_input)
        
        num_of_inputs = np.shape(training_input)[1]
        
        delta_output = self.output_vector - expected_output #delta according to cross-entropy and softmax gradient
        
        delta_W2 = delta_output @ self.hidden_vector.T / num_of_inputs
        delta_b2 = np.sum(delta_output) / num_of_inputs
        
        delta_hidden_layer = np.multiply(self.W2.T @ delta_output,  relu_deriv(self.hidden_vector))
        
        delta_W1 = delta_hidden_layer @ training_input.T / num_of_inputs
        delta_b1 = np.sum(delta_hidden_layer) / num_of_inputs
        
        self.update_params(delta_W2, delta_b2, delta_W1, delta_b1, learn_rate)
    
    def update_params(self, delta_W2, delta_b2, delta_W1, delta_b1, learn_rate):
        self.W2 -= normalize(delta_W2) * learn_rate
        self.b2 -= normalize(delta_b2) * learn_rate
        self.W1 -= normalize(delta_W1) * learn_rate
        self.b1 -= normalize(delta_b1) * learn_rate
    
    def train_network(self, training_input, training_output, batch_size = 100, num_of_iterations = 10000, learn_rate = 0.1):
        training_input = np.asmatrix(training_input, dtype=float)
        training_output = np.asmatrix(training_output, dtype=float)

        for i in range(num_of_iterations): #Also possible to take random batches so it doesn't "Overfit"
            i = int(i % np.shape(training_input)[0] / batch_size)
            batch_input = training_input[i * batch_size : (i+1) * batch_size : ].T
            batch_output = training_output[i * batch_size : (i+1) * batch_size : ].T
            self.backwards_propagate(batch_input , batch_output, learn_rate)
    

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
