import numpy as np
#06/10/2022
############
#Memory Setup

input_layer_size = 2
hidden_layer_size = 1 #5
output_layer_size = 1 #10

W1 = np.random.rand(hidden_layer_size, input_layer_size)
W2 = np.random.rand(output_layer_size, hidden_layer_size)

bias_vector = np.random.rand(hidden_layer_size, 1)

############

def activation_function(s, deriv=False):
     if (deriv == True):
         return np.multiply(s, 1 - s)
     return 1/(1 + np.exp(-s))

def forward_propagate(input_vector):
    global hidden_vector, output_vector
    #accepts columns as input
    hidden_vector = activation_function(W1 @ input_vector + bias_vector)
    output_vector = activation_function(W2 @ hidden_vector)
    
    #returns output as columns
    return output_vector
    
def predict(input_vector):
    input_vector = np.asmatrix(input_vector, dtype=float).T
    return forward_propagate(input_vector)
    
def backwards_propagate(training_input, expected_output, learn_rate = 0.1):
    global hidden_vector, output_vector, W2, W1, bias_vector
 
    forward_propagate(training_input)
    
    delta_output = np.multiply(output_vector - expected_output, activation_function(output_vector, True))
    W2_grad = delta_output @ np.transpose(hidden_vector)
    W2 -= W2_grad * learn_rate

    bias_grad = W2.T @ delta_output
    bias_grad = np.multiply(bias_grad, activation_function(hidden_vector, True))
    test_ones_vector = np.ones((np.shape(training_input)[1], 1)) #Sum all gradients for all tests at once
    bias_vector -= (bias_grad @ test_ones_vector) * learn_rate
    
    W1_grad = bias_grad @ training_input.T
    W1 -= W1_grad * learn_rate

def train_network(training_input, training_output, num_of_iterations = 10000, learn_rate = 0.1):
    training_input = np.asmatrix(training_input, dtype=float).T
    training_output = np.asmatrix(training_output, dtype=float).T
    
    for i in range(num_of_iterations): #Also possible to take random batches so it doesn't "Overfit"
        backwards_propagate(training_input, training_output, learn_rate)

########################################

#DATA SET AND TRAINING:


input_layer = np.array([[2, 9], [1, 5], [3, 6]])
output_layer = np.array([[92], [86], [89]]) #REMEMBER TO NORMALIZE RESULTS FROM 0 TO 1

output_layer = output_layer/100 # maximum test score is 100

train_network(input_layer, output_layer, 10000, 1)

print("Weights and Bias:\nW1:", W1, "\nW2:", W2, "\nBias:", bias_vector)
print("Example tests:", predict([[2, 9], [1, 5], [3, 6]]))

#Note: This is a probability Classifier (Because of the sigmoid activation function), So feel free to change activation function!