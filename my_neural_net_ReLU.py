import numpy as np

############
#Memory Setup

input_layer_size = 2
hidden_layer_size = 2 #5
output_layer_size = 1 #10

W1 = np.random.rand(hidden_layer_size, input_layer_size)
W2 = np.random.rand(output_layer_size, hidden_layer_size)

b1 = np.random.rand(hidden_layer_size, 1)
b2 = np.random.rand(output_layer_size, 1)

############

def relu_deriv(s, EPS = 0.01):
    return (np.multiply((s > 0), 1 - EPS) + EPS)

def relu(s, EPS = 0.01):
    return (np.multiply((s > 0), s * (1 - EPS)) + EPS*s)

def normalize(A, EPS = 0.1):
    m = np.abs(A).max()
    if(m < EPS):
        return A
    return A / m

def forward_propagate(input_vector):
    global hidden_vector, output_vector
    #accepts columns as input
    hidden_vector = relu(W1 @ input_vector + b1)
    output_vector = W2 @ hidden_vector + b2
    
    #returns output as columns
    return output_vector
    
def predict(input_vector):
    input_vector = np.asmatrix(input_vector, dtype=float).T
    return forward_propagate(input_vector)
    
def backwards_propagate(training_input, expected_output, learn_rate = 0.1):
    global hidden_vector, output_vector, W2, W1, b2, b1
    
    forward_propagate(training_input)
    
    num_of_inputs = np.shape(training_input)[1]
    
    delta_output = output_vector - expected_output
    delta_W2 = delta_output @ np.transpose(hidden_vector) / num_of_inputs
    delta_b2 = np.sum(delta_output) / num_of_inputs
    
    delta_hidden_layer = np.multiply(W2.T @ delta_output,  relu_deriv(hidden_vector))
    
    delta_W1 = delta_hidden_layer @ training_input.T
    delta_b1 = np.sum(delta_hidden_layer) / num_of_inputs
    
    update_params(delta_W2, delta_b2, delta_W1, delta_b1, learn_rate)


def update_params(delta_W2, delta_b2, delta_W1, delta_b1, learn_rate):
    global W1, W2, b1, b2
    
    W2 -= normalize(delta_W2) * learn_rate
    b2 -= normalize(delta_b2) * learn_rate
    W1 -= normalize(delta_W1) * learn_rate
    b1 -= normalize(delta_b1) * learn_rate


def train_network(training_input, training_output, batch_size = 100, num_of_iterations = 10000, learn_rate = 0.1):
    training_input = np.asmatrix(training_input, dtype=float)
    training_output = np.asmatrix(training_output, dtype=float)

    for i in range(num_of_iterations): #Also possible to take random batches so it doesn't "Overfit"
        i = int(i % np.shape(training_input)[1] / batch_size)
        batch_input = training_input[i * batch_size : (i+1) * batch_size : ].T
        batch_output = training_output[i * batch_size : (i+1) * batch_size : ].T
        backwards_propagate(batch_input , batch_output, learn_rate)

#====================== Data ==============================

input_layer = np.array([[2,9], [3,1], [4,1]])
output_layer = np.array([[92], [28], [69]])

output_layer = output_layer/100 # maximum test score is 100

train_network(input_layer, output_layer, 1, 10000, 0.5)

print("Weights and Bias:\nW1:", W1, "\nW2:", W2, "\nb1:", b1, "\nb2:", b2)
print("Example tests:", predict([[2,9], [3,1], [4,1]]))
