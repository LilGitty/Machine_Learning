import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

############
#Memory Setup

input_layer_size = 28*28
hidden_layer_size = 10 #5
output_layer_size = 10 #10

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

#====================== Data ==============================
dataset = loadmat('mnist.mat')
training = dataset['training'][0][0]
training_count = training[0][0][0]
training_height = training[1][0][0]
training_width = training[2][0][0]
training_images = training[3]
training_labels = training[4]
training_labels = np.squeeze(training_labels)
print("Data Loaded")
# ======================= Parameters ==========================
N = 1 #Number of tests per digit

# ======================= Create A ============================
A_all = np.zeros((10*N, 28*28))
b_all = np.zeros((10*N, 10))
for i in range(10*N):
    block = int(i / N) % 10
    #if(i % N == 0):
        #print("Preparing New Block: " + str(block))

    A_all[i,  : ] = np.reshape(training_images[:, :, training_labels == block][:, :, int(i%N)], (1,28*28))

# ======================= Training ============================

for i in range(10*N): #prepare b
    digit = int(i / N) % 10
    for j in range(10):
        if j == digit:
            b_all[i][j] = 1
        else:
            b_all[i][j] = 0

A_train = A_all
b_train = b_all
    
input_layer = 1000*A_train
output_layer = b_train

print("Begin Training")
train_network(input_layer, output_layer, 10000, 10)

print("Training Done")

print("Weights and Bias:\nW1:", W1, "\nW2:", W2, "\nBias:", bias_vector)

for i in range(2):
    probabilities = predict(A_train[i, :])
    print(list(probabilities))
    probabilities = zip(range(10), probabilities)

    prediction = sorted(probabilities, key = lambda x: -1 * x[1])[0]

    print(prediction)
    print(b_train[i])
    plt.imshow(np.reshape(A_train[i, :], (28, 28)), cmap='gray')
    plt.title('problematic digit. prediction: ' + str(prediction[0]) + " confidence:" + str(prediction[1]) + "\n real value: " + str(b_train[i]))
    plt.axis('image')
    plt.axis('off')
    plt.show(block=False)
    plt.waitforbuttonpress()
