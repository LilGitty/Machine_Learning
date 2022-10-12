import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

############
#Memory Setup

input_layer_size = 28 * 28
hidden_layer_size = 10
output_layer_size = 10

W1 = np.random.rand(hidden_layer_size, input_layer_size)
W2 = np.random.rand(output_layer_size, hidden_layer_size)

b1 = np.random.rand(hidden_layer_size, 1)
b2 = np.random.rand(output_layer_size, 1)

############

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

def forward_propagate(input_vector):
    global hidden_vector, output_vector
    #accepts columns as input
    hidden_vector = relu(W1 @ input_vector + b1)
    output_vector = softmax(W2 @ hidden_vector + b2)
    
    #returns output as columns
    return output_vector
    
def predict(input_vector):
    #accepts rows as input (easier to format)
    input_vector = np.asmatrix(input_vector, dtype=float).T
    #returns output as rows
    return forward_propagate(input_vector).T
    
def backwards_propagate(training_input, expected_output, learn_rate = 0.1):
    global hidden_vector, output_vector, W2, W1, b2, b1
    
    forward_propagate(training_input)
    
    num_of_inputs = np.shape(training_input)[1]
    
    delta_output = output_vector - expected_output #delta according to cross-entropy and softmax gradient
    
    delta_W2 = delta_output @ hidden_vector.T
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
        i = int(i % np.shape(training_input)[0] / batch_size)
        batch_input = training_input[i * batch_size : (i+1) * batch_size : ].T
        batch_output = training_output[i * batch_size : (i+1) * batch_size : ].T
        backwards_propagate(batch_input , batch_output, learn_rate)

#===================== Utilities =============================

def one_hot_predict(input):
    probabilities = predict(input)
    return probabilities.argmax(axis=1)
    
def test_accuracy(input_layer, output_layer):
    num_of_inputs =  np.shape(input_layer)[0]
    predicted = one_hot_predict(input_layer) #remember that this is a matrix size Nx1
    results = np.array([(predicted[i][0] == output_layer[i]) for i in range(num_of_inputs)])
    return np.sum(results.astype(int)) / num_of_inputs

def load_weights():
    global W1, W2, b1, b2
    W1 = np.load("res/W1.mat")
    b1 = np.load("res/b1.mat")
    W2 = np.load("res/W2.mat")
    b2 = np.load("res/b2.mat")
    print("Weights Loaded")

def dump_weights():
    print("Saving Weights")
    W1.dump("res/W1.mat")
    b1.dump("res/b1.mat")
    W2.dump("res/W2.mat")
    b2.dump("res/b2.mat")

#====================== Data ==============================
dataset = loadmat('mnist.mat')
training = dataset['training'][0][0]
training_count = training[0][0][0]
training_height = training[1][0][0]
training_width = training[2][0][0]
training_images = training[3]
training_labels = training[4]
training_labels = np.squeeze(training_labels)
# ======================= Parameters ==========================
N = 1500 #Number of tests per digit

# ======================= Create A ============================
A_all = np.zeros((10*N, 28*28))
b_all = np.zeros((10*N, 10))
labels_all = np.zeros((10*N, 1))

for i in range(10*N):
    block = int(i / N) % 10
    A_all[i,  : ] = np.reshape(training_images[:, :, training_labels == block][:, :, int(i%N)], (1,28*28))

for i in range(10*N): #prepare b
    digit = int(i / N) % 10
    labels_all[i] = digit
    for j in range(10):
        if j == digit:
            b_all[i][j] = 1
        else:
            b_all[i][j] = 0

print("Data Loaded")

# ======================= Training ============================

load_mat = "y"#input("Do you want to load current weights? (y/n)")
if(load_mat == "y"):
    load_weights()

done = False

try:
    while not done:
        data_permutation = np.arange(A_all.shape[0])
        np.random.shuffle(data_permutation)

        #Shuffle Training data to get Random Batches:
        input_layer = A_all[data_permutation]
        output_layer = b_all[data_permutation]
        
        print("Begin Training")
        train_network(input_layer, output_layer, N ,10000, 0.1)   
        print("Training Done")
        
        train_accuracy = test_accuracy(A_all, labels_all)
        print("Accuracy: " + str(train_accuracy))
        
        dump_weights()
        done = train_accuracy > 0.95 #input("Do you want to continue training? (y/n)") != "y"
    
except KeyboardInterrupt:
    pass
