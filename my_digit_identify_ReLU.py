import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

############
#Memory Setup

input_layer_size = 28*28
hidden_layer_size = 10
output_layer_size = 10

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
N = 10 #Number of tests per digit

# ======================= Create A ============================
A_all = np.zeros((20*N, 28*28))
b_all = np.zeros((20*N, 10))
for i in range(20*N):
    block = int(i / N) % 10
    A_all[i,  : ] = np.reshape(training_images[:, :, training_labels == block][:, :, int(i%N)], (1,28*28))

for i in range(20*N): #prepare b
    digit = int(i / N) % 10
    for j in range(10):
        if j == digit:
            b_all[i][j] = 1
        else:
            b_all[i][j] = 0

print("Data Loaded")
#============================ Load Weights =========================

W1 = np.load("res/W1.mat")
b1 = np.load("res/b1.mat")
W2 = np.load("res/W2.mat")
b2 = np.load("res/b2.mat")

#============================ Test Problematic ===========================

for i in range(10*N):
    test_index = i
    probabilities = predict(A_all[test_index, :])
    probabilities = zip(range(10), probabilities)
    prediction = sorted(probabilities, key = lambda x: -1 * x[1])[0]
    plt.imshow(np.reshape(A_all[test_index, :], (28, 28)), cmap='gray')
    plt.title('problematic digit. prediction: ' + str(prediction[0]) + " confidence:" + str(prediction[1]) + "\n real value: " + str(b_all[test_index]))
    plt.axis('image')
    plt.axis('off')
    plt.show(block=False)
    plt.waitforbuttonpress()
