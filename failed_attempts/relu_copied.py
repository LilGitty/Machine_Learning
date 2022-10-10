import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def init_params():
    global input_size, hidden_size, output_size

    W1 = np.random.rand(hidden_size, input_size) - 0.5
    b1 = np.random.rand(hidden_size, 1) - 0.5
    W2 = np.random.rand(output_size, hidden_size) - 0.5
    b2 = np.random.rand(output_size, 1) - 0.5
    return W1, b1, W2, b2

def normalize(Z, EPS = 0.1):
    if(np.abs(Z).max() < EPS):
        return Z
    return Z / np.abs(Z).max()

def ReLU(Z):
    return np.maximum(Z, 0)
    
def softmax(Z):
    A = np.exp(Z)
    #print(A)
    return A / np.sum(A)
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = np.shape(X)[1] #num of inputs
    dZ2 = A2 - Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    print(W2.shape, dZ2.shape, Z1.shape)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    dW1, db1, dW1, db2 = normalize(dW1), normalize(db1), normalize(dW2), normalize(db2)
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
    
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return A2

####

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
    
input_layer = A_train.T
output_layer = b_train.T


input_size, hidden_size, output_size = 28 * 28, 5, 10

W1, b1, W2, b2 = gradient_descent(input_layer, output_layer, 0.10, 70)

#print("Weights and Bias:\nW1:", W1, "\nW2:", W2, "\nb1:", b1, "\nb2:", b2)