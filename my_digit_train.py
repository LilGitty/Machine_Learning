from my_neural_net_library import *
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

#===================== Utilities =============================

def one_hot_predict(neural_net, input):
    probabilities = neural_net.predict(input)
    return probabilities.argmax(axis=1)
    
def test_accuracy(neural_net, input_layer, output_layer):
    num_of_inputs =  np.shape(input_layer)[0]
    predicted = one_hot_predict(neural_net, input_layer) #remember that this is a matrix size Nx1
    results = np.array([(predicted[i][0] == output_layer[i]) for i in range(num_of_inputs)])
    return np.sum(results.astype(int)) / num_of_inputs

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

input_layer_size = 28*28
hidden_layer_size = 10
output_layer_size = 10
N = 100 #Number of tests per digit

# ======================= Create A ============================
A_all = np.zeros((10*N, input_layer_size))
b_all = np.zeros((10*N, output_layer_size))
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

neural_net = NeuralNet((input_layer_size, hidden_layer_size, output_layer_size))

def main():
    load_mat = "y" #input("Do you want to load current weights? (y/n)")
    if(load_mat == "y"):
        try:
            neural_net.load_weights("res/iden_digit.npz")
            print("Weights Loaded")
        except:
            print("No Weights Found, Randomly Generating.")

    done = False

    try:
        while not done:
            data_permutation = np.arange(A_all.shape[0])
            np.random.shuffle(data_permutation)

            #Shuffle Training data to get Random Batches:
            input_layer = A_all[data_permutation]
            output_layer = b_all[data_permutation]
            
            print("Begin Training")
            neural_net.train_network(input_layer, output_layer, N ,100, 0.1)   
            print("Training Done")
            
            train_accuracy = test_accuracy(neural_net, A_all, labels_all)
            print("Accuracy: " + str(train_accuracy))
            
            print("Saving Weights")
            neural_net.save_weights("res/iden_digit")
            done = train_accuracy > 0.95 #input("Do you want to continue training? (y/n)") != "y"
        
    except KeyboardInterrupt:
        pass


main()