from my_neural_net_library import *
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

#===================== Utilities =============================

def one_hot_predict(neural_net, input):
    probabilities = neural_net.predict(input)
    return probabilities.argmax(axis=1)
    
def problematic_indexes(neural_net, input_layer, output_layer):
    num_of_inputs =  np.shape(input_layer)[0]
    predicted = one_hot_predict(neural_net, input_layer) #remember that this is a matrix size Nx1
    results = np.array([(predicted[i] == np.argmax(output_layer[i])) for i in range(num_of_inputs)])
    return np.where(results != True)[0]
    
def test_accuracy(neural_net, input_layer, output_layer):
    num_of_inputs =  np.shape(input_layer)[0]
    return 1 - problematic_indexes(neural_net, input_layer, output_layer).shape[0] / num_of_inputs

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
hidden_layer_size = 5
output_layer_size = 2
N = 1000 #Number of tests per digit, 10*N for non-digit

# ======================= Create A ============================
A_all = np.zeros((20*N, input_layer_size))
b_all = np.zeros((20*N, output_layer_size))

for i in range(10*N):
    block = int(i / N) % 10
    A_all[i,  : ] = np.reshape(training_images[:, :, training_labels == block][:, :, int(i%N)], (1,28*28))
    b_all[i] = [0, 1] #first = non-digit, second = digit

for i in range(10*N, 20*N):
    A_all[i,  : ] = np.random.rand(1, input_layer_size)
    b_all[i] = [1, 0]


print("Data Loaded")

# ======================= Training ============================

neural_net = NeuralNet((input_layer_size, hidden_layer_size, output_layer_size))

def main():
    load_mat = "y" #input("Do you want to load current weights? (y/n)")
    if(load_mat == "y"):
        try:
            neural_net.load_weights("res/find_digit.npz")
            print("Weights Loaded")
        except:
            print("No Weights Found, Randomly Generating.")

    done = False

    try:
        while not done:
            # data_permutation = np.arange(A_all.shape[0])
            # np.random.shuffle(data_permutation)

            # #Shuffle Training data to get Random Batches:
            # input_layer = A_all[data_permutation]
            # output_layer = b_all[data_permutation]
            
            print("Begin Training")
            neural_net.train_network(A_all, b_all, 100, 0.1)   
            print("Training Done")
            
            train_accuracy = test_accuracy(neural_net, A_all, b_all)
            print("Accuracy: " + str(train_accuracy))
            
            print("Saving Weights")
            neural_net.save_weights("res/find_digit")
            done = train_accuracy > 0.95 #input("Do you want to continue training? (y/n)") != "y"
        
    except KeyboardInterrupt:
        pass

def test():
    neural_net.load_weights("res/find_digit.npz")
    
    prob_indexes = problematic_indexes(neural_net, A_all, b_all)
    print("Accuracy: " + str(1 - (prob_indexes.shape[0] / A_all.shape[0])))

    #Note: There is a bit of redundence and recalculation left to improve.
    for test_index in prob_indexes:
        probabilities = neural_net.predict(A_all[test_index, :])
        plt.imshow(np.reshape(A_all[test_index, :], (28, 28)), cmap='gray')
        plt.title('problematic digit. prediction: ' + str(np.argmax(probabilities)) + " confidence:" + str(np.max(probabilities)) + "\n real value: " + str(b_all[test_index]))
        plt.axis('image')
        plt.axis('off')
        plt.show(block=False)
        plt.waitforbuttonpress()
    print("Testing Done")

# def test():
    # try:
        # for test_index in range(A_all.shape[0]):
            # probabilities = neural_net.predict(A_all[test_index, :])
            # plt.imshow(np.reshape(A_all[test_index, :], (28, 28)), cmap='gray')
            # plt.title('problematic digit. prediction: ' + str(np.argmax(probabilities)) + " confidence:" + str(np.max(probabilities)) + "\n real value: " + str(np.argmax(b_all[test_index])))
            # plt.axis('image')
            # plt.axis('off')
            # plt.show(block=False)
            # plt.waitforbuttonpress()
        # print("Testing Done")

    # except KeyboardInterrupt:
        # pass


main()
test()