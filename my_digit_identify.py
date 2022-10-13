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

def load_weights(neural_net):
    weights = np.load("res/weights.npz")
    neural_net.W1 = weights["W1"]
    neural_net.b1 = weights["b1"]
    neural_net.W2 = weights["W2"]
    neural_net.b2 = weights["b2"]
    print("Weights Loaded")

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
data_permutation = np.arange(N)
np.random.shuffle(data_permutation)

random_training = training_images[:, : ,data_permutation]
labels_all = training_labels[data_permutation]

A_all = np.zeros((N, 28*28))

for i in range(N): #reformat A
    A_all[i] = random_training[:, :, i].reshape(1, 28*28)

print("Data Loaded")
#============================ Load Weights =========================

neural_net = NeuralNet((input_layer_size, hidden_layer_size, output_layer_size))

load_weights(neural_net)

#============================ Test Problematic ===========================
print("Accuracy: " + str(test_accuracy(neural_net, A_all, labels_all)))

output_difference = one_hot_predict(neural_net, A_all) - np.reshape(labels_all.T, (N,1))

problematic_indexes = np.where(output_difference != 0)[0]

#Note: There is a bit of redundence and recalculation left to improve.

try:
    for test_index in problematic_indexes:
        probabilities = neural_net.predict(A_all[test_index, :])
        plt.imshow(np.reshape(A_all[test_index, :], (28, 28)), cmap='gray')
        plt.title('problematic digit. prediction: ' + str(np.argmax(probabilities)) + " confidence:" + str(np.max(probabilities)) + "\n real value: " + str(labels_all[test_index]))
        plt.axis('image')
        plt.axis('off')
        plt.show(block=False)
        plt.waitforbuttonpress()
    print("Testing Done")

except KeyboardInterrupt:
    pass
    
# test_image = Image.open('test.png')
# test_image = np.reshape(test_image, (28,28)) / 255
# test_image = 1 - test_image
# probabilities = predict(np.reshape(test_image, (1,28*28)))
# probabilities = zip(range(10), probabilities)
# prediction = sorted(probabilities, key = lambda x: -1 * x[1])[0]
# plt.imshow(test_image, cmap='gray')
# plt.title('Test Image. prediction: ' + str(prediction[0]) + " confidence:" + str(prediction[1]) + "\n real value: ???")
# plt.axis('image')
# plt.axis('off')
# plt.show(block=False)
# plt.waitforbuttonpress()
# print("Testing Done")
