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
    input_vector = np.asmatrix(input_vector, dtype=float).T
    return forward_propagate(input_vector)

def one_hot_predict(input):
    probabilities = predict(input).T
    return probabilities.argmax(axis=1)
    
def test_accuracy(input_layer, output_layer):
    num_of_inputs =  np.shape(input_layer)[0]
    predicted = one_hot_predict(input_layer) #remember that this is a matrix size Nx1
    results = np.array([(predicted[i][0] == output_layer[i]) for i in range(num_of_inputs)])
    return np.sum(results.astype(int)) / num_of_inputs

def load_weights():
    global W1, W2, b1, b2
    weights = np.load("res/weights.npz")
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
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
N = training_count #Number of tests per digit

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

load_weights()

#============================ Test Problematic ===========================
print("Accuracy: " + str(test_accuracy(A_all, labels_all)))

output_difference = one_hot_predict(A_all) - np.reshape(labels_all.T, (N,1))

problematic_indexes = np.where(output_difference != 0)[0]

#Note: There is a bit of redundence and recalculation left to improve.

try:
    for i in problematic_indexes:
        test_index = i
        probabilities = predict(A_all[test_index, :])
        probabilities = zip(range(10), probabilities)
        prediction = sorted(probabilities, key = lambda x: -1 * x[1])[0]
        plt.imshow(np.reshape(A_all[test_index, :], (28, 28)), cmap='gray')
        plt.title('problematic digit. prediction: ' + str(prediction[0]) + " confidence:" + str(prediction[1]) + "\n real value: " + str(labels_all[test_index]))
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
