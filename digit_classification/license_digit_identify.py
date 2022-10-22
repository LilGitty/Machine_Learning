from my_neural_net_library import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#===================== Utilities =============================

def predict_probabilities(digit_nets, input):
    probabilities = np.asarray([digit_nets[i].predict(input)[:, 1].T for i in range(0, 10)]) #np.asarray([digit_nets[i].predict(input)[1] for i in range(0,10)])
    probabilities = np.squeeze(probabilities, axis=1).T
    return probabilities
    
def one_hot_predict(digit_nets, input):
    probabilities = predict_probabilities(digit_nets, input)
    return probabilities.argmax(axis=1)
    
def test_accuracy(digit_nets, input_layer, output_layer):
    num_of_inputs =  np.shape(input_layer)[0]
    predicted = one_hot_predict(digit_nets, input_layer) #remember that this is a matrix size Nx1
    results = np.array([(predicted[i] == output_layer[i]) for i in range(num_of_inputs)])
    return np.sum(results.astype(int)) / num_of_inputs

# ======================= Parameters ==========================

input_layer_size = 50*25
hidden_layer_size = 10
output_layer_size = 2
N=3862
image_size = (25, 50)
license_dir = "license_database/license_plate_numbers"

# ======================= Create A ============================
A_all = np.zeros((N, input_layer_size))
labels_all = np.zeros((N, 1))

index = 0
for label in range(10):
    dir_path = os.path.join(license_dir, str(label))
    for training_image_path in os.listdir(dir_path):
        training_image = Image.open(os.path.join(dir_path,  training_image_path))
        training_image = training_image.convert("L") #Black And White
        A_all[index] = np.reshape(np.asarray(training_image.resize(image_size)), (input_layer_size)) / 255
        
        labels_all[index] = label
        index += 1
        
print("Data Loaded")
#============================ Load Weights =========================

digit_nets = []

for digit in range(0, 10):
    neural_net = NeuralNet((input_layer_size, hidden_layer_size, output_layer_size))
    neural_net.load_weights("res/license_digit_" + str(digit) + ".npz")
    digit_nets += [neural_net]
    print("Weights Loaded For " + str(digit))

#============================ Test Problematic ===========================

print("Accuracy: " + str(test_accuracy(digit_nets, A_all, labels_all)))

output_difference = one_hot_predict(digit_nets, A_all) - np.reshape(labels_all, (1, N))[0]
problematic_indexes = np.where(output_difference != 0)[0]

#Note: There is a bit of redundence and recalculation left to improve.

np.set_printoptions(precision=4)

try:
    for test_index in problematic_indexes:
        probabilities = predict_probabilities(digit_nets, A_all[test_index, :])
        plt.imshow(np.reshape(A_all[test_index, :], (50, 25)), cmap='gray')
        plt.title('prediction: ' + str(np.argmax(probabilities)) + " probabilities:" + str(probabilities) + "\n real value: " + str(labels_all[test_index]))
        plt.axis('image')
        plt.axis('off')
        plt.show(block=False)
        plt.waitforbuttonpress()
    print("Testing Done")

except KeyboardInterrupt:
    pass
