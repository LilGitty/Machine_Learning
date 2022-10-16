from my_neural_net_library import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#===================== Utilities =============================

def one_hot_predict(neural_net, input):
    probabilities = neural_net.predict(input)
    return probabilities.argmax(axis=1)
    
def test_accuracy(neural_net, input_layer, output_layer):
    num_of_inputs =  np.shape(input_layer)[0]
    predicted = one_hot_predict(neural_net, input_layer) #remember that this is a matrix size Nx1
    results = np.array([(predicted[i][0] == output_layer[i]) for i in range(num_of_inputs)])
    return np.sum(results.astype(int)) / num_of_inputs

# ======================= Parameters ==========================

input_layer_size = 50*25
hidden_layer_size = 10
output_layer_size = 2
N=3872
train_digit = 2
image_size = (25, 50)
license_dir = "license_database/license_plate_numbers"

# ======================= Create A ============================
A_all = np.zeros((N, input_layer_size))
b_all = np.zeros((N, output_layer_size))
labels_all = np.zeros((N, 1))

index = 0
for label in range(10):
    dir_path = os.path.join(license_dir, str(label))
    for training_image_path in os.listdir(dir_path):
        training_image = Image.open(os.path.join(dir_path,  training_image_path))
        training_image = training_image.convert("L") #Black And White
        A_all[index] = np.reshape(np.asarray(training_image.resize(image_size)), (input_layer_size)) / 255
        if(train_digit == label):
            b_all[index][1] =  1 #the rest is automatically 0
        else:
            b_all[index][0] = 1
        
        labels_all[index] = int(train_digit == label)
        index += 1
        
print("Data Loaded")
#============================ Load Weights =========================

neural_net = NeuralNet((input_layer_size, hidden_layer_size, output_layer_size))

neural_net.load_weights("res/license_digit_" + str(train_digit) + ".npz")
print("Weights Loaded")

#============================ Test Problematic ===========================
print("Accuracy: " + str(test_accuracy(neural_net, A_all, labels_all)))

output_difference = one_hot_predict(neural_net, A_all) - np.reshape(labels_all.T, (N,1))

problematic_indexes = np.where(output_difference != 0)[0]

#Note: There is a bit of redundence and recalculation left to improve.

try:
    for test_index in problematic_indexes:
        probabilities = neural_net.predict(A_all[test_index, :])
        plt.imshow(np.reshape(A_all[test_index, :], (50, 25)), cmap='gray')
        plt.title('problematic digit. prediction: ' + str(np.argmax(probabilities)) + " confidence:" + str(np.max(probabilities)) + "\n real value: " + str(b_all[test_index]))
        plt.axis('image')
        plt.axis('off')
        plt.show(block=False)
        plt.waitforbuttonpress()
    print("Testing Done")

except KeyboardInterrupt:
    pass
