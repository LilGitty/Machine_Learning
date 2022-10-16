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

def label_b():
    global b_all, labels_all
    
    b_all = np.zeros((N, output_layer_size))
    labels_all = np.zeros((N, 1))
    
    index = 0
    for label in range(10):
        dir_path = os.path.join(license_dir, str(label))
        for training_image_path in os.listdir(dir_path):
            if(train_digit == label):
                b_all[index][1] =  1 #the rest is automatically 0
            else:
                b_all[index][0] = 1
            
            labels_all[index] = int(train_digit == label)
            index += 1
    print("Labels Loaded for digit " + str(train_digit))

# ======================= Parameters ==========================

input_layer_size = 50*25
hidden_layer_size = 10
output_layer_size = 2
N=3872
train_digit = 0
image_size = (25, 50) #width, height
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
        # if(train_digit == label):
            # b_all[index][1] =  1 #the rest is automatically 0
        # else:
            # b_all[index][0] = 1
        
        # labels_all[index] = int(train_digit == label)
        index += 1
print("Data Loaded")


# ======================= Training ============================

neural_net = NeuralNet((input_layer_size, hidden_layer_size, output_layer_size))

def main():
    load_mat = "y" #input("Do you want to load current weights? (y/n)")
    if(load_mat == "y"):
        try:
            neural_net.load_weights("res/license_digit_" + str(train_digit) + ".npz")
            print("Weights Loaded")
        except:
            print("No Weights Found, Randomly Generating.")

    done = False

    try:
        while not done:
        
            print("Begin Training")
            neural_net.train_network(A_all, b_all, N ,1000, 0.1)   
            print("Training Done")
            
            train_accuracy = test_accuracy(neural_net, A_all, labels_all)
            print("Accuracy: " + str(train_accuracy))
            
            print("Saving Weights")
            neural_net.save_weights("res/license_digit_" + str(train_digit))
            done = train_accuracy > 0.95 #input("Do you want to continue training? (y/n)") != "y"
        
    except KeyboardInterrupt:
        pass


#main()
for i in range(2, 10):
    train_digit = i
    label_b()
    main()