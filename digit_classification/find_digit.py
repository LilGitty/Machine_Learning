from my_neural_net_library import *
from my_filter import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

input_layer_size = 50*25
hidden_layer_size = 10
output_layer_size = 2
digit_width, digit_height = 25, 50

#is_digit_net = NeuralNet((28*28, 5, 2)) #Is this even a digit?
#classify_digit_net = NeuralNet((digit_width*digit_height, 10, 10)) #Which digit from 0 - 10

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

# ======================= Load Weights ==========================

license_dir = "license_database/license_plate_numbers"

digit_nets = []

for digit in range(0, 10):
    neural_net = NeuralNet((input_layer_size, hidden_layer_size, output_layer_size))
    neural_net.load_weights("res/license_digit_" + str(digit) + ".npz")
    digit_nets += [neural_net]
    print("Weights Loaded For " + str(digit))
        
print("Data Loaded")
#============================ =========================

    
#Algorithm: Sliding Window, check in different sizes, rescale to 25x50

image = Image.open("0ld_test.png")
image = image.convert("L") #Black And White
#image = image.resize((image.size[0]*2, image.size[1]*2))

digits_found = []

real_digit_width, real_digit_height = 54, 30
image = np.asarray(image) / 255
for k in range(min(image.shape) // min(real_digit_width, real_digit_height), 0, -1):
    for x in range(0, image.shape[0] - k*real_digit_width, 10):
        for y in range(0, image.shape[1] - k*real_digit_height, 10):
            test_window = image[x : x + k*real_digit_width, y : y + k*real_digit_height]
            test_window = np.asarray(Image.fromarray(test_window).resize((digit_width, digit_height))) #Classic Python Moment
            #test_window = filter_image(test_window)
            #===== prediction
            probabilities = predict_probabilities(digit_nets, np.reshape(test_window, (digit_width*digit_height)))
            predicted_digit = np.argmax(probabilities)
            possible_digits = np.array(np.where(probabilities > 0.9)[1])
            
            # if len(possible_digits) > 0: #When we want to analyze results later
                # digits_found += [(x, y, k, predicted_digit)]
            
            #===== visualization
            plt.imshow(test_window, cmap='gray')
            plt.title('Looking for digit. prediction: ' + str(np.argmax(probabilities)) + " probabilities:" + str(probabilities) + " possible digits: " + np.array_str(possible_digits))
            plt.axis('image')
            plt.axis('off')
            plt.show(block=False)
            if len(possible_digits) > 0:
                plt.waitforbuttonpress()
            plt.clf()


#TODO: filter spots where it finds the same digit again and again - mark those as correct
#===== visualization
# test_window = np.asarray(image.resize((digit_width, digit_height))) / 255 #Classic Python Moment
# test_window = np.reshape(test_window, (digit_width*digit_height))
# test_window = filter_image(test_window)
# probabilities = predict_probabilities(digit_nets, test_window)
# plt.imshow(np.resize(test_window, (digit_height, digit_width)), cmap='gray')
# plt.title('Looking for digit. prediction: ' + str(np.argmax(probabilities)) + " probabilities:" + str(probabilities))
# plt.axis('image')
# plt.axis('off')
# plt.show(block=False)
# plt.waitforbuttonpress()
# plt.clf()

##old
# test_window = np.asarray(image.resize((digit_width, digit_height))) / 255
# probabilities = predict_probabilities(digit_nets, np.reshape(test_window, (digit_width*digit_height)))
# predicted_digit = np.argmax(probabilities)
# possible_digits = np.array(np.where(probabilities > 0.9)[1])
# plt.imshow(test_window, cmap='gray')
# plt.title('prediction: ' + str(np.argmax(probabilities)) + " probabilities:" + str(probabilities) + " possible digits: " + np.array_str(possible_digits))
# #plt.title('Looking for Digit. prediction: ' + str(predicted_digit) + " confidence:" + str(probabilities[predicted_digit]))
# plt.axis('image')
# plt.axis('off')
# plt.show(block=False)
# plt.waitforbuttonpress()
# plt.clf()