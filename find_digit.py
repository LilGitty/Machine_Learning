from my_neural_net_library import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

is_digit_net = NeuralNet((28*28, 5, 2)) #Is this even a digit?
what_digit_net = NeuralNet((28*28, 10, 2)) #Which digit from 0 - 10

is_digit_net.load_weights("res/find_digit.npz")
what_digit_net.load_weights("res/iden_digit.npz")

#Algorithm: Sliding Window, check in different sizes, rescale to 28x28

image = Image.open("test.png")
image = image.convert("L") #Black And White
image = np.asarray(image) / 255

digits_found = []

for window_size in range(min(image.shape), 80, -10):
    for x in range(0, image.shape[0] - window_size, 10):
        for y in range(0, image.shape[1] - window_size, 10):
            test_window = image[x : x + window_size, y : y + window_size]
            test_window = np.asarray(Image.fromarray(test_window).resize((28, 28))) #Classic Python Moment
            
            test_window = 1 - test_window
            
            probabilities = is_digit_net.predict(np.reshape(test_window, (28*28))).T
            predicted_digit = np.argmax(probabilities)
            plt.imshow(test_window, cmap='gray')
            plt.title('Looking for Digit. prediction: ' + str(predicted_digit) + " confidence:" + str(probabilities[predicted_digit]))
            plt.axis('image')
            plt.axis('off')
            plt.show(block=False)
            
            if predicted_digit == 1: #Loop until you find 0
                digits_found.append([x, y, window_size])
                probabilities = what_digit_net.predict(np.reshape(test_window, (28*28))).T
                predicted_digit = np.argmax(probabilities)
                plt.title('Digit Found. prediction: ' + str(predicted_digit) + " confidence:" + str(probabilities[predicted_digit]))
                plt.waitforbuttonpress()
            
            plt.pause(0.0001)
            plt.clf()

