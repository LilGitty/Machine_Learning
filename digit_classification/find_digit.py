from my_neural_net_library import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

digit_width, digit_height = 25, 50

#is_digit_net = NeuralNet((28*28, 5, 2)) #Is this even a digit?
classify_digit_net = NeuralNet((digit_width*digit_height, 10, 10)) #Which digit from 0 - 10

#is_digit_net.load_weights("res/find_digit.npz")
classify_digit_net.load_weights("res/license_digit.npz")

#Algorithm: Sliding Window, check in different sizes, rescale to 25x50

image = Image.open("f.png")
image = image.convert("L") #Black And White
#image = np.asarray(image) / 255

#digits_found = []

test_window = np.asarray(image.resize((digit_width, digit_height))) / 255
probabilities = classify_digit_net.predict(np.reshape(test_window, (digit_width*digit_height))).T
predicted_digit = np.argmax(probabilities)
plt.imshow(test_window, cmap='gray')
plt.title('Looking for Digit. prediction: ' + str(predicted_digit) + " confidence:" + str(probabilities[predicted_digit]))
plt.axis('image')
plt.axis('off')
plt.show(block=False)
plt.waitforbuttonpress()
plt.clf()


# for k in range(min(image.shape) // min(digit_width, digit_height) - 5, 0, -1):
    # for x in range(0, image.shape[0] - k*digit_width, 10):
        # for y in range(0, image.shape[1] - k*digit_height, 10):
            # test_window = image[x : x + k*digit_width, y : y + k*digit_height]
            # test_window = np.asarray(Image.fromarray(test_window).resize((digit_width, digit_height))) #Classic Python Moment
            
            # #prediction
            # probabilities = classify_digit_net.predict(np.reshape(test_window, (digit_width*digit_height))).T
            # predicted_digit = np.argmax(probabilities)
            # plt.imshow(test_window, cmap='gray')
            # plt.title('Looking for Digit. prediction: ' + str(predicted_digit) + " confidence:" + str(probabilities[predicted_digit]))
            # plt.axis('image')
            # plt.axis('off')
            # plt.show(block=False)
            # plt.waitforbuttonpress()
            # plt.clf()

