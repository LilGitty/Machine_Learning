Welcome to Machine Learning Project.

Here I try to implement various Neural Net and Machine Learning Algorithms.

Cheat Sheet for the math:

Backpropagation for one hidden-layered network:

W1 - Matrix of weights from input to hidden (size: hidden X in),
b1 - Column Vector of hidden layer bias (size: hidden X 1),
W2 - etc. hidden to output (size: out X hidden),
b2 - etc. output layer bias (size: out X 1).

Cost = L2, Activation = Sigmoid / ReLU

delta_output = (output - expected_output) * activation_deriv(output) (or the mean for multiple inputs)
b2 = delta_output  (or the mean for multiple columns of input)
W2 = delta_output @ hidden_layer.T

delta_hidden_layer = multiply_by_value(W2.T @ delta_output, activation_deriv(hidden))

delta_b1 = delta_hidden_layer
delta_W1 = delta_hidden_layer @ input.T #W1 is always like the bias, times the transpose of input

###########

Cost = Cross-Entropy, Activation = Sigmoid / Relu, Output Activation = Softmax

Same as above, but
delta_output = output - expected_output
(Weird probability magic makes the gradient really similar)