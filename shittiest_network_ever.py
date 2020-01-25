import random

import numpy as np


class Network:

    def __init__(self, sizes):
        # [784, 16, 16, 10]
        self.num_layers = len(sizes)
        self.sizes = sizes

        # [(16, 1), (16, 1), (10, 1)]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # [(784, 16), (16, 16), (16, 10)]
        # ex. looking at self.weights[0], we have matrix w (784, 16)
        #     element w(jk) links neuron k on 1st layer 784 and neuron j on 2nd layer 16
        # notice how x,y swap places being used as y,x
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        number_biases = sum(b.flatten().size for b in self.biases)
        number_weights = sum(w.flatten().size for w in self.weights)
        print("layout={}, biases={}, weights={}".format(list(self.sizes), number_biases, number_weights))

    def feed_network(self, a):
        # receives (784,1), outputs (10,1)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train_test_network(self, train_data, iterations, size, rate, test_data, test_every_iteration):
        """
        trains the neural network using mini-batch stochastic gradient descent and then test it against test_data
        https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31
        """
        print("iterations={}, mini_batch_size={}, learning_rate={}".format(iterations, size, rate))
        new, old = None, None
        for i in range(iterations):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k + size] for k in range(0, len(train_data), size)]

            for mini_batch in mini_batches:
                self.apply_mini_batch(mini_batch, rate)

            if test_every_iteration:
                new = self.test_network(test_data)
                print_test_result(new, old, i)
                old = new

        if not test_every_iteration:
            new = self.test_network(test_data)
            print_test_result(new)

    def apply_mini_batch(self, mini_batch, rate):
        """
        upgrades weights and biases according to a single iteration of gradient descent
        proportional to the learning rate
        """
        b_total = [np.zeros(b.shape) for b in self.biases]
        w_total = [np.zeros(w.shape) for w in self.weights]

        # calculates the gradients
        for input_data, desired_output in mini_batch:
            b_diff, w_diff = self.backward_prop(input_data, desired_output)
            b_total = [total + diff for total, diff in zip(b_total, b_diff)]
            w_total = [total + diff for total, diff in zip(w_total, w_diff)]

        mini_batch_len = len(mini_batch)

        # applies the average gradients proportionally to learning_rate
        self.weights = [old - rate * (total / mini_batch_len)
                        for old, total in zip(self.weights, w_total)]
        self.biases = [old - rate * (total / mini_batch_len)
                       for old, total in zip(self.biases, b_total)]

    def backward_prop(self, input_data, desired_output):
        # return biases_diff and weights_diff as the gradient of cost function to each bias and weight
        b_diff = [np.zeros(b.shape) for b in self.biases]
        w_diff = [np.zeros(w.shape) for w in self.weights]

        # feed network storing a and z per layer
        # a' = σ(wa+b) = σ(z)
        # a' = σ(z)
        current_activation = input_data
        # list to store activations per layer
        a_array = [current_activation]
        # list to store z vector per layer
        z_array = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, current_activation) + b
            z_array.append(z)
            current_activation = sigmoid(z)
            a_array.append(current_activation)

        '''
        C = ((a(L)-y)^2)/2
        a(L) = σ(z(L))
        z(L) = w(L).a(L-1) + b(L)
        a(L) = σ(w(L).a(L-1) + b(L))
        a(L-1) = σ(w(L-1).a(L-2) + b(L-1))
        
        partials:
        dC/da(L)      = a(L)-y
        da(L)/dz(L)   = σ'(z(L))
        dz(L)/db(L)   = 1
        dz(L)/dw(L)   = a(L-1)
        dz(L)/da(L-1) = w(L)
        
        chain rule:
        dC/db(L)   = dC/da(L) * da(L)/dz(L) * dz(L)/db(L)   = (a(L)-y) * σ'(z(L)) * 1      = dC/db(L)
        
        dC/dw(L)   = dC/da(L) * da(L)/dz(L) * dz(L)/dw(L)   = (a(L)-y) * σ'(z(L)) * a(L-1) = dC/db(L) * a(L-1)
        
        dC/da(L-1) = dC/da(L) * da(L)/dz(L) * dz(L)/da(L-1) = (a(L)-y) * σ'(z(L)) * w(L)   = dC/db(L) * w(L)
        
        dC/db(L-1) = dC/da(L-1)      * da(L-1)/dz(L-1) * dz(L-1)/db(L-1) 
                   = dC/db(L) * w(L) * σ'(z(L-1))      * 1
                   = dC/db(L) * w(L) * σ'(z(L-1))                                          = dC/db(L-1)
        
        dC/dw(L-1) = dC/da(L-1)      * da(L-1)/dz(L-1) * dz(L-1)/dw(L-1) 
                   = dC/db(L) * w(L) * σ'(z(L-1))      * a(L-2)
                   = dC/db(L-1)                        * a(L-2)
        
        operation            |            actual operation               |    diff value after operation
        set                  |         diff = dC/db(L)                   |    dC/db(L)      
        use                  |         dC/dw(L) = diff * a(L-1)          |    dC/db(L)
        
        set                  |         diff = diff * w(L) * σ'(z(L-1))   |    dC/db(L-1)
        use                  |         dC/dw(L-1) = diff * a(L-2)        |    dC/db(L-1)
        
        set                  |         diff = diff * w(L-1) * σ'(z(L-2)) |    dC/db(L-2)
        use                  |         dC/dw(L-2) = diff * a(L-3)        |    dC/db(L-2)
        
        ...                  |                  ...                      |       ...
        
        '''
        # first iteration finding dC/dB(L) and dC/dW(L)
        # dC/db(L) = ( a(L) - y) * σ'(z(L)), but in matrix notation
        diff = (a_array[-1] - desired_output) * derivative_sigmoid(z_array[-1])
        b_diff[-1] = diff
        # dC/dw(L) = dC/db(L) * a(L-1), but in matrix notation
        w_diff[-1] = np.dot(diff, a_array[-2].transpose())

        # from second-last (-2) to the second (num_layers exclusive) layer
        for i in range(2, self.num_layers):
            diff = np.dot(self.weights[-i + 1].transpose(), diff) * derivative_sigmoid(z_array[-i])
            b_diff[-i] = diff
            w_diff[-i] = np.dot(diff, a_array[-i - 1].transpose())

        return b_diff, w_diff

    def test_network(self, data):
        """
        data is the test data as (inputs, desiredOutputs)
        returns how many inputs were correctly categorized from 1(100%) to 0(0%)
        note the correct result is whichever neuron in the final layer with the highest activation
        """
        results = [(np.argmax(self.feed_network(x)), np.argmax(y)) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results) / len(data)


def sigmoid(z):
    # https://en.wikipedia.org/wiki/Sigmoid_function
    return 1.0 / (1.0 + np.exp(-z))


def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def print_test_result(new, old=None, i=None):
    string = ""
    formatted_new = "{:.2%}".format(new)
    if i is not None:
        string += "iteration={:<4}".format(i)
    string += "test={:<6}".format(formatted_new)
    if old:
        diff = "{:+.2%}".format(new - old)
        string += "{:>9}".format(diff)
    print(string)
