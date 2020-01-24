import random
import numpy as np


class Network:

    # sizes is an array with the the quantity of neurons
    def __init__(self, sizes):

        # [784, 16, 16, 10]
        self.num_layers = len(sizes)
        self.sizes = sizes

        # [(16, 1), (16, 1), (10, 1)]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # [(784, 16), (16, 16), (16, 10)]
        # ex. looking at self.weights[0], we have matrix w (784, 16)
        #     element w(jk) links neuron k on 1st layer 784 and neuron j on 2nd layer 16
        #
        # a' = σ(wa+b)
        # a' is the neuron layer after a
        #
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

    def train_test_network(self, train_data, iterations, mini_batch_size, learning_rate, test_data):
        # train the neural network using mini-batch stochastic gradient descent
        # https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31
        print("iterations={}, mini_batch_size={}, learning_rate={}".format(iterations, mini_batch_size, learning_rate))
        old_result = None
        for i in range(iterations):
            random.shuffle(train_data)
            # range(0,10,3): [[0,1,2],[3,4,5],[6,7,8],[9]]
            mini_batches = [train_data[k:k + mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]

            for mini_batch in mini_batches:
                self.apply_mini_batch(mini_batch, learning_rate)

            # test network efficiency against test
            new_result = self.test_network(test_data)

            # just formatting strings
            new_result_string = "{:.2%}".format(new_result)
            result_string = "iteration={: <4} test={: <6}".format(i, new_result_string)
            if old_result:
                diff = "{:+.2%}".format(new_result - old_result)
                result_string += "{: >9}".format(diff)
            print(result_string)
            old_result = new_result

        print("")

    def apply_mini_batch(self, mini_batch, learning_rate):
        # upgrades weights and biases according to a single iteration of gradient descent
        b_gradient_sum = [np.zeros(b.shape) for b in self.biases]
        w_gradient_sum = [np.zeros(w.shape) for w in self.weights]

        # calculates the gradients
        for input_data, desired_output in mini_batch:
            b_gradient_diff, w_gradient_diff = self.apply_single_example(input_data, desired_output)
            b_gradient_sum = [total_diff + diff for total_diff, diff in zip(b_gradient_sum, b_gradient_diff)]
            w_gradient_sum = [total_diff + diff for total_diff, diff in zip(w_gradient_sum, w_gradient_diff)]

        mini_batch_len = len(mini_batch)

        # applies the average gradients proportionally to learning_rate
        self.weights = [old - learning_rate * (sum_diff / mini_batch_len)
                        for old, sum_diff in zip(self.weights, w_gradient_sum)]
        self.biases = [old - learning_rate * (sum_diff / mini_batch_len)
                       for old, sum_diff in zip(self.biases, b_gradient_sum)]

    def apply_single_example(self, input_data, desired_output):
        # return biases_diff and weights_diff as the gradient of cost function to each bias and weight
        biases_diff = [np.zeros(b.shape) for b in self.biases]
        weights_diff = [np.zeros(w.shape) for w in self.weights]

        # feed network storing a and z per layer
        # a' = σ(wa+b) = σ(z) => a' = σ(z)
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

        # backward pass
        # first iteration manually with finding dC/dB(L) and dC/dW(L)
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
        
        dC/db(L)   = dC/da(L) * da(L)/dz(L) * dz(L)/db(L)   = (a(L)-y) * σ'(z(L)) * 1      = dC/db(L)
        
        dC/dw(L)   = dC/da(L) * da(L)/dz(L) * dz(L)/dw(L)   = (a(L)-y) * σ'(z(L)) * a(L-1) = dC/db(L) * a(L-1)
        
        dC/da(L-1) = dC/da(L) * da(L)/dz(L) * dz(L)/da(L-1) = (a(L)-y) * σ'(z(L)) * w(L)   = dC/db(L) * w(L)
        
        dC/db(L-1) = dC/da(L-1)      * da(L-1)/dz(L-1) * dz(L-1)/db(L-1) 
                   = dC/db(L) * w(L) * σ'(z(L-1))      * 1
                   = dC/db(L) * w(L) * σ'(z(L-1)) = dC/db(L-1)
        
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
        # dC/db(L) = ( a(L) - y) * σ'(z(L)), but in matrix notation
        diff = (a_array[-1] - desired_output) * derivative_sigmoid(z_array[-1])
        biases_diff[-1] = diff
        # dC/dw(L) = a(L-1) * dC/d(L), but in matrix notation
        weights_diff[-1] = np.dot(diff, a_array[-2].transpose())

        # from second-last (-2) to the second(num_layers exclusive) layer
        for i in range(2, self.num_layers):
            diff = np.dot(self.weights[-i + 1].transpose(), diff) * derivative_sigmoid(z_array[-i])
            biases_diff[-i] = diff
            weights_diff[-i] = np.dot(diff, a_array[-i - 1].transpose())

        return biases_diff, weights_diff

    def test_network(self, data):
        # data is the test data as (inputs, desiredOutputs)
        # returns how many inputs were correctly categorized from 1(100%) to 0(0%)
        # note the correct result is whichever neuron in the final layer with the highest activation
        results = [(np.argmax(self.feed_network(x)), np.argmax(y)) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results) / len(data)


# https://en.wikipedia.org/wiki/Sigmoid_function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
