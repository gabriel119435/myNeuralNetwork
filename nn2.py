import random

import numpy as np

import nn1

'''
all previous comments were removed and only improvements are commented out now:
  1. better weight initialization
  2. cross entropy cost instead of quadratic cost
  3. l2 regularization
  4. early stopping
'''


# to be a cost function one must satisfy two conditions:
#   1. C(a,y) >= 0 always
#   2. as y~a => C(a,y)~0

# examples of C_per_input: https://stats.stackexchange.com/a/154880
# quadratic cost:     C_per_input = ((a-y)^2)/2
# cross entropy cost: C_per_input = -y.ln(a)-(1-y)ln(1-a) = - (y.ln(a) + (1-y)ln(1-a))

# C_total = sum(C_per_input)/n_inputs + reg*(w^2)/(2*n_all)

# this only defines C_per_input!
class CrossEntropyCost:
    @staticmethod
    def cost(a, y):
        # if both a and y are 1 or both zero, np.nan_to_num is used to return 0*ln(0) = 0*(-infinity) = 0
        return np.sum(
            np.nan_to_num(
                -y * np.log(a) - (1 - y) * np.log(1 - a)
            )
        )

    @staticmethod
    def dc_db(z, a, y):
        # a(L) = σ(z(L))
        # dC/db(L) = dC/da(L)   * da(L)/dz(L)     * dz(L)/db(L)
        #          = dC/da(L)   * σ'(z(L))        * 1

        # dC/da(L) = - ( y/a - (1-y)/(1-a) ) = - ( y/a + (y-1)/(1-a) )
        #          =  -y/a + (1-y)/(1-a)
        #          = ( -y(1-a) + (1-y)a ) / a(1-a)
        #
        # dC/db(L) = ( ( -y(1-a)+(1-y)a ) / a(1-a) ) * σ(z(L)) * (1-σ(z(L)))
        #          = ( ( -y(1-a)+(1-y)a ) / a(1-a) ) * a       * (1-a)
        #          = ( -y(1-a) + (1-y)a )
        #          = -y + ya + a -ya
        #          = a - y
        return a - y


class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

        # lowering standard deviation on weights makes neurons ahead less likely to saturate and doing so,
        # more sensitive to changes from input.
        # here, every weight is divided by sqrt(x)
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        # using cross entropy instead of quadratic cost
        self.cost_function = CrossEntropyCost()
        number_biases = sum(b.flatten().size for b in self.biases)
        number_weights = sum(w.flatten().size for w in self.weights)
        print("network created with layout={} biases={} weights={}".format(
            list(self.sizes), number_biases, number_weights)
        )

    def train_test_network(
            self,
            train_data,
            iterations,
            size,
            rate,
            test_data,
            validate_data,
            reg,
            early_stopping
    ):
        print("iterations={} miniBatch.size={} learningRate={} reg={} earlyStopping={}"
              .format(iterations, size, rate, reg, early_stopping))
        print("miniBatch.count={}".format(
            int(len(train_data) / size)
        ))

        bad_iterations = 0
        old_acc = None

        for i in range(iterations):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k + size] for k in range(0, len(train_data), size)]

            for mini_batch in mini_batches:
                # now we also pass reg and train data size
                self._apply_mini_batch(mini_batch, rate, reg, len(train_data))

            new_acc = self._test_network(validate_data)
            print("{}: {}".format(i, new_acc))

            if old_acc is not None:
                if new_acc > old_acc:
                    bad_iterations = 0
                else:
                    bad_iterations += 1

            if bad_iterations == early_stopping:
                print("early stopping")
                break

            old_acc = new_acc

        acc = self._test_network(test_data)
        print(acc)
        print("network tested\n")
        return acc

    def _feed_network(self, a):
        for b, w in zip(self.biases, self.weights):
            a = nn1.sigmoid(np.dot(w, a) + b)
        return a

    def _test_network(self, data):
        results = [(np.argmax(self._feed_network(x)), np.argmax(y)) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results) / len(data)

    def _apply_mini_batch(self, mini_batch, rate, reg, train_data_len):
        b_total = [np.zeros(b.shape) for b in self.biases]
        w_total = [np.zeros(w.shape) for w in self.weights]

        for _input, _output in mini_batch:
            b_diff, w_diff = self.backward_prop(_input, _output)
            b_total = [total_diff + diff for total_diff, diff in zip(b_total, b_diff)]
            w_total = [total_diff + diff for total_diff, diff in zip(w_total, w_diff)]

        # old formula was
        # b = old - rate * dC_total_db
        # b = old - rate * (sum(dC_per_input_db) / n_inputs)
        # w = old - rate * dC_total_dw
        # w = old - rate * (sum(dC_per_input_dw) / n_inputs)

        # using L2 regularization:
        # C_total = sum(C_per_input)/n_inputs + reg*(w^2)/(2*n_all)

        # new formula is
        # b = old - rate * dC_total_db
        #   = old - rate * (sum(dC_per_input_db) / n_inputs)

        # w = old - rate * dC_total_dw
        #   = old - rate * (sum(dC_per_input_dw)/n_inputs + reg*w/n_all)

        #   renaming:
        #
        #   sum(dC_per_input_dw) = total
        #   n_inputs = mini
        #   n_all = n
        #
        # w = old - rate * (total/mini + reg*old/n)
        # w = old - rate*total/mini - rate*reg*old/n
        # w = old(1 - rate*reg/n) - total*(rate/mini)
        # w = old(1 - r1) - total*r2

        mini = len(mini_batch)
        r1 = rate * reg / train_data_len
        r2 = rate / mini

        self.weights = [
            old * (1 - r1) - total * r2
            for old, total in zip(self.weights, w_total)
        ]
        self.biases = [
            old - total * r2
            for old, total in zip(self.biases, b_total)
        ]

    def backward_prop(self, _input, _output):
        b_diff = [np.zeros(b.shape) for b in self.biases]
        w_diff = [np.zeros(w.shape) for w in self.weights]

        current_activation = _input
        a_array = [current_activation]
        z_array = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, current_activation) + b
            z_array.append(z)
            current_activation = nn1.sigmoid(z)
            a_array.append(current_activation)

        diff = self.cost_function.dc_db(z_array[-1], a_array[-1], _output)
        b_diff[-1] = diff
        w_diff[-1] = np.dot(diff, a_array[-2].transpose())
        for i in range(2, self.num_layers):
            diff = np.dot(self.weights[-i + 1].transpose(), diff) * nn1.derivative_sigmoid(z_array[-i])
            b_diff[-i] = diff
            w_diff[-i] = np.dot(diff, a_array[-i - 1].transpose())
        return b_diff, w_diff

    def total_cost(self, data, reg):
        # besides not being used, useful to keep
        # C_total = sum(C_per_input)/n_inputs + reg*(w^2)/(2*n_all)
        #         = sum(C_per_input)/n        + reg*(w^2)/(2*n)
        # C_total = firstCost                 + RegCost
        n = len(data)
        first_cost, reg_cost = 0.0, 0.0
        for x, y in data:
            a = self._feed_network(x)
            first_cost += self.cost_function.cost(a, y) / n

        w_2 = sum(np.linalg.norm(w) ** 2 for w in self.weights)
        reg_cost = reg * w_2 / (2 * n)
        return first_cost + reg_cost
