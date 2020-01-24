import random
import numpy as np

'''
all previous comments were removed and only improvements are commented out now. The improvements are:
  1. better weight initialization
  2. cross entropy cost instead of quadratic cost
  3. L2 regularization
  4. early stopping
'''


# to be a cost function one must satisfy two conditions:
#   1. C(a,y) >= 0 always
#   2. as y~a => C(a,y)~0

# examples of C_per_input
# https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
# quadratic cost:     C = ((a-y)^2)/2
# cross entropy cost: C = -y.ln(a)-(1-y)ln(1-a) = - (y.ln(a) + (1-y)ln(1-a))

# C_total = sum(C_per_input)/n_inputs + reg*(w^2)/2n_all
# this only defines C_per_input!
class CrossEntropyCost:
    @staticmethod
    def cost(a, y):
        # if both a and y are 1, np.nan_to_num is used to return 0*ln(0) = 0*(-infinity) = 0
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def dc_db(z, a, y):
        # a(L) = σ(z(L))
        # dC/db(L) = dC/da(L)   * da(L)/dz(L)     * dz(L)/db(L)
        #          = dC/da(L)   * σ'(z(L))        * 1

        # dC/da(L) = - ( y/a - (1-y)/(1-a) ) = - ( y/a + (y-1)/(1-a) )
        #          =  -y/a + (1-y)/(1-a)
        #          = ( -y(1-a) + (1-y)a ) / a(1-a)
        #
        # dC/db(L) = (( -y(1-a) + (1-y)a ) / a(1-a)) * σ(z(L))*(1-σ(z(L)))
        #          = (( -y(1-a) + (1-y)a ) / a(1-a)) * a*(1-a)
        #          =  ( -y(1-a) + (1-y)a ) = -y + ya + a -ya = a - y
        return a - y


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]

        # lowering standard deviation on weights makes neurons ahead less likely to saturate and doing so,
        # more sensitive to changes from back propagation.
        # here, every weight is divided by sqrt(x)
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        # using cross entropy instead of quadratic cost
        self.cost_function = CrossEntropyCost()

        number_biases = sum(b.flatten().size for b in self.biases)
        number_weights = sum(w.flatten().size for w in self.weights)
        print("layout={}, biases={}, weights={}".format(list(self.sizes), number_biases, number_weights))

    def feed_network(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train_test_network(self, train_data, validate_data, test_data,
                           iterations, mini_batch_size, learning_rate, reg, early_stopping):
        # if early_stopping = 3 and the network doesn't improve validate data acc during 3 iterations, it stops
        print("iterations={}, mini_batch_size={}, learning_rate={}, regularization={}, early_stopping={}"
              .format(iterations, mini_batch_size, learning_rate, reg, early_stopping))

        new_validate_result, old_validate_result = None, None
        iterations_without_improvement = 0

        for i in range(iterations):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k + mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]

            for mini_batch in mini_batches:
                # now we also pass regularization_factor and train_data_size
                self.apply_mini_batch(mini_batch, learning_rate, reg, len(train_data))

            train_result = self.get_accuracy_cost(train_data, reg)
            validate_result = self.get_accuracy_cost(validate_data, reg)

            # just formatting strings
            print("iteration={: <3} {{{}}} {{{}}}".format(
                i,
                build_string("train", train_result),
                build_string("valid", validate_result, old_validate_result)
            ))

            # early stopping:
            new_validate_result = validate_result
            if old_validate_result:
                if new_validate_result[0] > old_validate_result[0]:
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            if iterations_without_improvement == early_stopping:
                print("early-stopping, {} iterations without improvement!".format(early_stopping))
                break
            old_validate_result = new_validate_result

        # finally is used against test_data
        test_result = self.get_accuracy_cost(test_data, reg)
        print("{{{}}}".format(build_string("test", test_result)))

    def apply_mini_batch(self, mini_batch, rate, reg, entire_train_set_size):
        b_gradient_sum = [np.zeros(b.shape) for b in self.biases]
        w_gradient_sum = [np.zeros(w.shape) for w in self.weights]

        for input_data, desired_output in mini_batch:
            b_gradient_diff, w_gradient_diff = self.apply_single_example(input_data, desired_output)
            b_gradient_sum = [total_diff + diff for total_diff, diff in zip(b_gradient_sum, b_gradient_diff)]
            w_gradient_sum = [total_diff + diff for total_diff, diff in zip(w_gradient_sum, w_gradient_diff)]

        # old formula was
        # b = old - rate * dC_total_db
        # b = old - rate * (sum(dC_per_input_db) / n_inputs)
        # w = old - rate * dC_total_dw
        # w = old - rate * (sum(dC_per_input_dw) / n_inputs)

        # using L2 regularization:
        # C_total = sum(C_per_input)/n_inputs + reg*(w^2)/2n_all
        # new formula is

        # b = old - rate * dC_total_db
        #   = old - rate * (sum(dC_per_input_db) / n_inputs)

        # w = old - rate * dC_total_dw
        #   = old - rate * (sum(dC_per_input_dw)/n_inputs + reg*w/n_all)

        #   renaming:
        #
        #   old = w
        #   sum(dC_per_input_dw) = dc_dw
        #   n_inputs = mini
        #   n_all = n
        #

        # w = w - rate * (dc_dw/mini + w*reg/n)
        # w = w - rate*dc_dw/mini - rate*w*reg/n
        # w = w   * (1-rate*reg/n)) - rate*dc_dw/mini
        # w = w   * (1-r)           - rate*dc_dw/mini
        # w = old * (1-r)           - rate*(dc_dw/mini)

        r = rate * (reg / entire_train_set_size)
        mini = len(mini_batch)
        self.weights = [old * (1 - r) - rate * (sum_diff / mini)
                        for old, sum_diff in zip(self.weights, w_gradient_sum)]

        # b = old - sum(d(C_per_input)_db) * rate / n_inputs
        self.biases = [old - rate * (sum_diff / mini)
                       for old, sum_diff in zip(self.biases, b_gradient_sum)]

    def apply_single_example(self, input_data, desired_output):
        biases_diff = [np.zeros(b.shape) for b in self.biases]
        weights_diff = [np.zeros(w.shape) for w in self.weights]

        current_activation = input_data
        a_array = [current_activation]
        z_array = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, current_activation) + b
            z_array.append(z)
            current_activation = sigmoid(z)
            a_array.append(current_activation)

        diff = self.cost_function.dc_db(z_array[-1], a_array[-1], desired_output)
        biases_diff[-1] = diff
        weights_diff[-1] = np.dot(diff, a_array[-2].transpose())
        for i in range(2, self.num_layers):
            diff = np.dot(self.weights[-i + 1].transpose(), diff) * derivative_sigmoid(z_array[-i])
            biases_diff[-i] = diff
            weights_diff[-i] = np.dot(diff, a_array[-i - 1].transpose())
        return biases_diff, weights_diff

    def test_network(self, data):
        # returns percentage between 0 and 1
        results = [(np.argmax(self.feed_network(x)), np.argmax(y)) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results) / len(data)

    def total_cost(self, data, reg):
        cost = 0.0

        # C_total =  sum(C_per_input)/n_inputs + reg*(w^2)/2n_all
        # C_total =  sum(C_per_input)/n + reg*(w^2)/2n
        # C_total = (sum(C_per_input)   + reg*(w^2)/2)   / n
        # C_total = (sum(C_per_input)   + (reg/2)*(w^2)) / n

        for x, y in data:
            a = self.feed_network(x)
            cost += self.cost_function.cost(a, y) / len(data)

        # np.linalg.norm(w) ** 2 is the same as sum(w^2)
        cost += .5 * (reg / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def get_accuracy_cost(self, data, reg):
        return self.test_network(data), self.total_cost(data, reg)


def build_string(data_type, new_result, old_result=None):
    # new_result = (.905, 1.2)
    new_acc = new_result[0]
    new_cost = new_result[1]
    new_acc_formatted = "{:<8}".format("{:.2%}".format(new_acc))
    result_string = "{} acc={}".format(data_type, new_acc_formatted)

    if old_result:
        old_acc = old_result[0]
        result_string += "{:<8}".format("{:+.2%}".format(new_acc - old_acc))

    result_string += "cost={:>7}".format("{:.4f}".format(new_cost))
    return result_string


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
