import numpy as np

import nn2
import data_loader

# fixed hyper params
percentage = 1
layout = [784, 30, 10]
iterations = 30
early_stopping = 4

train_data, validate_data, test_data = data_loader.load_formatted_data(percentage)


def calculate_average_accuracy(times, size, rate, reg):
    average_accuracy = 0.0
    for _ in range(times):
        net = nn2.Network(layout)
        average_accuracy += net.train_test_network(
            train_data, iterations, size, rate, test_data, validate_data, reg, early_stopping) / times
    return average_accuracy


def grid_search():
    """
     simple raw grid search was used to find best size, rate and reg values
     notice this takes several hours to run (almost a day)
     (3 sizes * 5 rates * 5 regs) * 3 times each = 225 executions!
    """
    best_result = 0
    best_hyper_params = (None, None, None)

    for size in range(5, 15 + 1, 5):  # 5 10 15
        for rate in np.arange(.2, .6 + .1, .1):  # .2 .3 .4 .5 .6
            rate = round(rate, 2)
            for reg in range(4, 8 + 1, 1):  # 4 5 6 7 8
                result = calculate_average_accuracy(3, size, rate, reg)
                if result > best_result:
                    best_result = result
                    best_hyper_params = (size, rate, reg)

    print("{} {}".format(best_result, best_hyper_params))


grid_search()
# ~ 0.9661 (10, .2, 4)
