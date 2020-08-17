import data_loader
import nn2

# fixed hyper params
percentage = 100
layout = [784, 30, 10]
iterations = 20
early_stopping = 4

train_data, validate_data, test_data = data_loader.load_formatted_data(percentage)


def get_avg_acc(times, size, rate, reg):
    avg_acc = 0.0
    for _ in range(times):
        net = nn2.Network(layout)
        avg_acc += net.train_test_network(train_data, iterations, size, rate, test_data, validate_data, reg,
                                          early_stopping) / times
    return avg_acc


def grid_search():
    """
     simple raw grid search was used to find best size, rate and reg values
    """
    best_acc = 0
    best_hyper_params = (None, None, None)
    size_arr = [10, 20]
    rate_arr = [.1, .2, .3]
    reg_arr = [3, 4, 5]

    for size in size_arr:
        for rate in rate_arr:
            for reg in reg_arr:
                curr_acc = get_avg_acc(3, size, rate, reg)
                if curr_acc > best_acc:
                    best_acc = curr_acc
                    best_hyper_params = (size, rate, reg)

    print("{} {}".format(best_acc, best_hyper_params))


grid_search()
# ~ 0.965 (10, .2, 4)
