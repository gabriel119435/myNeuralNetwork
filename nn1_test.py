import data_loader
import nn1

# fixed hyper params
percentage = 100
layout = [784, 30, 10]
iterations = 20
test_every_iteration = True

train_data, validate_data, test_data = data_loader.load_formatted_data(percentage)


def get_avg_acc(times, size, rate):
    avg_acc = 0.0
    for _ in range(times):
        net = nn1.Network(layout)
        avg_acc += net.train_test_network(train_data, iterations, size, rate, test_data, test_every_iteration) / times
    return avg_acc


# variable hyper params
mini_size_batch = 10
learning_rate = 5

print(get_avg_acc(3, mini_size_batch, learning_rate))
# .947
