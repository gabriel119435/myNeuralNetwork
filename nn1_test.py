import data_loader
import nn1

percentage = 100
layout = [784, 30, 10]
iterations = 30
mini_size_batch = 10
learning_rate = 3
test_every_iteration = True

train_data, validate_data, test_data = data_loader.load_formatted_data(percentage)
net = nn1.Network(layout)
net.train_test_network(train_data, iterations, mini_size_batch, learning_rate, test_data, test_every_iteration)
# ~ .95
