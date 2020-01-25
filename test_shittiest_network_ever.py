import data_loader
import shittiest_network_ever

data_percentage = 100
layout = [784, 30, 10]
iterations = 30
mini_size_batch = 10
learning_rate = 3
test_every_iteration = True

train_data, validate_data, test_data = data_loader.load_formatted_data(data_percentage)
net = shittiest_network_ever.Network(layout)
# usually ~94.95%
net.train_test_network(train_data, iterations, mini_size_batch, learning_rate, test_data, test_every_iteration)
