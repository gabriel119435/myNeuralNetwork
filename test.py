import data_loader
import simplest_network
import better_network

percentage = 100
train_data, validate_data, test_data = data_loader.load_formatted_data(percentage)

sizes = [784, 30, 10]
iterations = 30
mini_size_batch = 10
learning_rate = 3
# 91.29% - 95.34% -- book
# 90.87% - 95.25% -- original code
# 90.84% - 95.41% -- this
simple = simplest_network.Network(sizes)
simple.train_test_network(train_data, iterations, mini_size_batch, learning_rate, test_data)

sizes = [784, 30, 10]
iterations = 30
mini_size_batch = 10
learning_rate = .4
regularization_factor = 6
early_stopping = 4
# 96.49% -- book
# learning_rate = .5 regularization_factor = 5
# {train acc=98.19%  cost= 0.1458} {valid acc=95.68%  cost= 0.3498} -- original code (bug sums 0 L2 reg on total cost)
# learning_rate = .4 regularization_factor = 6
# {train acc=97.20%  +0.03%  cost= 0.3553} {valid acc=96.49%  -0.06%  cost= 0.9351}
improved = better_network.Network(sizes)
improved.train_test_network(
    train_data,
    validate_data,
    test_data,
    iterations,
    mini_size_batch,
    learning_rate,
    regularization_factor,
    early_stopping
)
