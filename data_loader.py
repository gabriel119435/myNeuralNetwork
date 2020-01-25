import gzip
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np


def load_raw_data():
    """
    returns 3 tuples: train, validate and test
    each tuple is build as (input, result)
    input and result are ndarray: (50000,784) and (50000,) respectively
    sizes are 50000(train), 10000(validate) and 10000(test)
    each element from input is a ndarray (784,)
    each element from result is int64
    """
    f = gzip.open('entire_data.gz', 'rb')
    train, validate, test = pickle.load(f, encoding="latin1")
    f.close()
    return train, validate, test


def convert_data_return_percentage(data, percentage):
    formatted_data = list(zip([x.reshape((784, 1)) for x in data[0]],
                              [vectorized_result(y) for y in data[1]]))
    return formatted_data[:int(percentage * (len(formatted_data) / 100))]


def vectorized_result(number):
    # converts 7 into [[0], [0], [0], [0], [0], [0], [1], [0], [0], [0]]
    # converts 2 into [[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]
    e = np.zeros((10, 1))
    e[number] = 1.0
    return e


def load_formatted_data(percentage):
    """
    to each data set:
    input is converted from numpy ndarray (784,) to numpy ndarray (784,1)
    diff explained here: https://stackoverflow.com/a/49220851/3026886
    result is converted from int64 to numpy ndarray (10,1)
    output lists of tuples
    then, retrieves a percentage of total examples
    """
    train, validate, test = load_raw_data()

    # saves random images at numbers directory with the result as the name of image
    # just to expose visually some data
    index = random.randint(0, len(train[0]))
    image = train[0][index].reshape([28, 28])
    plt.imshow(image, "gray")
    plt.savefig("numbers/{}.png".format(train[1][index]))

    train = convert_data_return_percentage(train, percentage)
    validate = convert_data_return_percentage(validate, percentage)
    test = convert_data_return_percentage(test, percentage)

    print("train={}, validate={}, test={}\n".format(len(train), len(validate), len(test)))

    return train, validate, test
