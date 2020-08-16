import gzip
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np


def _load_raw_data():
    """
    returns 3 tuples: train, validate and test
    each tuple has 2 ndarrays here called input and output
    input and output are (someSize,784) and (someSize,) respectively
    someSize is 50000(train), 10000(validate) and 10000(test)
    each element from input is a ndarray (784,)
    each element from result is int64
    """
    f = gzip.open('entire_data.gz')
    train, validate, test = pickle.load(f, encoding="latin1")
    f.close()

    # printing data structure on input
    _input = train[0]
    print("input.size={}".format(len(_input)))
    print("input.shape={}".format(_input.shape))
    _one_input = _input[0]
    print("oneInput.size={}".format(len(_one_input)))
    print("oneInput.shape={}\n".format(_one_input.shape))

    # printing data structure on output
    _output = train[1]
    print("output.size={}".format(len(_output)))
    print("output.shape={}".format(_output.shape))
    _one_output = _output[0]
    print("oneOutput.value={}\n".format(_one_output))

    print("entire dataset:")
    print("train={}, validate={}, test={}\n".format(len(train[0]), len(validate[0]), len(test[0])))
    return train, validate, test


def _format_data_filter_percentage(data, percentage):
    """
    to each data(which is a tuple explained at _load_raw_data):
      1. input is converted from ndarray (784,) to ndarray (784,1)
         diff explained here: https://stackoverflow.com/a/49220851/3026886
      2. result is converted from int64 to ndarray (10,1)
      3. returns a list of tuples

    then shuffles it and retrieves first n elements from it according to percentage
    """
    formatted_data = list(
        zip(
            [x.reshape((784, 1)) for x in data[0]],
            [_vectorized_result(y) for y in data[1]]
        )
    )
    random.shuffle(formatted_data)
    return formatted_data[:int(percentage * (len(formatted_data) / 100))]


def _vectorized_result(number):
    # converts 7 into [[0], [0], [0], [0], [0], [0], [0], [1], [0], [0]]
    # converts 2 into [[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]]
    #                   0    1    2    3    4    5    6    7    8    9
    e = np.zeros((10, 1))
    e[number] = 1.0
    return e


def load_formatted_data(percentage):
    """
    saves a random number image with its value as the file name
    formats data and returns a percentage of it
    """
    train, validate, test = _load_raw_data()

    # saves random images at /numbers with the result as the name of image
    # just to visually expose some data
    index = random.randint(0, len(train[0]))
    image = train[0][index].reshape([28, 28])
    plt.imshow(image, "gray")
    plt.savefig("numbers/{}.png".format(train[1][index]))

    print("percentage={}\n".format(percentage))

    train = _format_data_filter_percentage(train, percentage)
    validate = _format_data_filter_percentage(validate, percentage)
    test = _format_data_filter_percentage(test, percentage)

    print("selected dataset:")
    print("train={}, validate={}, test={}".format(len(train), len(validate), len(test)))
    return train, validate, test


# just to test it
load_formatted_data(1)
