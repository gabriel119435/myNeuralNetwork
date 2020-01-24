import gzip
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt


# loads data from mnist file

def load_data():
    # returns 3 tuples: train, validate and test
    # train size: 50000
    # validate and test sizes: 10000 each
    # (input,result) both are numpy ndarray
    # each element from input is a numpy ndarray (784,)
    # each element from result is numpy int64
    f = gzip.open('entire_data.gz', 'rb')
    train, validate, test = pickle.load(f, encoding="latin1")
    f.close()
    return train, validate, test


def load_formatted_data(percentage):
    # to each data set:
    # input is converted from numpy ndarray (784,) to numpy ndarray (784,1)
    # diff explained at https://stackoverflow.com/a/49220851/3026886
    # result is converted from int64 to numpy ndarray (10,1)
    # output lists of tuples
    # then, retrieves a percetage of total examples
    train, validate, test = load_data()

    # saves random images with the result as the name of image, just to expose some data visually
    index = random.randint(0, len(train[0]))
    image = train[0][index].reshape([28, 28])
    plt.gray()
    plt.imshow(image)
    plt.savefig("numbers/{}.png".format(train[1][index]))

    train = convert_data(train)
    validate = convert_data(validate)
    test = convert_data(test)

    train = train[:int(len(train) * percentage / 100)]
    test = test[:int(len(test) * percentage / 100)]
    validate = validate[:int(len(validate) * percentage / 100)]

    print("train={}, validate={}, test={}\n".format(len(train), len(validate), len(test)))

    return train, validate, test


def convert_data(data):
    return list(
        zip(
            [np.reshape(x, (784, 1)) for x in data[0]],
            [vectorized_result(y) for y in data[1]]
        )
    )


def vectorized_result(number):
    # converts 7 into [[0], [0], [0], [0], [0], [0], [1], [0], [0], [0]]
    # converts 2 into [[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]
    e = np.zeros((10, 1))
    e[number] = 1.0
    return e
