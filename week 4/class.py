import numpy as np
from matplotlib import pyplot as plt


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    # y = step_function(x)
    y = sigmoid_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


if __name__ == '__main__':
    main()
