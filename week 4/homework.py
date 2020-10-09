import numpy as np


# method provided by prof.
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# method provided by prof.
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# solution for homework : OR from NAND
def OR_FROM_NAND(x1, x2):
    s1 = NAND(x1, x1)
    s2 = NAND(x2, x2)
    y = NAND(s1, s2)
    return y


def main():
    print('OR\n', list(OR(x1, x2) for x2 in [0, 1] for x1 in [0, 1]))

    print('OR_FROM_NAND\n', list(OR_FROM_NAND(x1, x2) for x2 in [0, 1] for x1 in [0, 1]))


if __name__ == '__main__':
    main()
