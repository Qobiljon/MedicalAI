import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def main():
    print(f'AND = &')
    print(f'NAND = ⊼')
    print(f'OR = +')
    print(f'XOR = ⊕\n')
    print('x1\tx2\t|\t&\t⊼\t+\t⊕')
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f'{x1}\t{x2}\t|\t{AND(x1, x2)}\t{NAND(x1, x2)}\t{OR(x1, x2)}\t{XOR(x1, x2)}')
    # print('AND', list(AND(x1, x2) for x1 in [0, 1] for x2 in [0, 1]))
    # print('NAND', list(NAND(x1, x2) for x1 in [0, 1] for x2 in [0, 1]))
    # print('OR', list(OR(x1, x2) for x1 in [0, 1] for x2 in [0, 1]))
    # print('XOR', list(XOR(x1, x2) for x1 in [0, 1] for x2 in [0, 1]))


if __name__ == '__main__':
    main()
