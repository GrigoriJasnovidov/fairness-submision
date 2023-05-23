import numpy as np


def zeros_ones_to_classes(x, length=3):
    n = int(len(x) / length)
    l = []
    for i in range(n):
        z = x[i * length:i * length + length]
        l.append(z.argmax())
    return np.array(l, dtype=int)


def answer_creator(x, y, grouper):
    x = np.array(x)  # array of 1
    y = np.array(y)  # array of 0
    grouper = np.array(grouper)
    ans = []
    x_ind = 0
    y_ind = 0
    l = len(grouper)
    for i in range(l):
        if grouper[i] == 0:
            ans.append(y[y_ind])
            y_ind += 1
        else:
            ans.append(x[x_ind])
            x_ind += 1
    return np.array(ans)
