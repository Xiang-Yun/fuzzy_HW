import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol

def trimf(x, abc):
    assert len(abc) == 3,     'abc parameter must have exactly three elements.'
    a, b, c = np.r_[abc]       # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = np.zeros(len(x))

    # Left side
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]   # 返回非0的x座標
        y[idx]=(x[idx] - a) / float(b - a)

    # Right side
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx]=(c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y

# n個fuzzyset
def get_fuzzyset(arange, n):

    h = (arange[-1]-arange[0])/(n-1)
    tmp = arange[0]
    Ax = []
    for i in range(n):
        if i == 0:
            Ax.append(trimf(arange, [tmp, tmp, tmp + h]))
        elif i == n-1:
            Ax.append(trimf(arange, [tmp,  arange[-1],  arange[-1]]))
        else:
            Ax.append(trimf(arange, [tmp, tmp+h, tmp + 2*h]))
            tmp = tmp + h
    return Ax



def plot_fuzzyset(arange, component, xlabel='x', ylabel='Fuzzy membership'):
    fig, ax = plt.subplots(nrows=1, figsize=(10, 9))
    for i in range(len(component)):
        ax.plot(arange, component[i])
        ax.set_title('')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    plt.show()


def fuzzy_relation_product(x, y):
    return np.multiply(np.mat(x).reshape((-1, 1)), np.mat(y))


def fx(x1, x2):
    return 0.5*x1**2 + 0.2*x2**2 + 0.7*x2 - 0.5*x1*x2

def f(fg, x, y):
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    return fg.evalf(subs={x1:x, x2:y}) 
