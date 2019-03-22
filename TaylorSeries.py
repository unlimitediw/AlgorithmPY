import math

import matplotlib.pyplot as plt
import numpy as np


def ex(a):
    return math.exp(a)

def calc_e_small(x,a):
    # at a = 0
    # taylor series of exp(x) = (x^0 / 0!) + (x^1 / 1!) + (x^2 / 2!) +...
    n = 10
    x-=a
    # cumprod: return the cumulative product of elements along a given axis
    f = np.arange(1, n + 1).cumprod()
    b = np.array([x] * n).cumprod()
    d = np.array([ex(a)] * n)
    return np.sum(d * b / f) + 1

'''
#高精度泰勒展开for e^x
def calc_e(x):
    reverse = False
    if x < 0:
        x = -x
        reverse = True
    LN2 = 0.69314718055994530941723212145818
    c = x / LN2
    a = int(c + 0.5)
    b = x - a * LN2
    y = (2 ** a) * calc_e_small(b,0)
    if reverse:
        return 1 / y
    return y
'''



t1 = np.linspace(-2, 0, 10, endpoint=False)
t2 = np.linspace(0, 2, 20)
t = np.concatenate((t1, t2))
print(t)
# return a new array with same shape and type as a given array
hypo = np.empty_like(t)
y = np.empty_like(t)
acc = 0
for i, x in enumerate(t):
    # a值越大，在越远处拟合效果越好
    hypo[i] = calc_e_small(x,0)
    y[i] = math.exp(x)
    #print(hypo[i] - y[i])
    acc += abs(hypo[i] - y[i])
print(acc/len(t))

# rcParams: set the matplotlib backend to one of the known backends
plt.plot(t, y, 'r-', t, hypo, 'go', linewidth=2)
plt.title(u'Taylor application')
plt.xlabel('x')
plt.ylabel('exp(x)')
plt.grid(True)
plt.show()
