import ru_libs as libs
import numpy as np
import matplotlib.pyplot as plt

#x =     np.asarray([-2, -1, 0, 1, 2])
#lmbda = np.asarray([-1, 0, 1, 2, 3])
#print(libs.yeojohnson_x(x, lmbda))


def  f(x):
    return x[:, 0]

g = libs.yeojohnson_f(f, [0.5])
def h(x):
    ya_ll = []
    for xi in x:
        if xi>0:
            s = 1 + xi/2
            y = s * (-1 + s**2)
        else:
            s = (1 - xi*3/2)
            y = s**(-1/3) * (1 - s**(2/3))
        ya_ll.append(y)

    return ya_ll


x = np.expand_dims(np.arange(-10, 10, 0.01), axis=-1)         # n, 1
print(x.shape)
y_g = g(x)
y_h = h(x)

print(y_h)
print(y_g)

plt.plot(x, y_g)
plt.plot(x, y_h)
plt.show()