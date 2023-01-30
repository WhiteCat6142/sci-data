import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 感染モデル SIR
# パラーメーターを変えて
def func(v, t, b, r):
    return [1,-b*v[1]*v[2], b*v[1]*v[2]-r*v[2]]

b=1
r=2
x=0.1
v0 = [0,1-x, x]
t = np.arange(0, 100, 0.01)

v1= odeint(func, v0, t, args=(b, r))
v2= odeint(func, v0, t, args=(3, 1))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(v1[:, 0], v1[:, 1], v1[:, 2],color='green')
ax.plot(v2[:, 0], v2[:, 1], v2[:, 2],color='red')
ax.set_xlabel('T')
ax.set_ylabel('X1')
ax.set_zlabel('X2')
plt.show()
