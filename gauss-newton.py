# https://qiita.com/picric_acid/items/1c9e3ef9625ca7e4d7c3
# Fortranで非線形最小二乗法(Gauss-Newton法) - Qiita

# https://ja.wikipedia.org/wiki/ガウス・ニュートン法
# ガウス・ニュートン法 - Wikipedia

# https://sterngerlach.github.io/doc/gauss-newton.pdf
# ガウス・ニュートン法とレーベンバーグ・マーカート法



from numpy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt

b1=[]

for h in range(200):
      x0=np.array([0.038,0.194,0.425,0.626,1.253,2.500,3.740])
      y0=np.array([0.050,0.127,0.094,0.2122,0.2729,0.2665,0.3317])
      j=np.zeros((2,7))
      #beta1=0.362
      #beta2=0.556
      beta1=2
      beta2=0.01*h
      dbeta=0.1

      for i in range(10):
            func = (x0*beta1)/(x0+beta2)
            deltafunc1 = (x0*(beta1+dbeta))/(x0+beta2)
            deltafunc2 = (x0*beta1)/(x0+beta2 + dbeta)
            j[0] = (deltafunc1 - func)/dbeta
            j[1]= (deltafunc2 - func)/dbeta
            e = y0 - func
            
            a = np.matmul(j,j.T)
            b = -np.matmul(j,e)
            d=solve(a+0.001, b)
            
            beta1-=d[0]
            beta2-=d[1]
      b1.append(d[0]**2+d[1]**2)

fig = plt.figure()
ax = fig.gca()
ax.plot(np.log10(b1),color="green")
plt.show()