import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# https://toukei-lab.com/mahalanobis
#【３分で分かる】マハラノビス距離って何！？分かりやすく解説！｜スタビジ
# https://qiita.com/kotai2003/items/c66cfcc5266edec9a063
#【異常検知】マハラノビス距離を嚙み砕いて理解する (2) - Qiita
# https://www.nicovideo.jp/watch/sm23615316
# 月読アイの理系なお話 『距離ってなに？ [実務編]』 - ニコニコ動画

# Davis Dataset
# https://www.kaggle.com/datasets/ravinduabey/davis-data-set
df = pd.read_csv('Davis.csv')
print(df.head())
print(df.columns)
df = df[['weight', 'height']]

data = df.to_numpy()

print('data shape',data.shape)

#mean
mu_mat = np.mean(data, axis=0)
print(mu_mat)

#covariance matrix
cov_mat = np.cov(data.T)
cov_i_mat = np.linalg.pinv(cov_mat)
print(cov_mat)
print(cov_i_mat)

x0=np.linspace(30,200,100)
y0=np.linspace(50,210,100)
x=x0-mu_mat[0]
y=y0-mu_mat[1]

data_m_mat = data - mu_mat

def mahala_map(j):
 x1=np.row_stack((x,np.full_like(x,j)))
 s=x1*(cov_i_mat@x1)
 y1=np.sqrt(s[0]+s[1])
 return y1

mahala = np.array(list(map(mahala_map, y)))

print('mahala shape',mahala.shape)

fig3 = plt.figure()
ax1 = fig3.add_subplot()
ax1.scatter(data.T[0], data.T[1])
ax1.set_title('Mahalanobis')
CS = ax1.contour(x0, y0, mahala)
ax1.clabel(CS)
ax1.set_xlabel('weight')
ax1.set_ylabel('height')
ax1.set_aspect('equal')
ax1.grid()
plt.show()
