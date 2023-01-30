# https://ja.wikipedia.org/wiki/K-means%2B%2B%E6%B3%95
# k-means++法 - Wikipedia

# https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
# OpenCV: K-Means Clustering in OpenCV

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

x = np.random.randint(25,100,25)
y = np.random.randint(175,255,25)
z = np.hstack((x,y))
z = z.reshape((50,1))
z = np.float32(z)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

compactness,labels,centers = cv.kmeans(z,2,None,criteria,10,cv.KMEANS_PP_CENTERS)

A = z[labels==0]
B = z[labels==1]

plt.hist(A,256,[0,256],color = 'red')
plt.hist(B,256,[0,256],color = 'blue')
plt.hist(centers,32,[0,256],color = 'yellow')
plt.show()

