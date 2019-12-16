import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(n_samples=40, centers=2, random_state=20)
# scikit-learn 은 데이터 분류를 목적으로 데이터를 생성해 주는 make_blobs 라는 함수를 제공.
# 이를 이용해서 위와 같이 2종류의 총 40개의 샘플 데이터를 생성한다.

clf = svm.SVC(kernel='linear')
# clf = svm.SVC(gamma='scale')
# clf = svm.SVC(kernel='rbf')
# clf = svm.SVC(kernel='poly')
clf.fit(x, y)

# SVM 은 선형분류와 비선형 분류를 지원
# 그 중 선형 모델을 위해 kernel 을 linear 로 지정
# 비선형에 대한 kernel 로는 rbf와 poly 등이 있음.

newData = [[3, 4]]
# print(clf.predict(newData))

# 시각화
# 샘플데이터와 초평면(Hyper-Plane), 지지벡터(Support Vector)를 그래프에 표시하는 코드

# 샘플 데이터 표현
plt.scatter(x[:,0], x[:,1], c=y, s=30, cmap=plt.cm.Paired)

# 초평면(Hyper-Plane) 표현
ax = plt.gca()

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xlim = ax.get_xlim()
ylim = ax.get_ylim()
 
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
 
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])
 
# 지지벡터(Support Vector) 표현
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=60, facecolors='r')
 
plt.show()
