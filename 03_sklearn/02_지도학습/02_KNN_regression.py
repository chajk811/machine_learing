from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

mglearn.plots.plot_knn_regression(n_neighbors=1)
# plt.show()
# 최근접 이웃을 한 개만 이용할 때 예측은 그냥 가장 가까운 이웃의 타깃값

mglearn.plots.plot_knn_regression(n_neighbors=3)
# plt.show()
# 이웃이 둘 이상 사용하여 회귀 분석을 할 수 있다. 여러 개의 최근접 이웃을 사용할 땐 이웃 간의 평균이 예측됨.
# 평균 => 'average or mean' 으로 표현. KNeighborsRegressor 의 weights 매개변수가 기본값 'uniform'일 때는 np.mean 함수를 사용하여 단순 평균을 계산.
# 'distance'일 때는 거리를 고려한 가중치 평균(average)을 계산한다.

X, y = mglearn.datasets.make_wave(n_samples=40)

# wave 데이터셋을 훈련 세트와 테스트 세트로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 이웃의 수를 3으로 하여 모델의 객체를 만듭니다.
reg = KNeighborsRegressor(n_neighbors=3)
# 훈련 데이터와 타깃을 사용하여 모델을 학습시킵니다.
reg.fit(X_train, y_train)

# print("테스트 예측 : \n", reg.predict(X_test))

# score 메서드: 회귀일 땐 R^2 값을 반환한다.
# R^2(결정계수): 예측의 적합도를 측정한 것으로, 보통 0 에서 1 사이의 값이 된다.
# print("테스트 세트 R^2: {:.2f}".format(reg.score(X_test, y_test)))

fig, axes= plt.subplots(1, 3, figsize=(15, 4))
# -3 과 3 사이에 1000 개의 데이터 포인트를 만듭니다.
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 1, 3, 9 이웃을 사용한 예측을 합니다.
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title(
        "{} 이웃의 훈련 스코어: {:.2f} 테스트 코어: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)
        )
    )
    ax.set_xlabel("특성")
    ax.set_ylabel("타깃")
axes[0].legend(["모델예측", "훈련 데이터/타깃", "테스트 데이터/타깃"], loc="best")
plt.show()