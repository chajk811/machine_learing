from sklearn.datasets import load_iris
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

iris_dataset = load_iris()

# print(iris_dataset.keys())
# print(iris_dataset['DESCR'][:193])
# print(iris_dataset['target_names'])
# print(iris_dataset['feature_names'])

# print(type(iris_dataset['data']))
# print('data의 크기 : ', iris_dataset['data'].shape)
# 이 배열은 150개의 붓꽃 데이터를 가지고 있음.
# 머신러닝에서 각 아이템은 '샘플'이라 하고, 속성은 '특성'이라고 부름.

# print(iris_dataset['data'][:5])
# print(type(iris_dataset['target']))
# print(iris_dataset['target'].shape)
# target 은 각 원소가 붓꽃 하나에 해당하는 1차원 배열

# 붓꽃의 종류는 0에서 2까지의 정수로 기록(3종류)
# print(iris_dataset['target'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# Numpy 배열을 pandas의 DataFrame으로 변경해야 한다.
# X_train 데이터를 사용해서 데이터프레임을 만든다.
# 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터 프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
# pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=8, cmap=mglearn.cm3)
# plt.show()

from sklearn.neighbors import KNeighborsClassifier
# 모델을 사용하려면 클래스로부터 객체를 만들어야 함.
# 이때 모델에 필요한 매개변수를 넣는다.(이웃의 개수)
knn = KNeighborsClassifier(n_neighbors=1)

# knn 객체는 훈력 데이터로 모델을 만들고 새로운 데이터 포인트에 대해 예측하는 알고리즘을 캡슐화한 것.
# 또한, 훈련 데이터로부터 추출한 정보를 담고 있음.
knn.fit(X_train, y_train)
# fit 매서드는 knn 객체 자체를 반환함.
 
X_new = np.array([[5, 2.9, 1, 0.2]])
# print(X_new.shape)

prediction = knn.predict(X_new)
# print(prediction)
# print(iris_dataset['target_names'][prediction])

y_pred = knn.predict(X_test)
# print(y_pred)
print('테스트 세트의 정확도: {:.2f}'.format(np.mean(y_pred == y_test)))
print('테스트 세트의 정확도: {:.2f}'.format(knn.score(X_test, y_test)))
