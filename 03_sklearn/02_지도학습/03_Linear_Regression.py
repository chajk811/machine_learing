from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

# 기울기 파라미터(w)는 가중치 또는 계수라고 하며, lr 객체의 coef 속성에 저장되어 있고
# 편향 또는 절편 파라미터(b)는 intercept 속성에 저장되어 있습니다.

# print("lr.coef_: ", lr.coef_)
# print("lr.intercept_: ", lr.intercept_)

# print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
# print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))

# 1차원 데이터셋에서는 모델이 매우 단순하므로 과대적합을 걱정할 필요 없지만,
# 고차원 데이터셋에서는 선형 모델의 성능이 매우 높아져 과대적합될 가능성이 높다.

# 보스턴 주택가격 데이터셋: 샘플 506개, 특성 104개
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))