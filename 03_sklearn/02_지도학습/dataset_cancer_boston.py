from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston

## dataset : cancer
cancer = load_breast_cancer()
# print(cancer.keys())
# scikit-learn 에 포함된 데이터셋은 실제 데이터셋 관련 정보를 담고 있는 Bunch 객체에 저장되어 있다.
# Bunch 객체는 파이썬 딕셔너리와 비슷하지만 점 표기법을 사용할 수 있다.(즉 bunch['key'] 대신 bunch.key)

# print(cancer.data.shape)
# print("클래스별 샘플 개수 :\n", {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
# print(cancer.feature_names)


## dataset : boston
boston = load_boston()
# print(boston.data.shape)

## 13개의 특성 뿐만 아니라 특성끼리 곱하여(상호작용) 의도적으로 확장.
X, y = mglearn.datasets.load_extended_boston()
# print(X.shape)