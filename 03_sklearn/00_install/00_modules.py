## Numpy
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
# print(x)


## Scipy
from scipy import sparse

# 대각선 원소는 1 이고 나머지는 0인 2차원 Numpy 배열을 만듭니다.
eye = np.eye(4)
# print(eye)

# Numpy 배열을 CSR 포맷의 Scipy 희박 행렬로 변환합니다.
# 0이 아닌 원소만 저장됩니다.
sparse_matrix = sparse.csr_matrix(eye)
# print(sparse_matrix)

# 보통 희소 행렬을 0이 모두 채워진 2차원 배열로부터 만들지 않음.(메모리 부족 문제)
# 희소 행렬을 직접 만들 수 있어야 합니다. (COO 포맷)

data = np.ones(4)
# print(data)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
# print(eye_coo)

## Matplotlib
import matplotlib.pyplot as plt

# -10 에서 10 까지 100개의 간격으로 나뉘어진 배열을 생성
x = np.linspace(-10, 10, 100)
# print(x)
# 사인함수를 사용하여 y 배열을 생성합니다.
y = np.sin(x)
plt.plot(x, y, marker="x")
# plt.show()

## Pandas
import pandas as pd

# 회원 정보가 들어간 간단한 데이터 셋을 생성합니다.
data = {'Name': ["John", "Anna", "Peter", "Linda"], 
'Location' : ["New York", "Paris", "Berlin", "London"],
'Age' : [24, 13, 53, 33]
}

data_pandas = pd.DataFrame(data)
# print(data_pandas

# Age 열의 값이 30이상인 모든 행을 선택합니다.
# print(data_pandas[data_pandas.Age > 30])