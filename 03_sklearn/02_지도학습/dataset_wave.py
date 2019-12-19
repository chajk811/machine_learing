from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

## wave 데이터 셋 : 입력 특성 하나와 모델링할 타깃 변수(또는 응답)을 가짐
## 회귀 알고리즘 설명을 위한 인위적으로 만듦.

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("특성")
plt.ylabel("타깃")
plt.show()