### Diagnosis

- Malignant 악성

- Benign 양성

악성인지 양성인지 판단 (숫자 데이터를 보고)



### 패키지

numpy => 선형대수, 숫자에 관련된 행렬을 연산하기 위해여 사용

pandas => csv를 읽기 위해

matplotlib.pyplot

seaborn

=> 그래프를 이쁘게 그리기 위해 위 두 가지를 사용



sklearn

```python
from sklearn.model_selection import train_test_split
# train_test_split
# 학습 데이터와 테스트 뎅터를 분리하는 데 사용한다.

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
# KFold, cross_val_score
# Cross validation 을 구현할 때 사용한다.

from sklearn import metrics
# 매트릭을 확인하기 위해 사용한다.

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
```



`df.info()`

데이터의 타입과 null 개수를 출력한다.



`Unnamed:32 0 non-null float64`

=>RangeIndex:569 인데  non-null 0이므로 전부다 null

id는 필요없는 정보이므로



`df.drop()`

데이터프레임에서 특정 데이터를 삭제한다.



`df.map()`

데이터프레임 안에서 값을 대체하고 싶을 때 사용한다. 복잡한 연산을 할 경우 df.apply()를 사용한다.



`df.describe()`

데이터프레임의 개요를 출력한다.



`sns.countplot()`

데이터의 개수를 출력하는 그래프를 그린다.

ex) sns.countplot(data['diagnosis']) => diagnosis 칼럼의 분포를 그래프로 확인할 수 있다.

확인하려는 데이터가 너무 불균형하면 정확하지 않아 맞춰주는 작업이 필요하다.



`train_test_split()`

학습/검증 데이터를 분리한다.

```python
## Split Train and Test

train, test = train_test_split(data, test_size=0.2, random_state=2019)
# size=0.2 =>20% 사용하겠다 라는 뜻

x_train = train.drop(['diagnosis'], axis=1)
y_train = train.diagnosis
# x => (어떤 모델) => y
# x 를 어떤 모델에 넣어서 y를 끌어내고 싶다.
# x의 data feature 들을 활용하여 원하는 컬럼을 예측

x_test = teat.drop(['diagnosis'], axis=1)
y_test = test.diagnosis
```



### 여러 메서드를 사용해서 모델이 얼마나 정확한지 정확도를 검증한다.

Support  Vector Machine  

```python
model = svm.SVC(gamma='scale')
model.fit(x_train, y_train)
# model.fit()
# 주어진 데이터로 모델을 학습한다.

## 학습이된 모델을 검증시킨다.
y_pred = model.predict(x_test)
# model.precict()
# 주어진 x 값에 대해서 y 를 예측한다.

print('SVM : %.2f' %(metrics.accuracy_score(y_pred, y_test) * 100))
# accuracy_score()
# 정확도를 측정한다.
```



### 에러

```bash
$pip install scikit-learn
=> 0.22 버전이 깔림

Traceback (most recent call last):
  File "test2.py", line 1, in <module>
    import sklearn
  File "C:\Users\jk\AppData\Local\Programs\Python\Python37\lib\site-packages\sklearn\__init__.py", line 75, in <module>
    from .utils._show_versions import show_versions
  File "C:\Users\jk\AppData\Local\Programs\Python\Python37\lib\site-packages\sklearn\utils\_show_versions.py", line 12, in <module>
    from ._openmp_helpers import _openmp_parallelism_enabled
ImportError: DLL load failed: 지정된 모듈을 찾을 수 없습니다.
```

scikit-learn 의 버전을 dowgrade 시켜서 해결

원인은 잘 모르겠음.



```bash
$ pip install scikit-learn==0.20.2
```

```python
# 버젼 확인
import sklearn
print(sklearn.__version__)
```



### cross_validation

![cross_validation](https://user-images.githubusercontent.com/52685241/71572570-1e466100-2b23-11ea-9a0f-c6c17033618e.png)



### 참고

[https://everyday-deeplearning.tistory.com/entry/%ED%98%84%EC%97%85%EC%97%90%EC%84%9C-%EB%A7%8E%EC%9D%B4-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-Python-%EB%AA%A8%EB%93%88-ScikitLearn](https://everyday-deeplearning.tistory.com/entry/현업에서-많이-사용하는-Python-모듈-ScikitLearn)