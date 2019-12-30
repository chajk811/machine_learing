## 실습

내가 재밌게 본 영화와 다른 사용자들의 평가간의 상관관계(correlation)로 영화를 추천



### data set

https://www.kaggle.com/rounakbanik/the-movies-dataset



### module 사용하기

```python
import numpy as np
import pandas as pd
import json
```

numpy란?

- Numerical Python 의 줄임말로 고성능 수치계산을 위해 만들어진 파이썬 패키지

- 파이썬을 이용한 data 분석을 하기 위해서는 numpy, pandass를 알아야 한다.

- numpy : 행렬 연산 패키지, pandas : 데이터 분석 패키지

- 넘파이(numpy)와 pandas 는 외부 라이브러리이다. (install 필요)

  ```bash
  pip install numpy
  pip install pandas
  ```





### Pivot Table

| user/title | 영화1 | 영화2 | 영화3 |
| ---------- | ----- | ----- | ----- |
| 1          | 3.5   |       | 1.0   |
| 2          | 5.0   | 4.5   | 2.5   |
| 3          | 0.5   | 3.5   |       |
| 4          |       |       | 0.5   |



### Pearson Correlation

| userId/title | 영화1_s1      | 영화2_S2       |
| ------------ | ------------- | -------------- |
| 1            | 2.5           |                |
| 2            | 5.0           | 4.5            |
| 3            | 4.5           | 1.5            |
| 4            |               |                |
| 평균         | 3 = s1.mean() | 3 = s2. mean() |



| userId/title | s1_c                  | s2_c                  |
| ------------ | --------------------- | --------------------- |
| 1            | -0.5                  |                       |
| 2            | 2.0                   | 1.5                   |
| 3            | 1.5                   | -1.5                  |
| 4            |                       |                       |
|              | s1_c = s1 - s1.mean() | s2_c = s2 - s2.mean() |



| userId/title | s1_c | s2_c | s1_c * s2_c |
| ------------ | ---- | ---- | ----------- |
| 1            | -0.5 |      |             |
| 2            | 2.0  | 1.5  | 3.0         |
| 3            | 1.5  | -1.5 | -2.25       |
| 4            |      |      |             |
| Sum          |      |      | 0.75        |



곱했을 때 큰 숫자가 나오면 상관관계가 크다.

곱했을 때 음수로 큰수가 나오면 음의 방향으로 상관관계가 크다.







### error

1. 파일 로드 에러

```python
meta = pd.read_csv('파일명.csv')
meta.head()
---
sys:1: DtypeWarning: Columns (10) have mixed types. Specify dtype option on import or set low_memory=False.
```

dtypes 를 추측하는 것이 매우 많은 메모리를 요구하기 때문이다. 팬더는 각 열의 데이터를 분석하여 설정할 dtype을 결정하려고 합니다.(전체 파일을 읽은 후 열이 가져야하는 dtype을 판별 할 수 만 있습니다.)

해결 : 처음부터 dtype을 정해주거나, 어디 열까지 확인한다는 "foobar"을 지정해주는 방법이 있다. 혹은 `low_memory=False` 을 지정해준다. 

`meta = pd.read_csv('the-movies-dataset/movies_metadata.csv', low_memory=False)`

참고 : http://shovel-ing.blogspot.com/2019/06/pandasreadcsv.html



2. 데이터 출력 생략

`DataFrame.to_strint()` 활용하면 된다. 하지만, 열과 행이 정확히 맞지 않는 경우도 더러 존재한므로 주의가 필요하다.

참고 : https://kongdols-room.tistory.com/107