## scikit-learn 설치

```bash
$ pip install numpy scipy matplotlib ipython scikit-learn pandas pillow
```

-----

### Numpy

- 파이썬으로 과학 계산을 하기 위한 필수 패키지

- 다차원 배열을 위한 기능, 선형 대수 연산, 푸리에 변환, 유사 난수 생성기를 포함

  - 유사 난수 : 초깃값을 이용하여 이미 결정되어 있는 메커니즘에 의해 생성되는 난수로, 초깃값을 알면 언제든 같은 값을 다시 만들 수 있으므로 진짜 난수와 구별하여 유사 난수라 한다.

- Numpy의 핵심 기능은 다차원(n-차원) 배열인 ndarray 클래스 입니다. 

  이 배열의 모든 원소는 동일한 데이터 타입



#### 오류

실습 파일을 `numpy.py`로 작성 => `AttributeError: module 'numpy' has no attribute 'array'`

파일명을 모듈명으로 작성은 피한다.

---

### Scipy

- 과학 계산용 함수를 모아놓은 파이썬 패키지
- 고성능 선형 대수, 함수 최적화, 신호 처리, 특수한 수학 함수와 통계 분포 등을 포함
- scikit-learn 은 알고리즘을 구현할 때 Scipy 의 여러 함수를 사용
- `scipy.sparse` (희소행렬) : 0을 많이 포함한 2차원 배열을 저장할 때 사용

------

### Matplotlib

- 파이썬의 대표적인 과학 계산용 그래프 라이브러리
- 선그래프, 히스토그램, 산점도 등을 지원
- 주피터 노트북에서 `%matplotlib notebook`  혹은 `%matplotlib inline` 명렁어 선언 후 사용

- 커널 환경에서는 `plt.plot()`으로 그래프를 지정해준 후 `plt.show()` 로 출력할 수 있다.

-----

### Pandas

- 데이터 처리와 분석을 위한 파이썬 라이브러리(테이블을 수정하고 조작하는 다양한 기능 제공)
- DataFrame이라는 데이터 구조를 기반 => 엑셀의 스프레드시트와 비슷한 테이블 형태
- 전체 배열의 원소가 동일한 타입이어야 하는 Numpy와는 달리 Pandas는 각 열의 타입이 달라도 됨.
- SQL, 엑셀파일, CSV 파일같은 다양한 파일의 데이터를 읽어 들일 수 있음.

-----

### mglearn

- https://github.com/rickiepark/introduction_to_ml_with_python

```python
# 앞으로의 실습에서 기보으로 import 하고 시작

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
```

----

### 버전확인

```python
import sys
import numpy as np
import matplotlib
import pandas as pd
import scipy
import IPython
import sklearn

print('Python 버전:', sys.version)
print('Package 버전:', Package.__version__)
```

버전이 정확히 같아야 하는 것은 아니지만 scikit-learn 은 가능한 최신버전 0.20.2 이상

