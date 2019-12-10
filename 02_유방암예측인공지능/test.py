import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('breast-cancer-wisconsin-data/data.csv')

# data.head()

# data.info()

data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
# 컬럼 삭제
# drop 명령어를 통해 컬럼 전체를 삭제할 수 있다. axis=1은 컬럼을 뜻한다.
# axis=0인 경우, 로우를 삭제하며 이것이 디폴트이다.
# inplace의 경우 drop한 후의 데이터프레임으로 기존 데이터프레임을 대체하겠다는 뜻이다.
# 즉, 아래의 inplace=True는 df = df.drop('A', axis=1)과 같다.

data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# print(data.head())

# data.describe()
# sns.countplot(data['diagnosis'])

## Split Train and Test
train, test = train_test_split(data, test_size=0.2, random_state=2019)

x_train = train.drop(['diagnosis'], axis=1)
y_train = train.diagnosis

x_test = test.drop(['diagnosis'], axis=1)
y_test = test.diagnosis

# print(len(train), len(test))

## SVM (Support Vector Machine)
model = svm.SVC(gamma='scale')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# print('SVM: %.2f' %(metrics.accuracy_score(y_pred, y_test) * 100))

## DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# print('DecisionTreeClassifier: %.2f' %(metrics.accuracy_score(y_pred, y_test) * 100))

## KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# print('KNeighborClassifier: %.2f' %(metrics.accuracy_score(y_pred, y_test) * 100))


## LogisticRegression
# model = LogisticRegression(solver='lbfgs', max_iter=2000)
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# print('LogisticRegression: %.2f' %(metrics.accuracy_score(y_pred, y_test) * 100))

## RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# print('RandomForestClassifier: %.2f' %(metrics.accuracy_score(y_pred, y_test) * 100))


## Comput Feature Importance
# model.feature_importance_
# 모델을 예측하는데 있어 어떤 파라미터가 중요한지 계산한다.

features = pd.Series(
    model.feature_importances_,
    index=x_train.columns
).sort_values(ascending=False)

# print(features)

## Extract Top 5 Features
top_5_features = features.keys()[:5]

# print(top_5_features)

## SVM (Top 5)
model = svm.SVC(gamma='scale')
model.fit(x_train[top_5_features], y_train)

y_pred = model.predict(x_test[top_5_features])

# print('SVM(Top 5): %.2f' %(metrics.accuracy_score(y_pred, y_test) * 100))

## Cross Validation(Tedious)
# 데이터 셋이 적을때 효과적

model = svm.SVC(gamma='scale')

cv = KFold(n_splits=5, random_state=2019)

accs = []

for train_index, test_index in cv.split(data[top_5_features]):
    x_train = data.iloc[train_index][top_5_features]
    y_train = data.iloc[train_index].diagnosis

    x_test = data.iloc[test_index][top_5_features]
    y_test = data.iloc[test_index].diagnosis

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accs.append(metrics.accuracy_score(y_test, y_pred))

# print(accs)


## Cross Validation(Simple)
model = svm.SVC(gamma='scale')

cv = KFold(n_splits=5, random_state=2019)

accs = cross_val_score(model, data[top_5_features], data.diagnosis, cv=cv)

# print(accs)

## Test All Models
models = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
}

cv = KFold(n_splits=5, random_state=2019)

for name, model in models.items():
    scores = cross_val_score(model, data[top_5_features], data.diagnosis, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100))

## Normalize Dataset
# MinMaxScaler()
# 데이터를 표준화 한다. 딥러닝때 자주 썼던 cv2.normalize()와 같음.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[top_5_features])

models = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
}

cv = KFold(n_splits=5, random_state=2019)

for name, model in models.items():
    scores = cross_val_score(model, scaled_data, data.diagnosis, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100))
