import pandas as pd
import os
import sklearn
import numpy as np
from sklearn import svm
from dummy_coding import dummy_df

df_raw = pd.read_csv("Data/Titanic/train.csv")

df_raw = df_raw.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

df_raw.isnull().sum().sort_values(ascending = False)

#imputing nulls
#df_raw['Cabin'].fillna('U', inplace=True)

df_raw['Age'].fillna(df_raw['Age'].mean(), inplace=True)

df_raw['Embarked'].fillna(df_raw['Embarked'].value_counts().index[0], inplace=True)


#adding family and dropping the source columns
df_raw['Family'] = np.where((df_raw['SibSp']>0) | (df_raw['Parch']>0), '1', '0')
df_raw = df_raw.drop(['SibSp','Parch'],axis=1)

df_raw.info()

df_raw.hist(column="Survived")

df_raw.groupby(by=['Survived'])['Survived'].count()

cat_to_num = [x for x in df_raw.columns if df_raw[x].dtypes=='object']

df_raw = dummy_df(df_raw,cat_to_num)

df_raw.corr()['Survived']

x = df_raw.loc[:,df_raw.columns != 'Survived']
y = df_raw.loc[:,df_raw.columns == 'Survived']

from sklearn.cross_validation import train_test_split

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2)


from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)

#bagging = AdaBoostClassifier(n_estimators=120000, learning_rate=0.0001, max_depth=1)

# Create and fit an AdaBoosted decision tree
bagging = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

#bagging.

bagging.fit(x_train,y_train)

pred_bag = bagging.predict(x_test)

from sklearn.metrics import accuracy_score
score_bag = accuracy_score(y_test, pred_bag)


#predictions

df_test = pd.read_csv("Data/Titanic/test.csv")

df_test.info()

df_test = df_test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)

df_test['Embarked'].fillna(df_test['Embarked'].value_counts().index[0], inplace=True)


df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)
df_test.isnull().sum().sort_values(ascending = False)

df_test['Family'] = np.where((df_test['SibSp']>0) | (df_test['Parch']>0), '1', '0')
df_test = df_test.drop(['SibSp','Parch'],axis=1)


cat_to_num = [x for x in df_test.columns if df_test[x].dtypes=='object']

df_test = dummy_df(df_test,cat_to_num)


pred_x = df_test.loc[:, df_test.columns != 'Survived' ]
pred_y = df_test.loc[:, df_test.columns == 'Survived' ]

pred_test_set = bagging.predict(pred_x)

from sklearn.metrics import accuracy_score
score_bag_test = accuracy_score(pred_y, pred_test_set)
print score_bag_test


df_ref = pd.read_csv("Data/Titanic/test.csv")

result = pd.DataFrame({'PassengerId':df_ref['PassengerId'],'Survived':pred_test_set})

result.to_csv("Data/Titanic/result_1.csv",index=False)