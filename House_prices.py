import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.feature_selection import VarianceThreshold
from dummy_coding import dummy_df
from sklearn.preprocessing import LabelEncoder
from math import sqrt


data = pd.read_csv('Data/House Prices/train.csv')

data.info()

data.isnull().sum().sort_values(ascending = False)

data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'Id'] , axis=1)

cols_with_null = data.columns[data.isnull().any()].tolist()

numeric = [x for x in data.columns if data[x].dtypes != 'object']

category = [x for x in data.columns if data[x].dtypes == 'object']

number = LabelEncoder()
for column in category:
    data[column] = (number.fit_transform(data[column]))

#Imputing numerical data with mean
for column in numeric:
    data[column].fillna((data[column].mean()), inplace=True)


#co-relation between variables
cor_matrix = data.corr().abs()

cor_y =  cor_matrix[cor_matrix.index == 'SalePrice']

req_var = list(cor_y[cor_y >= 0.5].dropna(axis=1).columns)

req_var.remove('SalePrice')
len(req_var)

x = data.loc[:, req_var]
y = data['SalePrice']

x.shape
y.shape

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# from sklearn import linear_model
# reg = linear_model.LinearRegression()

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(max_depth=2, random_state=0)

mdl = reg.fit(x_train, y_train)

pred = reg.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

rmse = sqrt(mean_squared_error(y_test, pred))

r_square = r2_score(y_test, pred)


#--------predictions---------#

test = pd.read_csv('Data/House Prices/test.csv')
len(test)
test.isnull().sum().sort_values(ascending = False)

test = test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'] , axis=1)

category = [x for x in test.columns if test[x].dtypes == 'object']

for column in category:
    test[column] = (number.fit_transform(test[column]))

numeric = [x for x in test.columns if test[x].dtypes != 'object']

for column in numeric:
    test[column].fillna((test[column].mean()), inplace=True)


x_eval = test.loc[:, req_var]

y_eval = pd.DataFrame({'SalePrice' :mdl.predict(x_eval)})

len(y_eval)
len(test['Id'])

df_result = pd.concat([test['Id'], y_eval], axis=1)

df_final = pd.concat([test['Id'],df_result], axis=1)

filename = 'submission.csv'
try:
    os.remove(filename)
except OSError:
    pass

df_result.to_csv(filename, index=False)