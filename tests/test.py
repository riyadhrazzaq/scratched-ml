"""
Test module
"""
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from linear_regression import LinearRegress
from sklearn.model_selection import train_test_split

subprocess.run("pwd".split(), check=True)

print("Loading data..")
df = pd.read_csv("data/housing.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train,  X_test, y_train,y_test = train_test_split(X, y)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
lr = 0.0001
model = LinearRegress(learning_rate=lr,max_iteration=1000,batch_size=100)
model.fit(X_train, y_train, verbose=True)
print(model.score(X_test,y_test))

sk = SGDRegressor(verbose=1,learning_rate='constant',eta0=lr)
sk.fit(X_train,y_train)
print(model.score(X_test,y_test))

