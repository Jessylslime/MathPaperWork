import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

np.random.seed(42)

data = pd.read_csv("TRAIN.csv", delimiter=',')
data.drop(data.columns[0], axis = 1, inplace=True)

le = LabelEncoder()
cut = le.fit_transform(np.array(data["cut"].tolist()))
data["cut"] = data["cut"].replace(list(le.classes_), set(cut))
color = le.fit_transform(np.array(data["color"].tolist()))
data["color"] = data["color"].replace(list(le.classes_), set(color))
clarity = le.fit_transform(np.array(data["clarity"].tolist()))
data["clarity"] = data["clarity"].replace(list(le.classes_), set(clarity))

y = data["price"].to_numpy()
data.drop(data.columns[6], axis = 1, inplace=True)
X = data.to_numpy()
X, y = shuffle(X, y, random_state=42)

reg1 = LinearRegression()
score1 = cross_val_score(reg1, X, y, cv=10, scoring='r2').mean()

reg2 = LinearRegression()
score2 = cross_val_score(reg2, X, y, cv=10, scoring='r2').mean()

reg3 = LinearRegression()
score3 = cross_val_score(reg3, X, y, cv=10, scoring='r2').mean()

reg4 = DecisionTreeRegressor(criterion="squared_error", max_depth=1, random_state=42)
score4 = cross_val_score(reg4, X, y, cv=10, scoring='r2').mean()

reg5 = DecisionTreeRegressor(criterion="squared_error", max_depth=150, random_state=42)
score5 = cross_val_score(reg5, X, y, cv=10, scoring='r2').mean()

reg6 = DecisionTreeRegressor(criterion="poisson", max_depth=33, random_state=42)
score6 = cross_val_score(reg6, X, y, cv=10, scoring='r2').mean()

print(score1)
print(score2)
print(score3)
print(score4)
print(score5)
print(score6)
