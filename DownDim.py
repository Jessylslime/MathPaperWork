import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

from tqdm import tqdm

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
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
acc = []

for subspaces_dim in tqdm(range(1,9)):
  acc_ = []
  for max_depth in range(20, 33):
    pca = PCA(n_components=subspaces_dim)
    X_tr = pca.fit_transform(X_train)
    x_ts = pca.transform(x_test)
    dt = DecisionTreeRegressor(max_depth=max_depth, criterion="squared_error", random_state=42)
    dt.fit(X_tr, y_train)
    preds = dt.predict(x_ts)
    ans = r2_score(y_test, preds)
    acc_.append([ans, max_depth])
  acc_ = np.array(acc_)
  ans, max_depth = acc_[np.argmax(acc_[:, 0])]
  acc.append([ans, max_depth, subspaces_dim])
          
#acc = np.array(acc)
print(acc)