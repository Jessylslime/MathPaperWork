import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

np.random.seed(42)

class sample(object):
  def __init__(self, X, n_subspace):
    self.idx_subspace = self.random_subspace(X, n_subspace)
  
  def __call__(self, X, y):
    idx_obj = self.bootstrap_sample(X)
    X_sampled, y_sampled = self.get_subsample(X, y, self.idx_subspace, idx_obj)
    return X_sampled, y_sampled

  @staticmethod
  def bootstrap_sample(X, random_state=42):
    return np.unique(np.random.choice(X.shape[0]-1, X.shape[0]-1))
  @staticmethod
  def random_subspace(X, n_subspace, random_state=42):
    return np.random.choice(X.shape[1]-1, n_subspace, replace = False)

  @staticmethod
  def get_subsample(X, y, idx_subspace, idx_obj):
    y_sampled = y[idx_obj]
    X_ = X[idx_obj]
    X_sampled = X_.take(idx_subspace, axis = 1)
    return X_sampled, y_sampled

N_ESTIMATORS = 1
MAX_DEPTH = 16
SUBSPACE_DIM = 3

class random_forest(object):
  def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.subspaces_dim = subspaces_dim
    self.random_state = random_state
    self._estimators = np.array([])

  def fit(self, X, y):
    for i in range(self.n_estimators):
      s = sample(X, n_subspace=self.subspaces_dim)
      idx_subspace = s.random_subspace(X, n_subspace=self.subspaces_dim)
      idx_obj = s.bootstrap_sample(X)
      X_sampled, y_sampled = s.get_subsample(X, y, idx_subspace, idx_obj)
      dt = DecisionTreeClassifier(criterion="gini", max_depth= self.max_depth, random_state=self.random_state)
      dt.fit(X_sampled, y_sampled)
      self._estimators = np.append(self._estimators, dt)

  def predict(self, X):
    s = sample(X, n_subspace=self.subspaces_dim)
    idx_subspace = s.random_subspace(X, n_subspace=self.subspaces_dim)
    X_sampled = X.take(idx_subspace, axis = 1)
    #print(X_sampled)
    preds = []
    for dt in self._estimators:
      preds.append(dt.predict(X_sampled))
    preds = np.array(preds)
    preds = np.transpose(preds)
    return np.mean(preds, axis = 1).astype(int)


data = pd.read_csv("TRAIN.csv", delimiter=',')
data.drop(data.columns[0], axis = 1, inplace=True)

le = LabelEncoder()
cut = le.fit_transform(np.array(data["cut"].tolist()))
data["cut"] = data["cut"].replace(list(le.classes_), set(cut))
color = le.fit_transform(np.array(data["color"].tolist()))
data["color"] = data["color"].replace(list(le.classes_), set(color))
clarity = le.fit_transform(np.array(data["clarity"].tolist()))
data["clarity"] = data["clarity"].replace(list(le.classes_), set(clarity))

y = data["color"].to_numpy()
data.drop(data.columns[2], axis = 1, inplace=True)
print(data)
X = data.to_numpy()
X, y = shuffle(X, y, random_state=42)

acc = []
params = []

for n_estimators in tqdm(range(1, 5)):
    for max_depth in range(1, 50):
        for subspaces_dim in range(6, 9):
            X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
            rf = random_forest(n_estimators, max_depth, subspaces_dim, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(x_test)
            ans = accuracy_score(y_test, preds)
            #print(ans)
            params.append([n_estimators, max_depth, subspaces_dim])
            acc.append(ans)
print(max(acc), acc.index(max(acc)), params[acc.index(max(acc))])
