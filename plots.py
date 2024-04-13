import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("TRAIN.csv", delimiter=',')
data.drop(data.columns[0], axis = 1, inplace=True)

le = LabelEncoder()
cut = le.fit_transform(np.array(data["cut"].tolist()))
data["cut"] = data["cut"].replace(list(le.classes_), set(cut))
color = le.fit_transform(np.array(data["color"].tolist()))
data["color"] = data["color"].replace(list(le.classes_), set(color))
clarity = le.fit_transform(np.array(data["clarity"].tolist()))
data["clarity"] = data["clarity"].replace(list(le.classes_), set(clarity))

Y = data["price"].to_numpy()
data.drop(data.columns[6], axis = 1, inplace=True)
names = data.columns.to_list()

print(names)
for nameX in names:
        X = data[nameX].to_numpy()
        ax = plt.subplot()
        ax.scatter(X, Y)
        ax.set_title(f'Зависимость price от {nameX}')
        plt.show()
