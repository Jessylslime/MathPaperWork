import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tqdm import tqdm

from matplotlib import pyplot as plt

np.random.seed(42)

class CustomDataset(Dataset): #Класс для создание dataset из x_, y_ для data_loader
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return feature, label

class SimpleNet(torch.nn.Module):  #Модель сети
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(9, 50)
        self.fc_1 = nn.Linear(50, 50) #1 слой 0.84
        #self.fc_2 = nn.Linear(100, 50) #2 слоя 0.89
        self.bn = nn.BatchNorm1d(50) #3 слоя > 0.9
        self.fc_out = nn.Linear(50, 1)

    def forward(self, x):
        x = F.tanh(self.fc_in(x))
        x = self.bn(x)
        x = F.tanh(self.fc_1(x))
        #x = F.relu(self.fc_2(x))
        x = self.bn(x)
        x = self.fc_out(x)
        return x


#предобработка данных
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
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
# фулл переобучение при test_size = 0.5, shuffle = false, n_epoch = 100
train_data = CustomDataset(x_train.astype(np.float32), y_train)
test_data = CustomDataset(x_test.astype(np.float32), y_test)
train_size = int(len(train_data) * 0.8)
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


def evaluate(model, dataloader, loss_fn):
    y_pred_list = []
    y_true_list = []
    losses = []
    for i, batch in enumerate(dataloader):
        X_batch, y_batch = batch
        with torch.no_grad():
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss = loss.item()
            losses.append(loss)
            y_pred = logits.numpy().flatten()
        y_pred_list.extend(y_pred)
        y_true_list.extend(y_batch.numpy().flatten())
    accuracy = r2_score(y_pred_list, y_true_list)

    return accuracy, np.mean(losses)

def train(model, loss_fn, optimizer, n_epoch=6):
    model.train(True)
    data = {
        'acc_train': [],
        'loss_train': [],
        'acc_val': [],
        'loss_val': []
    }
    for epoch in range(n_epoch):

        for i, batch in enumerate(train_loader):
            X_batch, y_batch = batch
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            optimizer.zero_grad() # обнуляем значения градиентов оптимизаторв
            loss.backward() # backpropagation (вычисление градиентов)
            optimizer.step() # обновление весов сети

            #print('On epoch end', epoch)
        acc_train_epoch, loss_train_epoch = evaluate(model, train_loader, loss_fn)
            #print('Train acc:', acc_train_epoch, 'Train loss:', loss_train_epoch)
        acc_val_epoch, loss_val_epoch = evaluate(model, val_loader, loss_fn)
            #print('Val acc:', acc_val_epoch, 'Val loss:', loss_val_epoch)
        data['acc_train'].append(acc_train_epoch)
        data['loss_train'].append(loss_train_epoch)
        data['acc_val'].append(acc_val_epoch)
        data['loss_val'].append(loss_val_epoch)

    return model, data



model = SimpleNet()
loss_fn = nn.MSELoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_epoch = 10

model, data_ = train(model, loss_fn, optimizer, n_epoch=n_epoch)

print(evaluate(model, train_loader, loss_fn))
print(evaluate(model, test_loader, loss_fn))

_, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

ox = list(range(n_epoch))

axes[0][0].plot(ox, data_['acc_train'])
axes[0][0].title.set_text('acc_train')

axes[0][1].plot(ox, data_['acc_val'])
axes[0][1].title.set_text('acc_val')

axes[1][0].plot(ox, data_['loss_train'])
axes[1][0].title.set_text('loss_train')

axes[1][1].plot(ox, data_['loss_val'])
axes[1][1].title.set_text('loss_val')

plt.savefig('C:/games/course_paper/img/SimpleNN_15_tanh.png')