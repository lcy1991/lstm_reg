import numpy as np
import matplotlib.pyplot as plt

data_x= np.loadtxt('./G_walk.txt', dtype=float, skiprows=13, usecols=(2), unpack=True)
data_len = len(data_x)
#plt.plot(range(1, data_len, 1), data_x[1:data_len]) 
#plt.show()
data_x = data_x.astype('float32')

def create_dataset(dataset, look_back=1000, look_forward=250, slid_window_size=250):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back - look_forward ,slid_window_size):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back : i + look_back + look_forward])
    return np.array(dataX), np.array(dataY)

dataset_x, dataset_y = create_dataset(data_x)
#print(dataset_x[0])
#print(dataset_y[0], len(dataset_y[0]))
#print(len(dataset_x), len(dataset_y))

train_size = int(len(dataset_x) * 0.7)
test_size = len(dataset_x) - train_size
train_X = dataset_x[:train_size]
train_Y = dataset_y[:train_size]
test_X = dataset_x[train_size:]
test_Y = dataset_y[train_size:]

import torch

print(np.shape(train_X))
train_X = train_X.reshape(-1, 1, 1000)
print(np.shape(train_X))
train_Y = train_Y.reshape(-1, 1, 250)
test_X = test_X.reshape(-1, 1, 1000)
print(np.shape(test_Y))
#test_Y = test_Y.reshape(-1, 1)
test_Y = test_Y.flatten()
print(np.shape(test_Y))
train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
print(train_x.shape)
from torch import nn
from torch.autograd import Variable
# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=250, num_layers=2):
        super(lstm_reg, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.reg = nn.Linear(hidden_size, output_size) # 回归
        
    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x
net = lstm_reg(1000, 20)
net.cuda() 
print(net)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
# 开始训练
for e in range(2000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    var_x = var_x.cuda()
    var_y = var_y.cuda()
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #if (e + 1) % 100 == 0: # 每 100 次输出结果
    #    print('Epoch: {}, Loss: {:.10f}'.format(e + 1, loss.data[0]))
    if e%10==0:
        print('iteration:{}loss{}'.format(e,loss.item()))

net = net.eval() # 转换成测试模式
dataset_x = dataset_x.reshape(-1, 1, 1000)
dataset_x = torch.from_numpy(dataset_x)
var_data = Variable(test_x)
var_data = var_data.cuda()
pred_test = net(var_data) # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.cpu()
pred_test = pred_test.view(-1).data.numpy()
# 画出实际结果和预测的结果
print(np.shape(pred_test))
plt.plot(pred_test, 'r', label='prediction')
plt.plot(test_Y, 'b', label='test')

plt.legend(loc='best')
plt.show()
