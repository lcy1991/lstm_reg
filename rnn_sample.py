# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from visdom import Visdom
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#定义超参数
input_size=1
hidden_size=30
output_size=1
lr=0.005
#导入数据
num_time_steps=50
#定义模型
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.rnn=nn.RNN(
            input_size=input_size,hidden_size=hidden_size,
            num_layers=1,batch_first=True,
        )
        self.linear=nn.Linear(hidden_size,output_size)
    def forward(self,x,hidden_prev):
     out,hidden_prev=self.rnn(x,hidden_prev)
     out=out.view(-1,hidden_size)
     out=self.linear(out)
     out=out.unsqueeze(dim=0)
     return out,hidden_prev

#训练
model=Net()
critericn=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr)
hidden_prev=torch.zeros(1,1,hidden_size)#新建h0
for iter in range(6000):
    start=np.random.randint(10,size=1)[0]
    time_steps=np.linspace(start,start+10,num_time_steps)
    data=np.sin(time_steps)
    data=data.reshape(num_time_steps,1)
    x=torch.tensor(data[:-1]).float().view(1,num_time_steps-1,1)
    y=torch.tensor(data[1:]).float().view(1,num_time_steps-1,1)
    output,hidden_prev=model(x,hidden_prev)
    hidden_prev=hidden_prev.detach()

    loss=critericn(output,y)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    if iter%10==0:
        print('iteration:{}loss{}'.format(iter,loss.item()))

start=np.random.randint(3,size=1)[0]
time_steps=np.linspace(start,start+10,num_time_steps)
data=np.sin(time_steps)
data=data.reshape(num_time_steps,1)
x=torch.tensor(data[:-1]).float().view(1,num_time_steps-1,1)
y=torch.tensor(data[1:]).float().view(1,num_time_steps-1,1)
#预测
predictions=[]
input=x[:,0,:]
for _ in range(x.shape[1]):
    input=input.view(1,1,1)
    (pred,hidden_prev)=model(input,hidden_prev)
    input=pred
    predictions.append(pred.detach().numpy().ravel()[0])
x=x.data.numpy().ravel()
y=y.data.numpy()
plt.scatter(time_steps[:-1],x.ravel(),s=90)
plt.plot(time_steps[:-1],x.ravel())

plt.scatter(time_steps[1:],predictions)
plt.show()