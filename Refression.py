import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

"""
데이터 정의
"""
x_train = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6,1)
y_train = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6,1)

dataset = TensorDataset(x_train, y_train)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


"""
신경망 모델 구축
"""

class MyNeuralNetwork(nn.Module): # 신경망 모델 클래스

    def __init__(self): # 신경망 모델을 구성하는 계층 정의
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1,1)
        )

    def forward(self, x): # 피드포워드를 수행하고 결과값을 리턴함
        logits = self.linear_relu_stack(x)
        return logits

model = MyNeuralNetwork()

loss_function = nn.MSELoss() # 손실함수
"""
일반적인 회귀 데이터에스는 nn.MSELoss(),
분류 데이터에는 nn.CrossEntropyLoss()를 사용함
"""

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2) # 옵티마이저

"""
옵티마이저도 확룰적 경사하강법 이외에도 ADAM, RMSProp등의 다양한 옵티마이저가 있다.
"""

nums_epoch = 2000

for epoch in range(nums_epoch+1):

    prediction = model(x_train)
    """
    model에 데이터를 전달하면 model 클래스의 forward() 함수가 자동으로 실행된다
    """
    loss = loss_function(prediction, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('epoch = ', epoch, ' current loss = ', loss.item())

x_test = torch.Tensor([-3.1, 3.0, 1.2, -2.5]).view(4,1)

pred = model(x_test)

print(pred)