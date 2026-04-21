import torch
import matplotlib.pyplot as plt

x = torch.tensor([150, 160, 170, 175, 185]) #키
y = torch.tensor([55, 70, 64, 80, 75.]) #몸무게
N = len(x)
plt.scatter(x, y)
plt.show()

# 초기값 설정
a = 0.45
b = -35
"""
경사하강법은 현재 값에서 조금씩 개선하는 방식
시작점이 없으면 첫 번째 예측 자체가 불가능하기때문에 초기값을 설정해줘야한다.
  현재 a, b로 예측
      ↓                                                                                                                                                                                                                                  
  얼마나 틀렸는지 계산 (loss)
      ↓                                                                                                                                                                                                                                  
  기울기(grad) 계산
      ↓                                                                                                                                                                                                                                  
  a, b 업데이트   
      ↓                                                                                                                                                                                                                                  
  반복...
"""
x_plot = torch.linspace(145, 190, 100)
y_plot = a * x_plot + b

plt.plot(x, y, 'o')
plt.plot(x_plot, y_plot, 'r')

plt.show()

# a,b 를 바꿔가면서 loss 값을 일일히 구해서 가장 작아지게 하는 a, b를 선정

a = 0.5 + torch.linspace(-0.2, 0.2, 100)
b = -30 + torch.linspace(-20, 20, 100)

LOSS =  torch.zeros(len(b), len(a))
print(LOSS)
for i in range(len(b)):
    for j in range(len(a)):
        for n in range(N):
            LOSS[i, j] = LOSS[i, j] + (y[n] - (a[j] * x[n]+b[i])) ** 2

LOSS = LOSS / N # 평균제곱오차 MSE

print(torch.min(LOSS))
A, B = torch.meshgrid(a, b, indexing='ij')

a_opt = A[LOSS == torch.min(LOSS)]
b_opt = B[LOSS == torch.min(LOSS)]

print(a_opt)
print(b_opt)

x_plot = torch.linspace(145, 190, 100)
y_plot = a_opt * x_plot + b_opt

plt.plot(x,y, 'o')
plt.plot(x_plot, y_plot, 'r')

plt.show()