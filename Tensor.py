import torch
import numpy as np

list_data = [[10, 20], [30, 40]]

# 파이썬 리스트를 직접 텐서로 만들 수 있다.
tensor1 = torch.Tensor(list_data)

print(tensor1)
"""
[] 1차원 배열 (백터)
[[]] 2차원 배열 (행렬)
[[[]]] 3차원 배열 (텐서)
tensor([[10., 20.],
        [30., 40.]])
"""
print(f"tensor type: {type(tensor1)}, tensor shape: {tensor1.shape}")
print(f"tensor dtype: {tensor1.dtype}, tensor device: {tensor1.device}")

# GPU 사용
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

tensor1 = tensor1.to(device)

print(f"tensor type: {type(tensor1)}, tensor shape: {tensor1.shape}")
print(f"tensor dtype: {tensor1.dtype}, tensor device: {tensor1.device}")


numpy_data = np.array(list_data)
tensor2 = torch.from_numpy(numpy_data) # numpy 자료형을 텐서로 변경하면 기본적으로 정수형 타입을 리턴

print(tensor2)
print(f"tensor type: {type(tensor2)}, tensor shape: {tensor2.shape}")
print(f"tensor dtype: {tensor2.dtype}, tensor device: {tensor2.device}")


tensor2_1 = torch.from_numpy(numpy_data).float() # 딥러닝에서는 기본적으로 실수형을 사용하기 때문에 float으로 변경해줘야한다

print(tensor2_1)
print(f"tensor type: {type(tensor2_1)}, tensor shape: {tensor2_1.shape}")
print(f"tensor dtype: {tensor2_1.dtype}, tensor device: {tensor2_1.device}")

"""
머신러닝이나 딥러닝에서 정수형보다 실수형을 사용하는 이유

1. 역전파
- 학습 중 가중치를 업데이트하려면 미분이 필요한데 미분값은 소수점이 나온다.
- 정수형은 소수점을 표현 못해서 미분을 저장할 수 없다.

2. 활성화 함수
- sigmoid, tanh, ReLU 등의 출력값이 0.0~1.0 같은 실수다
- 정수로 표현하면 모두 0 또는 1로 잘려버려 의미가 없어진다

3. 작은 변화를 표현햐야 한다.
- 학습률이 보통 0.001, 0.0001 수준이다.
- 가중치 업데이트도 아주 미세하게 일어나는데, 정수형은 이를 표현할 수 없다

딥러닝의 핵심인 "조금씩 조금씩 가중치를 수정" 하는 과정 자체가 실수 연산이기 때문
"""

tensor3 = torch.rand(2, 2)
print(tensor3)

tensor4 = torch.randn(2, 2)
print(tensor4)

tensor5 = torch.randn(2, 2)
print(tensor5)

numpy_from_tensor = tensor5.numpy()
print(numpy_from_tensor)


tensor6 = torch.Tensor([[1, 2, 3], [4, 5, 6]])

tensor7 = torch.Tensor([[7, 8, 9], [10, 11, 12]])
print(tensor6[0])
print(tensor6[:, 1:])
print(tensor7[0:2, 0:-1])
print(tensor7[-1, -1])
print(tensor7[... , -2])

tensor8 = tensor6.mul(tensor7)  # tensor8 = tensor6 * tensor7

print(tensor8)
# tensor9 = tensor6.matmul(tensor7)

tensor7.view(3, 2)

tensor9 = tensor6.matmul(tensor7.view(3, 2))  # tensor6 @ tensor7.view(3, 2)

print(tensor9)

tensor_cat = torch.cat([tensor6, tensor7]) # 열을 기준으로 합친다는 의미

print(tensor_cat)
"""
tensor([[1., 2., 3.],
        [4., 5., 6.]])
tensor([[ 7.,  8.,  9.],
        [10., 11., 12.]])
        
tensor_cat 
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]])
"""

tensor_cat_dim0 = torch.cat([tensor6, tensor7], dim=0) # dim=0 이면 기본값으로 열을 기준으로 합친다는 의미

print(tensor_cat_dim0)

"""
tensor([[1., 2., 3.],
        [4., 5., 6.]])
tensor([[ 7.,  8.,  9.],
        [10., 11., 12.]])

tensor_cat_dim0
tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]])
"""

tensor_cat_dim1 = torch.cat([tensor6, tensor7], dim=1) # dim=1 행을 기준으로 합친다는 의미

print(tensor_cat_dim1)

"""
tensor([[1., 2., 3.],
        [4., 5., 6.]])
tensor([[ 7.,  8.,  9.],
        [10., 11., 12.]])

tensor_cat_dim1
tensor([[ 1.,  2.,  3.,  7.,  8.,  9.],
        [ 4.,  5.,  6., 10., 11., 12.]])
"""
