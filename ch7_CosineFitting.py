import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # OMP 중복 허용

# Model 클래스
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() # super() : 현재 클래스의 부모 클래스를 참조하는 객체를 반환(Model클래스가 상속받은 nn.Mdoel클래스의 생성자를 호출하여 해당 클래스의 속성과 메서드를 초기화하는 코드)
        self.fc1 = nn.Linear(1, 128) # hidden layer를 의미, (앞의 레이어의 노드, 뒤의 레이어의 노드)
        self.fc2 = nn.Linear(128, 128) # 128 * 128개의 w가 필요
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1, bias=False) # bias=False : 이 레이어는 편향 값을 사용하지 않는다.

    def forward(self, x):
        x = F.relu(self.fc1(x)) # 연산할 때 호출되는 함수(비선형 함수인 relu)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) # 활성화 함수를 사용하지 않고 선형 연산만 수행
        return x
    
def true_fun(X):
    noise = np.random.rand(X.shape[0]) * 0.4 - 0.2
    return np.cos(1.5 * np.pi * X) + X + noise

def plot_results(model):
    x = np.linspace(0, 5, 100)
    input_x = torch.from_numpy(x).float().unsqueeze(1) # x를 파이토치 텐서로 변환. float타입으로 변환. unsqueeze(1)는 텐서의 차원을 늘려서 새로운 차원을 추가한다. 새로운 차원이 첫 번째 차원으로 추가된다. --> 결과 (x, y)
    plt.plot(x, true_fun(x), label="Truth")
    plt.plot(x, model(input_x).detach().numpy(), label="Predicition") # detach().numpy()는 파이토치 텐서에서 연산을 멈추고, numpy 배열로 변환 -> 시각화를 위해
    plt.legend(loc='lower right', fontsize=15)
    plt.xlim((0,5))
    plt.ylim((-1,5))
    plt.grid()
    plt.show()

def main():
    data_x = np.random.rand(10000) * 5 # 0~5 사이 숫자 1만개를 샘플링하여 input으로 사용
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # model.parameters() : model의 학습 가능한 파라미터를 반환한다. 가중치(w)와 편향(b)과 같은 매개 변수들을 의미한다. lr(learning rate) : 한 번의 업데이트 양 조절

    for step in range(10000):
        batch_x = np.random.choice(data_x, 32) # 랜덤하게 뽑힌 32개의 데이터로 mini-batch를 구성

        batch_x_tensor = torch.from_numpy(batch_x).float().unsqueeze(1) # 파이토치의 텐서로 변환
        pred = model(batch_x_tensor)

        batch_y = true_fun(batch_x) # batch_x의 정답인 batch_y계산
        truth = torch.from_numpy(batch_y).float().unsqueeze(1)
        loss = F.mse_loss(pred, truth) # 손실 함수인 MSE를 계산하는 부분

        optimizer.zero_grad() # 모델의 gradient를 0으로 초기화
        loss.mean().backward() # 역전파를 통한 그라디언트 계산이 일어나는 부분
        optimizer.step() # 실제로 파라미터를 업데이트 하는 부분

    plot_results(model)

if __name__ == '__main__':
    main()