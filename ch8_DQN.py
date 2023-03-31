import gym
import collections # 선입선출의 특성을 갖고 있는 리플레이 버퍼를 쉽게 구현
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 하이퍼 파라미터 정의
# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

# 리플레이 버퍼 클래스
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], [] 

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition # done_mask : 종료 상태의 밸류를 마스킹해주기 위해 만든 변수
            s_lst.append(s)
            a_lst.append([a]) # 스칼라 값이 아닌 벡터 값이 나올 수 있기 때문
            r_lst.append([r]) # 스칼라 값이 아닌 벡터 값이 나올 수 있기 때문
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask]) # 스칼라 값이 아닌 벡터 값이 나올 수 있기 때문

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)
    
# Q 밸류 네트워크 클래스
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2) # 액션이 2개이기 때문에 output의 차원이 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # 마지막 아웃풋은 Q밸류이기 때문에 [-무한대, 무한대] 사이 어느 값이든 취할 수 있다.
        return x 
    
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item() # Q값이 제일 큰 액션을 선택

# 학습 함수        
def train(q, q_target, memory, optimizer):
    for i in range(10): # 1번의 업데이트에 32개의 데이터가 사용됨, 한 에피소드가 끝날 때마다 버퍼에서 총 320개의 데이터를 뽑아서 사용함
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a) # 실제 선택된 액션의 q값을 의미
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) # q_target 네트워크는 정답지를 계산할 떄 쓰이는 네트워크로 학습 대상이 아니다.
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward() # loss에 대한 그라디언트 계산이 일어남
        optimizer.step() # Qnet의 파라미터의 업데이트가 일어남

# 메인 함수
def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer= optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08-0.01*(n_epi/200))
        # Linear annealing from 8% to 1%
        s = env.reset()
        #s = [num for sublist in s for num in sublist]
        s = np.array(s).astype(np.float32)
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon) 
            s_prime, r, done, _= env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break
        
        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    
    env.close()

if __name__ == '__main__':
    main()