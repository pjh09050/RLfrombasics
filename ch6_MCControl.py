### 업그레이드 그리드 월드 ### 
# xy 0  1  2  3  4  5  6 #
# 0       #             #
# 1       #             #
# 2 S     #     #       #
# 3             #       #
# 4             #     G #
#############################
# 1. S에서 출발해서 G에 도착하면 끝
# 2. 회색 영역(벽)은 지나갈 수 없는 벽이 놓여 있는 곳
# 3. 보상은 스텝마다 -1 ( 즉 최단 거리로 G에 도달하는 것이 목적 ) 

# 수렴할 때까지 N번 반복
# -> 한 에피소드의 경험을 쌓고 -> 경험한 데이터로 q(s,a) 테이블의 값을 업데이트(정책 평가) -> 업데이트된 q(s,a) 테이블을 이용하여 입실론 그리디 정책을 만듬(정책 개선) -> 계속 반복

import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0

    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_up()
        elif a == 2:
            self.move_right()
        else:
            self.move_down()

        reward = -1 # 보상은 항상 1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done
    
    def move_left(self):
        if self.y == 0:
            pass
        elif self.y == 3 and self.x in [0, 1, 2]:
            pass
        elif self.y == 5 and self.x in [2, 3, 4]:
            pass
        else:
            self.y -= 1
    
    def move_right(self):
        if self.y == 1 and self.x in [0, 1, 2]:
            pass
        elif self.y == 3 and self.x in [2, 3, 4]:
            pass
        elif self.y == 6:
            pass
        else:
            self.y += 1
    
    def move_up(self):
        if self.x == 0:
            pass
        elif self.x == 3 and self.y == 2:
            pass
        else:
            self.x -= 1

    def move_down(self):
        if self.x == 4:
            pass
        elif self.x == 1 and self.y == 4:
            pass
        else:
            self.x += 1
    
    def is_done(self):
        if self.x == 4 and self.y == 6:
            return True
        else:
            return False
    
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)
    
class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4)) # q밸류를 저장하는 변수. 모두 0으로 초기화
        self.eps = 0.9
        self.alpha = 0.01

    def select_action(self, s):
        # eps-greedy로 액션을 선택해준다.
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,3)
        else:
            action_val = self.q_table[x,y,:]
            action = np.argmax(action_val)
        return action
    
    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x, y = s
            # 몬테카를로 방식을 이용하여 업데이트
            self.q_table[x, y, a] = self.q_table[x, y, a] + self.alpha * (cum_reward - self.q_table[x, y, a])
            cum_reward = cum_reward + r
    
    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max (self.eps, 0.1)

    def show_table(self):
        # 학습이 각 위치에서 어느 액션의 q 값이 가능 높았는지 보여주는 함수
        q_lst = self.q_table.tolist()
        data = np.zeros((5,7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)

def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000):
        done = False
        history = []

        s = env.reset()
        while not done: # 한 에피소드가 끝날 때까지
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s, a, r, s_prime))
            s = s_prime
        agent.update_table(history) # 히스토리를 이용하여 에이전트 업데이트
        agent.anneal_eps()

    agent.show_table() # 학습이 끝난 결과 출력

if __name__ == "__main__":
    main()