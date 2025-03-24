import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import datetime

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 공통 특징 추출 레이어
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 액터 네트워크 (정책) - 포지션 방향
        self.actor_direction = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # -1 ~ 1 범위로 제한
        )
        

        
        # 행동의 표준편차
        self.actor_direction_std = nn.Parameter(torch.zeros(1))
        
        # 크리틱 네트워크 (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        features = self.feature_extraction(state)
        
        # 액터: 행동 분포의 평균과 표준편차
        direction_mean = self.actor_direction(features)
        direction_std = torch.exp(self.actor_direction_std).expand_as(direction_mean)
        
        # 크리틱: 상태 가치
        value = self.critic(features)
        
        return direction_mean, direction_std, value

class PPO:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        model_name=None,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon=0.2,
        epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.feature_extraction.parameters()},
            {'params': self.actor_critic.actor_direction.parameters()},
            {'params': self.actor_critic.actor_direction_std},            
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic}
        ], lr=lr_actor)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.device = device
        self.model_name = model_name
        self.memory = deque()
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            direction_mean, direction_std, value = self.actor_critic(state)
            
        # 포지션 방향 샘플링
        direction_dist = Normal(direction_mean, direction_std)
        direction = direction_dist.sample()
        direction = torch.clamp(direction, -1.0, 1.0)
        
 
        
        # 로그 확률 계산
        log_prob= direction_dist.log_prob(direction)
        action = direction
        
        return (
            action.cpu().numpy()[0],
            value.cpu().numpy()[0],
            log_prob.cpu().numpy()[0]
        )
    
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def update(self, batch_size=64):
        # 메모리에서 데이터 추출
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        log_prob_batch = []
        value_batch = []
        done_batch = []
        
        for transition in self.memory:
            state, action, reward, next_state, log_prob, value, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            log_prob_batch.append(log_prob)
            value_batch.append(value)
            done_batch.append(done)
        
        # 텐서로 변환
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        old_log_prob_batch = torch.FloatTensor(np.array(log_prob_batch)).to(self.device)
        old_value_batch = torch.FloatTensor(np.array(value_batch)).to(self.device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # GAE 계산
        advantages = []
        returns = []
        gae = 0
        
        with torch.no_grad():
            next_value = self.actor_critic(next_state_batch)[2]  # value는 5번째 반환값
            next_value = next_value.squeeze()
            
            for r, v, done, next_v in zip(
                reversed(reward_batch),
                reversed(old_value_batch),
                reversed(done_batch),
                reversed(next_value)
            ):
                if done:
                    delta = r - v
                    gae = delta
                else:
                    delta = r + self.gamma * next_v - v
                    gae = delta + self.gamma * 0.95 * gae
                
                returns.insert(0, gae + v)
                advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 업데이트
        for _ in range(self.epochs):
            # 미니배치 생성
            indices = np.random.permutation(len(state_batch))
            for start_idx in range(0, len(state_batch), batch_size):
                idx = indices[start_idx:start_idx + batch_size]
                
                if len(idx) < batch_size:
                    break
                
                # 현재 미니배치
                state = state_batch[idx]
                action = action_batch[idx]
                advantage = advantages[idx]
                return_ = returns[idx]
                old_log_prob = old_log_prob_batch[idx]
                
                # 현재 정책의 행동 분포
                direction_mean, direction_std, value = self.actor_critic(state)
                
                # 방향과 거래량에 대한 분포
                direction_dist = Normal(direction_mean, direction_std)
                
                # 새로운 로그 확률 계산
                new_log_prob = direction_dist.log_prob(action[:, 0:1])
                
                #print(f"new_log_prob: {new_log_prob}")
                #print(f"old_log_prob: {old_log_prob}")
                # PPO 비율 계산
                ratio = torch.exp(new_log_prob - old_log_prob)
                #print( f"ratio: {ratio}")
                # 클리핑된 목적 함수
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 가치 함수 손실
                value = value.squeeze()
                critic_loss = nn.CrossEntropyLoss()(value, return_)
                
                # 전체 손실
                loss = actor_loss + 0.5 * critic_loss
                
                # 역전파 및 최적화
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        # 메모리 비우기
        self.memory.clear() 
    
    def save_model(self, data, path):
        torch.save(data, path)
    
    def checkpoint(self, data, path):
        torch.save(data, path)
    
    def load_model(self):
        self.actor_critic.load_state_dict(torch.load(f'futures_rl/models/{self.model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'))
    
    def load_checkpoint(self, path):
        return torch.load(path)