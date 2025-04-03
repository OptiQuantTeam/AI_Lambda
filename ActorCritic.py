import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 공통 특징 추출 레이어
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            
        )
        
        # 액터 네트워크 (정책) - 포지션 방향
        self.actor_direction = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
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