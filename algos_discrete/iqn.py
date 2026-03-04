import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import sys
from collections import deque

LR = 5e-5
GAMMA = 0.95
BATCH_SIZE = 32
N_SAMPLES = 64  # Number of tau samples during training
N_COSINE = 64   # Dimension of cosine embedding
REPLAY_SIZE = 500000
EPISODES = 200 * 10**5

class IQNEncoder(nn.Module):
    def __init__(self, num_actions):
        super(IQNEncoder, self).__init__()
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        
        self.phi = nn.Sequential(nn.Linear(N_COSINE, 512), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.cosine_weights = nn.Parameter(torch.arange(N_COSINE).float())

    def forward(self, x, tau):
        # x shape: (batch, 4, 84, 84)
        # tau shape: (batch, n_samples)
        batch_size = x.size(0)
        n_samples = tau.size(1)
        
        # State embedding
        state_feat = self.conv(x).view(batch_size, -1)
        state_feat = nn.Linear(64 * 7 * 7, 512).to(x.device)(state_feat) # Pre-fc layer
        
        # Cosine embedding of tau
        # cos_tau shape: (batch * n_samples, N_COSINE)
        cos_tau = torch.cos(tau.view(-1, 1) * self.cosine_weights * np.pi)
        phi_tau = self.phi(cos_tau).view(batch_size, n_samples, 512)
        
        # Combine embeddings via element-wise product (Eq. 4 in IQN paper)
        combined = state_feat.unsqueeze(1) * phi_tau
        quantiles = self.fc(combined) # (batch, n_samples, num_actions)
        return quantiles.transpose(1, 2) # (batch, num_actions, n_samples)

def quantile_huber_loss(current_quantiles, target_quantiles, tau):
    # current_quantiles: (batch, N_SAMPLES)
    # target_quantiles: (batch, N_SAMPLES)
    # tau: (batch, N_SAMPLES)
    
    diff = target_quantiles.unsqueeze(2) - current_quantiles.unsqueeze(1)
    loss = F.smooth_l1_loss(current_quantiles.unsqueeze(1), target_quantiles.unsqueeze(2), reduction='none')
    
    weight = torch.abs(tau.unsqueeze(1) - (diff < 0).float())
    loss = weight * loss
    return loss.sum(dim=1).mean(dim=1).mean()

def train_step(net, target_net, optimizer, buffer):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)
    
    # Sample tau for current and target
    tau = torch.rand(BATCH_SIZE, N_SAMPLES)
    next_tau = torch.rand(BATCH_SIZE, N_SAMPLES)
    
    current_quantiles = net(states, tau)[range(BATCH_SIZE), actions]
    
    with torch.no_grad():
        # Double DQN style action selection
        next_quantiles_eval = net(next_states, next_tau)
        next_q = next_quantiles_eval.mean(dim=2)
        best_actions = torch.argmax(next_q, dim=1)
        
        # Target quantiles from target network
        next_quantiles_target = target_net(next_states, next_tau)
        best_next_quantiles = next_quantiles_target[range(BATCH_SIZE), best_actions]
        target_quantiles = rewards + GAMMA * (1 - dones) * best_next_quantiles
        
    loss = quantile_huber_loss(current_quantiles, target_quantiles, tau)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    
    env_id = sys.argv[1]
    env = gym.make(env_id, repeat_action_probability=0.3)
    num_actions = env.action_space.n
    
    net = IQNEncoder(num_actions)
    target_net = IQNEncoder(num_actions)
    target_net.load_state_dict(net.state_dict())
    
    optimizer = optim.Adam(net.parameters(), lr=LR)
    buffer = deque(maxlen=REPLAY_SIZE)
    
    total_steps = 0
    update_target_steps = 10000
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            eps = max(0.01, 1.0 - total_steps / 10**6)
            if random.random() < eps:
                action = random.randint(0, num_actions - 1)
            else:
                state_v = torch.FloatTensor(state).unsqueeze(0)
                # For acting, sample more points for a better estimate of Q
                tau_eval = torch.rand(1, 64) 
                with torch.no_grad():
                    quantiles = net(state_v, tau_eval)
                    action = torch.argmax(quantiles.mean(dim=2)).item()
            
            next_state, reward, done, _, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_steps += 1
            
            if len(buffer) > BATCH_SIZE:
                train_step(net, target_net, optimizer, buffer)
            
            if total_steps % update_target_steps == 0:
                target_net.load_state_dict(net.state_dict())

if __name__ == "__main__":
    main()
