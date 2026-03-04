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
N_QUANTILES = 200
REPLAY_SIZE = 500000
EPISODES = 200 * 10**5

class QRDQNEncoder(nn.Module):
    def __init__(self, num_actions):
        super(QRDQNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions * N_QUANTILES)
        )

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x).view(-1, num_actions, N_QUANTILES)

def quantile_huber_loss(current_quantiles, target_quantiles):
    batch_size = current_quantiles.size(0)
    tau = torch.arange(0.5, N_QUANTILES, 1.0).to(current_quantiles.device) / N_QUANTILES
    
    diff = target_quantiles.unsqueeze(2) - current_quantiles.unsqueeze(1)
    loss = F.smooth_l1_loss(current_quantiles.unsqueeze(1), target_quantiles.unsqueeze(2), reduction='none')
    
    weight = torch.abs(tau - (diff < 0).float())
    loss = weight * loss
    return loss.sum(dim=1).mean(dim=1).mean()

def train_step(net, target_net, optimizer, buffer):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    current_quantiles = net(states)[range(BATCH_SIZE), actions]
    
    with torch.no_grad():
        next_quantiles = target_net(next_states)
        next_q = next_quantiles.mean(dim=2)
        best_actions = torch.argmax(next_q, dim=1)
        best_next_quantiles = next_quantiles[range(BATCH_SIZE), best_actions]
        target_quantiles = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * best_next_quantiles
        
    loss = quantile_huber_loss(current_quantiles, target_quantiles)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    
    env_id = sys.argv[1]
    env = gym.make(env_id, repeat_action_probability=0.3)
    num_actions = env.action_space.n
    
    net = QRDQNEncoder(num_actions)
    target_net = QRDQNEncoder(num_actions)
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
                with torch.no_grad():
                    quantiles = net(state_v)
                    q_values = quantiles.mean(dim=2)
                    action = torch.argmax(q_values).item()
            
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
