import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import sys
import os
from collections import deque

LR = 1e-4
GAMMA = 0.95
BATCH_SIZE = 32
EPISODES = 200 * 10**5
LAMBDA_BIAS = 2.0 

class DQNCore(nn.Module):
    def __init__(self, num_actions):
        super(DQNCore, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

class PredictorWrapper(nn.Module):
    def __init__(self, num_actions):
        super(PredictorWrapper, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions))
    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

def train_step(net, target_net, optimizer, buffer):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    current_q = net(states)[range(BATCH_SIZE), actions]
    
    with torch.no_grad():
        next_q = target_net(next_states)
        max_next_q = torch.max(next_q, dim=1)[0]
        target_q = rewards + (1 - dones) * GAMMA * max_next_q
        
    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    if len(sys.argv) < 3:
        sys.exit(1)
    
    env_id = sys.argv[1]
    predictor_path = sys.argv[2]
    
    env = gym.make(env_id, repeat_action_probability=0.3)
    num_actions = env.action_space.n
    
    net = DQNCore(num_actions)
    target_net = DQNCore(num_actions)
    target_net.load_state_dict(net.state_dict())
    
    predictor = PredictorWrapper(num_actions)
    if os.path.exists(predictor_path):
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()
    else:
        sys.exit(1)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    buffer = deque(maxlen=500000)
    
    total_steps = 0
    update_target_steps = 1000
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            state_v = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                q_values = net(state_v)
                h_scores = predictor(state_v)
                
                # TAMER+RL(AB): Biasing the action selection
                biased_q = q_values + LAMBDA_BIAS * h_scores
                action = torch.argmax(biased_q).item()
            
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
