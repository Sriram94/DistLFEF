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
REPLAY_SIZE = 500000
EPISODES = 200 * 10**5

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

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

def get_feedback_label(state, action, predictor):
    with torch.no_grad():
        h_scores = predictor(torch.FloatTensor(state).unsqueeze(0))
        return 1.0 if h_scores[0][action] > 0.5 else 0.0

def train_reward_model(reward_net, reward_opt, buffer, predictor):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, _, _, _ = zip(*batch)
    states = torch.FloatTensor(np.array(states))
    
    labels = torch.FloatTensor([get_feedback_label(s, a, predictor) for s, a in zip(states, actions)]).unsqueeze(1)
    
    reward_preds = torch.sigmoid(reward_net(states))
    loss = F.binary_cross_entropy(reward_preds, labels)
    
    reward_opt.zero_grad()
    loss.backward()
    reward_opt.step()

def train_rl_step(net, target_net, reward_net, optimizer, buffer):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, env_rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    with torch.no_grad():
        learned_rewards = reward_net(states).squeeze()
        total_rewards = torch.FloatTensor(env_rewards) + learned_rewards
        
        next_q = target_net(next_states)
        max_next_q = torch.max(next_q, dim=1)[0]
        target_q = total_rewards + (1 - dones) * GAMMA * max_next_q
        
    current_q = net(states)[range(BATCH_SIZE), actions]
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
    
    reward_net = RewardModel()
    reward_opt = optim.Adam(reward_net.parameters(), lr=LR)
    
    predictor = DQNCore(num_actions)
    if os.path.exists(predictor_path):
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()
    else:
        sys.exit(1)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    buffer = deque(maxlen=REPLAY_SIZE)
    
    total_steps = 0
    update_target_steps = 1000
    
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
                    action = torch.argmax(net(state_v)).item()
            
            next_state, reward, done, _, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_steps += 1
            
            if len(buffer) > BATCH_SIZE:
                train_reward_model(reward_net, reward_opt, buffer, predictor)
                train_rl_step(net, target_net, reward_net, optimizer, buffer)
            
            if total_steps % update_target_steps == 0:
                target_net.load_state_dict(net.state_dict())

if __name__ == "__main__":
    main()
