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

V_MIN, V_MAX = -10.0, 10.0
N_ATOMS = 51
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
LR = 1e-4
GAMMA = 0.95
BATCH_SIZE = 32
EPISODES = 200 * 10**5

class C51Encoder(nn.Module):
    def __init__(self, num_actions):
        super(C51Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU())
        self.z_head = nn.Linear(512, num_actions * N_ATOMS)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        feat = self.fc(x)
        z = self.z_head(feat).view(-1, num_actions, N_ATOMS)
        return torch.softmax(z, dim=-1)

def get_expert_action(state, predictor_model):
    with torch.no_grad():
        h_scores = predictor_model(torch.FloatTensor(state).unsqueeze(0))
        return torch.argmax(h_scores).item()

def projection_step(next_probs, rewards, dones, support):
    batch_size = next_probs.size(0)
    projected_probs = torch.zeros(batch_size, N_ATOMS)
    Tz = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * support.unsqueeze(0)
    Tz = torch.clamp(Tz, V_MIN, V_MAX)
    b = (Tz - V_MIN) / DELTA_Z
    l, u = b.floor().long(), b.ceil().long()
    for i in range(batch_size):
        for j in range(N_ATOMS):
            projected_probs[i, l[i, j]] += next_probs[i, j] * (u[i, j] - b[i, j])
            projected_probs[i, u[i, j]] += next_probs[i, j] * (b[i, j] - l[i, j])
    return projected_probs

def train_step(net, optimizer, buffer, support):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    probs = net(states)
    current_probs = probs[range(BATCH_SIZE), actions]
    
    with torch.no_grad():
        next_probs = net(next_states)
        next_q = torch.sum(next_probs * support, dim=-1)
        best_actions = torch.argmax(next_q, dim=-1)
        target_probs = projection_step(next_probs[range(BATCH_SIZE), best_actions], rewards, dones, support)
    
    loss = -torch.sum(target_probs * torch.log(current_probs + 1e-8), dim=-1).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class HNetworkPredictor(nn.Module):
    def __init__(self, num_actions):
        super(HNetworkPredictor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions))
    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=-1)

def main():
    if len(sys.argv) < 3:
        sys.exit(1)
    
    env_id = sys.argv[1]
    predictor_path = sys.argv[2]
    
    env = gym.make(env_id, repeat_action_probability=0.3)
    global num_actions
    num_actions = env.action_space.n
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS)
    
    net = C51Encoder(num_actions)
    predictor = HNetworkPredictor(num_actions)
    
    if os.path.exists(predictor_path):
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()
    else:
        sys.exit(1)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    buffer = deque(maxlen=500000)
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            state_v = torch.FloatTensor(state).unsqueeze(0)
            
            if random.random() < 0.2:
                action = get_expert_action(state, predictor)
            else:
                with torch.no_grad():
                    z_probs = net(state_v)
                    q_values = torch.sum(z_probs * support, dim=-1)
                    action = torch.argmax(q_values).item()
                
            next_state, reward, done, _, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            
            if len(buffer) > BATCH_SIZE:
                train_step(net, optimizer, buffer, support)

if __name__ == "__main__":
    main()
