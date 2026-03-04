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
REWARD_MU = 100.0
REWARD_SIGMA = 20.0
FEEDBACK_ACCURACY = 0.95 
FEEDBACK_BUDGET = 5000
EPISODES = 20000000

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
        self.h_head = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        feat = self.fc(x)
        z = self.z_head(feat).view(-1, num_actions, N_ATOMS)
        probs = torch.softmax(z, dim=-1)
        h_scores = torch.sigmoid(self.h_head(feat))
        return probs, h_scores

def get_noisy_reward(reward):
    return reward + np.random.normal(REWARD_MU, REWARD_SIGMA)

def provide_near_optimal_feedback(state, action, predictor_model):
    with torch.no_grad():
        _, h_scores = predictor_model(torch.FloatTensor(state).unsqueeze(0))
        optimal_opinion = 1.0 if h_scores[0][action] > 0.5 else -1.0
        if random.random() > FEEDBACK_ACCURACY:
            return -optimal_opinion
        return optimal_opinion

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

def train_step(net, optimizer, buffer, support, predictor, budget_tracker):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    noisy_rewards = torch.FloatTensor([get_noisy_reward(r) for r in rewards])
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    probs, h_scores = net(states)
    current_probs = probs[range(BATCH_SIZE), actions]
    with torch.no_grad():
        next_probs, _ = net(next_states)
        next_q = torch.sum(next_probs * support, dim=-1)
        target_probs = projection_step(next_probs[range(BATCH_SIZE), torch.argmax(next_q, dim=-1)], 
                                       noisy_rewards, dones, support)
    
    z_loss = -torch.sum(target_probs * torch.log(current_probs + 1e-8), dim=-1).mean()

    h_loss = 0
    if budget_tracker['remaining'] > 0:
        predicted_h = h_scores[range(BATCH_SIZE), actions]
        labels_list = []
        for s, a in zip(states, actions):
            if budget_tracker['remaining'] > 0:
                labels_list.append(provide_near_optimal_feedback(s, a, predictor))
                budget_tracker['remaining'] -= 1
            else:
                labels_list.append(0.0)
        labels = (torch.FloatTensor(labels_list) + 1) / 2
        h_loss = F.binary_cross_entropy(predicted_h, labels)

    optimizer.zero_grad()
    (z_loss + h_loss).backward()
    optimizer.step()

def main():
    env_id, predictor_path = sys.argv[1], sys.argv[2]
    env = gym.make(env_id, repeat_action_probability=0.3)
    num_actions = env.action_space.n
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS)
    
    net = C51Encoder(num_actions)
    predictor = C51Encoder(num_actions)
    predictor.load_state_dict(torch.load(predictor_path))
    predictor.eval()
    
    optimizer = optim.Adam(net.parameters(), lr=LR)
    buffer = deque(maxlen=500000)
    budget_tracker = {'remaining': FEEDBACK_BUDGET}
    
    total_rewards = 0
    steps_count = 0
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            state_v = torch.FloatTensor(state).unsqueeze(0)
            avg_reward = total_rewards / max(1, steps_count)
            alpha = np.clip(avg_reward / 200.0, 0.1, 0.9)
            
            with torch.no_grad():
                z_probs, h_scores = net(state_v)
                q_values = torch.sum(z_probs * support, dim=-1)
                combined_score = (1 - alpha) * q_values + alpha * h_scores
                action = torch.argmax(combined_score).item()
                
            next_state, reward, done, _, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            
            total_rewards += reward
            steps_count += 1
            state = next_state
            
            if len(buffer) > BATCH_SIZE:
                train_step(net, optimizer, buffer, support, predictor, budget_tracker)

if __name__ == "__main__":
    main()
