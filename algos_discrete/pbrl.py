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

class PreferenceRewardModel(nn.Module):
    def __init__(self):
        super(PreferenceRewardModel, self).__init__()
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

def train_preference_model(reward_net, reward_opt, buffer, predictor):
    # Sample two segments (trajectories) for comparison
    seg_1 = random.sample(buffer, BATCH_SIZE)
    seg_2 = random.sample(buffer, BATCH_SIZE)
    
    states_1, actions_1, _, _, _ = zip(*seg_1)
    states_2, actions_2, _, _, _ = zip(*seg_2)
    
    s1_tensor = torch.FloatTensor(np.array(states_1))
    s2_tensor = torch.FloatTensor(np.array(states_2))
    
    # Calculate preference labels based on predictor's confidence in actions
    with torch.no_grad():
        pred_out_1 = predictor(s1_tensor)
        pred_out_2 = predictor(s2_tensor)
        
        # Scenario 1 Proxy: Predictor favors trajectories where it agrees with actions
        score_1 = pred_out_1[range(BATCH_SIZE), actions_1].sum()
        score_2 = pred_out_2[range(BATCH_SIZE), actions_2].sum()
        
        # label = 1 if seg_1 is preferred, 0.5 for tie, 0 if seg_2 is preferred
        label = torch.tensor([1.0 if score_1 > score_2 else 0.0]).to(s1_tensor.device)

    # Bradley-Terry Preference Model
    r1 = reward_net(s1_tensor).sum()
    r2 = reward_net(s2_tensor).sum()
    
    prob_1_preferred = torch.exp(r1) / (torch.exp(r1) + torch.exp(r2))
    loss = F.binary_cross_entropy(prob_1_preferred.unsqueeze(0), label)
    
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
        # Combine environmental reward with learned preference reward
        p_reward = reward_net(states).squeeze()
        total_reward = torch.FloatTensor(env_rewards) + p_reward
        
        next_q = target_net(next_states)
        max_next_q = torch.max(next_q, dim=1)[0]
        target_q = total_reward + (1 - dones) * GAMMA * max_next_q
        
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
    
    reward_net = PreferenceRewardModel()
    reward_opt = optim.Adam(reward_net.parameters(), lr=1e-4)
    
    predictor = DQNCore(num_actions) # Standard architecture for predictor
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
            
            if len(buffer) > BATCH_SIZE * 2:
                train_preference_model(reward_net, reward_opt, buffer, predictor)
                train_rl_step(net, target_net, reward_net, optimizer, buffer)
            
            if total_steps % update_target_steps == 0:
                target_net.load_state_dict(net.state_dict())

if __name__ == "__main__":
    main()
