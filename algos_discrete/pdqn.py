import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import sys
from collections import deque

LR = 1e-4
GAMMA = 0.95
BATCH_SIZE = 32
REPLAY_SIZE = 500000
EPISODES = 200 * 10**5
ALPHA = 0.6  # Prioritization exponent
BETA_START = 0.4  # Initial importance sampling weight
BETA_FRAMES = 10**6 # Steps to anneal beta to 1.0

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

class PrioritizedBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def append(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

def train_step(net, target_net, optimizer, buffer, beta):
    samples, indices, weights = buffer.sample(BATCH_SIZE, beta)
    states, actions, rewards, next_states, dones = zip(*samples)
    
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    weights = weights.to(states.device)

    current_q = net(states)[range(BATCH_SIZE), actions]
    
    with torch.no_grad():
        next_q = target_net(next_states)
        max_next_q = torch.max(next_q, dim=1)[0]
        target_q = rewards + (1 - dones) * GAMMA * max_next_q
    
    # Calculate TD-error for priority updates
    td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
    
    # Weighted MSE loss using IS weights
    loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    buffer.update_priorities(indices, td_errors + 1e-6)

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    
    env_id = sys.argv[1]
    env = gym.make(env_id, repeat_action_probability=0.3)
    num_actions = env.action_space.n
    
    net = DQNCore(num_actions)
    target_net = DQNCore(num_actions)
    target_net.load_state_dict(net.state_dict())
    
    optimizer = optim.Adam(net.parameters(), lr=LR)
    buffer = PrioritizedBuffer(REPLAY_SIZE, ALPHA)
    
    total_steps = 0
    update_target_steps = 1000
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            beta = min(1.0, BETA_START + total_steps * (1.0 - BETA_START) / BETA_FRAMES)
            eps = max(0.01, 1.0 - total_steps / 10**6)
            
            if random.random() < eps:
                action = random.randint(0, num_actions - 1)
            else:
                state_v = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = torch.argmax(net(state_v)).item()
            
            next_state, reward, done, _, _ = env.step(action)
            buffer.append(state, action, reward, next_state, done)
            state = next_state
            total_steps += 1
            
            if len(buffer.buffer) > BATCH_SIZE:
                train_step(net, target_net, optimizer, buffer, beta)
            
            if total_steps % update_target_steps == 0:
                target_net.load_state_dict(net.state_dict())

if __name__ == "__main__":
    main()
