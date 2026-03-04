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
W_RBRL = 0.4  # Weight for the RbRL feedback head

class RbRLNet(nn.Module):
    def __init__(self, num_actions):
        super(RbRLNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU())
        
        # Environmental Q-head
        self.q_env = nn.Linear(512, num_actions)
        # Human feedback Q-head (Reward-by-RL)
        self.q_human = nn.Linear(512, num_actions)

    def forward(self, x):
        feat = self.conv(x).view(x.size(0), -1)
        feat = self.fc(feat)
        return self.q_env(feat), self.q_human(feat)

def provide_external_feedback(state, action, predictor_model):
    with torch.no_grad():
        # Using the predictor as the human proxy for Scenario 1
        outputs = predictor_model(torch.FloatTensor(state).unsqueeze(0))
        # Logic: If predictor favors the action, feedback is +1, else -1
        feedback = 1.0 if torch.argmax(outputs).item() == action else -1.0
    return feedback

def train_step(net, target_net, optimizer, buffer, predictor):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    # Get current Q-values for both heads
    q_env, q_human = net(states)
    current_q_env = q_env[range(BATCH_SIZE), actions]
    current_q_human = q_human[range(BATCH_SIZE), actions]
    
    with torch.no_grad():
        # External feedback from the predictor
        h_feedback = torch.FloatTensor([
            provide_external_feedback(s, a, predictor) 
            for s, a in zip(states, actions)
        ])
        
        next_q_env, next_q_human = target_net(next_states)
        
        # Environmental Target
        max_next_q_env = torch.max(next_q_env, dim=1)[0]
        target_env = rewards + (1 - dones) * GAMMA * max_next_q_env
        
        # Human Feedback Target (RL update on the feedback)
        max_next_q_human = torch.max(next_q_human, dim=1)[0]
        target_human = h_feedback + (1 - dones) * GAMMA * max_next_q_human

    loss_env = F.mse_loss(current_q_env, target_env)
    loss_human = F.mse_loss(current_q_human, target_human)
    
    optimizer.zero_grad()
    (loss_env + loss_human).backward()
    optimizer.step()

def main():
    if len(sys.argv) < 3:
        print("Usage: python rbrl.py <EnvID> <PredictorPath>")
        sys.exit(1)
    
    env_id = sys.argv[1]
    predictor_path = sys.argv[2]
    
    env = gym.make(env_id, repeat_action_probability=0.3)
    num_actions = env.action_space.n
    
    net = RbRLNet(num_actions)
    target_net = RbRLNet(num_actions)
    target_net.load_state_dict(net.state_dict())
    
    # The predictor uses the same architecture but is pre-trained
    predictor = nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
        nn.Linear(512, num_actions)
    )
    
    if os.path.exists(predictor_path):
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()
    else:
        print(f"Predictor path {predictor_path} not found.")
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
                    q_env, q_human = net(state_v)
                    # RbRL combines the two heads for action selection
                    combined_q = (1 - W_RBRL) * q_env + W_RBRL * q_human
                    action = torch.argmax(combined_q).item()
            
            next_state, reward, done, _, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_steps += 1
            
            if len(buffer) > BATCH_SIZE:
                train_step(net, target_net, optimizer, buffer, predictor)
            
            if total_steps % update_target_steps == 0:
                target_net.load_state_dict(net.state_dict())

if __name__ == "__main__":
    main()
