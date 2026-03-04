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
K_ENSEMBLE = 5
EU_THRESHOLD = 0.05
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
        self.h_head = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        feat = self.fc(x)
        z = self.z_head(feat).view(-1, num_actions, N_ATOMS)
        probs = torch.softmax(z, dim=-1)
        h_scores = torch.sigmoid(self.h_head(feat))
        return probs, h_scores

def provide_external_feedback(state, action, predictor_model):
    with torch.no_grad():
        _, h_scores = predictor_model(torch.FloatTensor(state).unsqueeze(0))
        feedback = 1.0 if h_scores[0][action] > 0.5 else -1.0
    return feedback

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

def train_step(ensemble, h_net, optimizer, buffer, support, predictor, num_actions):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    total_z_loss = 0
    for net in ensemble:
        probs, _ = net(states)
        current_probs = probs[range(BATCH_SIZE), actions]
        with torch.no_grad():
            next_probs, _ = net(next_states)
            next_q = torch.sum(next_probs * support, dim=-1)
            best_actions = torch.argmax(next_q, dim=-1)
            target_probs = projection_step(next_probs[range(BATCH_SIZE), best_actions], rewards, dones, support)
        z_loss = -torch.sum(target_probs * torch.log(current_probs + 1e-8), dim=-1).mean()
        total_z_loss += z_loss
    _, h_scores = h_net(states)
    predicted_h = h_scores[range(BATCH_SIZE), actions]
    labels = torch.FloatTensor([provide_external_feedback(s, a, predictor) for s, a in zip(states, actions)])
    labels = (labels + 1) / 2
    h_loss = F.binary_cross_entropy(predicted_h, labels)
    optimizer.zero_grad()
    (total_z_loss + h_loss).backward()
    optimizer.step()

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <EnvID> <PredictorPath>")
        sys.exit(1)
    
    env_id = sys.argv[1]
    predictor_path = sys.argv[2]
    
    env = gym.make(env_id, repeat_action_probability=0.3)
    global num_actions
    num_actions = env.action_space.n
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS)
    
    ensemble = [C51Encoder(num_actions) for _ in range(K_ENSEMBLE)]
    h_net = C51Encoder(num_actions)
    
    predictor = C51Encoder(num_actions)
    if os.path.exists(predictor_path):
        predictor.load_state_dict(torch.load(predictor_path))
        predictor.eval()
    else:
        print(f"Error: Predictor file {predictor_path} not found.")
        sys.exit(1)

    optimizer = optim.Adam(list(h_net.parameters()) + [p for n in ensemble for p in n.parameters()], lr=LR)
    buffer = deque(maxlen=500000)
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            state_v = torch.FloatTensor(state).unsqueeze(0)
            q_vals = []
            for net in ensemble:
                z_probs, _ = net(state_v)
                q_vals.append(torch.sum(z_probs * support, dim=-1))
            q_stack = torch.stack(q_vals)
            eu = torch.var(q_stack, dim=0).mean()
            
            if eu > EU_THRESHOLD:
                _, h_scores = h_net(state_v)
                action = torch.argmax(h_scores).item()
            else:
                action = torch.argmax(torch.mean(q_stack, dim=0)).item()
                
            next_state, reward, done, _, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            
            if len(buffer) > BATCH_SIZE:
                train_step(ensemble, h_net, optimizer, buffer, support, predictor, num_actions)

if __name__ == "__main__":
    main()
