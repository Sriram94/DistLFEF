import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import sys
from collections import deque

V_MIN, V_MAX = -10.0, 10.0
N_ATOMS = 51
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
LR = 1e-4
GAMMA = 0.95
BATCH_SIZE = 32
K_ENSEMBLE = 5
EU_THRESHOLD = 20
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

def get_aleatoric_uncertainty(ensemble, state, action, support):
    state_v = torch.FloatTensor(state).unsqueeze(0)
    probs = torch.stack([net(state_v)[0][0, action] for net in ensemble])
    avg_probs = torch.mean(probs, dim=0)
    expected_val = torch.sum(avg_probs * support, dim=-1, keepdim=True)
    au = torch.sum(avg_probs * (support - expected_val)**2, dim=-1)
    return au.item()

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

def train_step(ensemble, h_net, optimizer, buffer, support, predictor, budget_tracker):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states_t = torch.FloatTensor(np.array(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    next_states_t = torch.FloatTensor(next_states)
    dones_t = torch.FloatTensor(dones)

    total_z_loss = 0
    for net in ensemble:
        probs, _ = net(states_t)
        current_probs = probs[range(BATCH_SIZE), actions_t]
        with torch.no_grad():
            next_probs, _ = net(next_states_t)
            next_q = torch.sum(next_probs * support, dim=-1)
            target_probs = projection_step(next_probs[range(BATCH_SIZE), torch.argmax(next_q, dim=-1)], 
                                           rewards_t, dones_t, support)
        total_z_loss += -torch.sum(target_probs * torch.log(current_probs + 1e-8), dim=-1).mean()

    h_loss = 0
    if budget_tracker['remaining'] > 0:
        _, h_scores = h_net(states_t)
        labels_list = []
        for s, a in zip(states, actions):
            if budget_tracker['remaining'] > 0:
                labels_list.append(provide_near_optimal_feedback(s, a, predictor))
                budget_tracker['remaining'] -= 1
            else:
                labels_list.append(0.0)
        labels = (torch.FloatTensor(labels_list) + 1) / 2
        h_loss = F.binary_cross_entropy(h_scores[range(BATCH_SIZE), actions_t], labels)

    optimizer.zero_grad()
    (total_z_loss + h_loss).backward()
    optimizer.step()

def main():
    if len(sys.argv) < 3:
        return
    env_id, predictor_path = sys.argv[1], sys.argv[2]
    env = gym.make(env_id, repeat_action_probability=0.3)
    num_actions = env.action_space.n
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS)
    
    ensemble = [C51Encoder(num_actions) for _ in range(K_ENSEMBLE)]
    h_net = C51Encoder(num_actions)
    predictor = C51Encoder(num_actions)
    predictor.load_state_dict(torch.load(predictor_path))
    predictor.eval()
    
    optimizer = optim.Adam(list(h_net.parameters()) + [p for n in ensemble for p in n.parameters()], lr=LR)
    buffer = deque(maxlen=500000)
    budget_tracker = {'remaining': FEEDBACK_BUDGET}
    
    epsilon = 1.0

    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        while not done:
            state_v = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                probs_h, h_scores_all = h_net(state_v)
                q_vals_ensemble = torch.stack([torch.sum(net(state_v)[0] * support, dim=-1) for net in ensemble])
                eu = torch.var(q_vals_ensemble, dim=0).mean().item()
                
                if eu <= EU_THRESHOLD:
                    greedy_action = torch.argmax(torch.mean(q_vals_ensemble, dim=0)).item()
                else:
                    greedy_action = torch.argmax(h_scores_all).item()

                au = get_aleatoric_uncertainty(ensemble, state, greedy_action, support)

            delta = 1.0 / (1.0 + au)
            epsilon = max(0.01, epsilon - delta)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action
                
            next_state, reward, done, _, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            
            if len(buffer) > BATCH_SIZE:
                train_step(ensemble, h_net, optimizer, buffer, support, predictor, budget_tracker)

if __name__ == "__main__":
    main()
