import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gym

LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32
V_MIN, V_MAX = -10.0, 10.0
N_ATOMS = 51
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
TAU = 0.001
UNCERTAINTY_THRESHOLD = 0.2
FEEDBACK_FREQUENCY = 0.2 # p=0.2 means feedback is only available 20% of the time

class C51_EF_Atari_Freq(nn.Module):
    def __init__(self, n_actions):
        super(C51_EF_Atari_Freq, self).__init__()
        self.n_actions = n_actions
        self.register_buffer("support", torch.linspace(V_MIN, V_MAX, N_ATOMS))
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )
        
        self.dist_head = nn.Linear(512, n_actions * N_ATOMS)
        self.h_head = nn.Linear(512, n_actions)

    def forward(self, state):
        feat = self.conv(state).view(state.size(0), -1)
        feat = self.fc(feat)
        dist = self.dist_head(feat).view(-1, self.n_actions, N_ATOMS)
        probs = F.softmax(dist, dim=-1)
        q_values = torch.sum(probs * self.support, dim=-1)
        h_logits = self.h_head(feat)
        return probs, q_values, h_logits

def get_epistemic_uncertainty(probs):
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy.mean(dim=-1)

def train_c51_ef_freq(model, target_model, optimizer, buffer, device):
    batch = random.sample(buffer, BATCH_SIZE)
    # feedback_mask: 1 if human signal was available, 0 otherwise
    s, a, r, s_, d, h_target_val, feedback_mask = zip(*batch)
    
    s = torch.FloatTensor(np.array(s)).to(device) / 255.0
    a = torch.LongTensor(a).to(device)
    r = torch.FloatTensor(r).to(device)
    s_ = torch.FloatTensor(np.array(s_)).to(device) / 255.0
    d = torch.FloatTensor(d).to(device)
    h_target = torch.FloatTensor(np.array(h_target_val)).to(device)
    mask = torch.FloatTensor(np.array(feedback_mask)).to(device).unsqueeze(1)

    with torch.no_grad():
        probs_next, q_next, _ = target_model(s_)
        a_next = torch.argmax(q_next, dim=1)
        p_next = probs_next[range(BATCH_SIZE), a_next]
        tz = r.unsqueeze(1) + GAMMA * (1 - d).unsqueeze(1) * model.support.unsqueeze(0)
        tz = tz.clamp(V_MIN, V_MAX)
        b = (tz - V_MIN) / DELTA_Z
        l, u = b.floor().long(), b.ceil().long()
        
        target_dist = torch.zeros(BATCH_SIZE, N_ATOMS).to(device)
        for i in range(BATCH_SIZE):
            target_dist[i].index_add_(0, l[i], p_next[i] * (u[i].float() - b[i]))
            target_dist[i].index_add_(0, u[i], p_next[i] * (b[i] - l[i].float()))

    probs, _, h_logits = model(s)
    current_dist = probs[range(BATCH_SIZE), a]
    c51_loss = -(target_dist * torch.log(current_dist + 1e-8)).sum(dim=-1).mean()
    
    # Masked Feedback Loss: Only backpropagate where human signal was available
    h_loss = (F.mse_loss(h_logits, h_target, reduction='none') * mask).mean()
    
    total_loss = c51_loss + h_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    for p, tp in zip(model.parameters(), target_model.parameters()):
        tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("BreakoutNoFrameskip-v4")
    n_actions = env.action_space.n
    model = C51_EF_Atari_Freq(n_actions).to(device)
    target_model = C51_EF_Atari_Freq(n_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = deque(maxlen=100000)

    for ep in range(1000):
        state = env.reset()
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
            probs, q_vals, h_logits = model(state_t)
            uncertainty = get_epistemic_uncertainty(probs)
            
            # Gating Logic
            if uncertainty > UNCERTAINTY_THRESHOLD:
                action = torch.argmax(h_logits, dim=1).item()
            else:
                action = torch.argmax(q_vals, dim=1).item()

            next_obs, reward, done, _ = env.step(action)
            
            # SIMULATED VARYING FREQUENCY
            has_feedback = 1 if random.random() < FEEDBACK_FREQUENCY else 0
            h_signal = np.zeros(n_actions) # Placeholder for actual oracle signal
            
            buffer.append((state, action, reward, next_obs, done, h_signal, has_feedback))
            state = next_obs

            if len(buffer) > 1000:
                train_c51_ef_freq(model, target_model, optimizer, buffer, device)
    env.close()

if __name__ == "__main__":
    main()
