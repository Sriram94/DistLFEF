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
N_ATOMS = 51
V_MIN, V_MAX = -10.0, 10.0
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
N_ENSEMBLE = 5  # Ensemble size for epistemic estimation
UNCERTAINTY_THRESHOLD = 0.08 
STATE_SHAPE = (4, 84, 84)

class Epistemic_C51_EF_Atari(nn.Module):
    def __init__(self, n_actions):
        super(Epistemic_C51_EF_Atari, self).__init__()
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
        
        # Multiple Distributional Heads
        self.dist_heads = nn.ModuleList([
            nn.Linear(512, n_actions * N_ATOMS) for _ in range(N_ENSEMBLE)
        ])
        
        self.h_head = nn.Linear(512, n_actions)

    def forward(self, state):
        feat = self.conv(state).view(state.size(0), -1)
        feat = self.fc(feat)
        
        all_probs = []
        all_q_values = []
        
        for head in self.dist_heads:
            dist = head(feat).view(-1, self.n_actions, N_ATOMS)
            probs = F.softmax(dist, dim=-1)
            q_vals = torch.sum(probs * self.support, dim=-1)
            all_probs.append(probs)
            all_q_values.append(q_vals)
            
        # ensemble_q: [N_ENSEMBLE, Batch, Action]
        ensemble_q = torch.stack(all_q_values)
        ensemble_probs = torch.stack(all_probs)
        
        h_logits = self.h_head(feat)
        
        return ensemble_probs, ensemble_q, h_logits

def get_epistemic_disagreement(ensemble_q):
    # Calculate variance across ensemble members to find epistemic uncertainty
    variance = torch.var(ensemble_q, dim=0) # [Batch, Action]
    return variance.mean(dim=-1) # Mean across actions for the gate

def train_epistemic_c51(model, target_model, optimizer, buffer, device):
    batch = random.sample(buffer, BATCH_SIZE)
    s, a, r, s_, d, h_feedback = zip(*batch)
    
    s = torch.FloatTensor(np.array(s)).to(device) / 255.0
    a = torch.LongTensor(a).to(device)
    r = torch.FloatTensor(r).to(device)
    s_ = torch.FloatTensor(np.array(s_)).to(device) / 255.0
    d = torch.FloatTensor(d).to(device)
    h_target = torch.FloatTensor(np.array(h_feedback)).to(device)

    # Train each ensemble head (Bootstrapping or Shared Feature)
    total_c51_loss = 0
    ensemble_probs, ensemble_q, h_logits = model(s)
    
    with torch.no_grad():
        target_probs, target_q, _ = target_model(s_)
        # Use ensemble mean for target action selection
        mean_q_next = target_q.mean(dim=0)
        a_next = torch.argmax(mean_q_next, dim=1)

        for i in range(N_ENSEMBLE):
            p_next = target_probs[i, range(BATCH_SIZE), a_next]
            tz = r.unsqueeze(1) + GAMMA * (1 - d).unsqueeze(1) * model.support.unsqueeze(0)
            tz = tz.clamp(V_MIN, V_MAX)
            b = (tz - V_MIN) / DELTA_Z
            l, u = b.floor().long(), b.ceil().long()
            
            t_dist = torch.zeros(BATCH_SIZE, N_ATOMS).to(device)
            for j in range(BATCH_SIZE):
                t_dist[j].index_add_(0, l[j], p_next[j] * (u[j].float() - b[j]))
                t_dist[j].index_add_(0, u[j], p_next[j] * (b[j] - l[j].float()))
            
            current_dist = ensemble_probs[i, range(BATCH_SIZE), a]
            total_c51_loss += -(t_dist * torch.log(current_dist + 1e-8)).sum(dim=-1).mean()

    h_loss = F.mse_loss(h_logits, h_target)
    loss = (total_c51_loss / N_ENSEMBLE) + h_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Soft update
    for p, tp in zip(model.parameters(), target_model.parameters()):
        tp.data.copy_(0.001 * p.data + (1 - 0.001) * tp.data)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    n_actions = env.action_space.n
    
    model = Epistemic_C51_EF_Atari(n_actions).to(device)
    target_model = Epistemic_C51_EF_Atari(n_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = deque(maxlen=100000)

    for ep in range(1000):
        state = env.reset()
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
            _, ensemble_q, h_logits = model(state_t)
            
            uncertainty = get_epistemic_disagreement(ensemble_q)
            
            if uncertainty > UNCERTAINTY_THRESHOLD:
                action = torch.argmax(h_logits, dim=1).item()
            else:
                mean_q = ensemble_q.mean(dim=0)
                action = torch.argmax(mean_q, dim=1).item()

            next_obs, reward, done, _ = env.step(action)
            h_signal = np.zeros(n_actions) # Placeholder for human signal
            buffer.append((state, action, reward, next_obs, done, h_signal))
            state = next_obs

            if len(buffer) > 1000:
                train_epistemic_c51(model, target_model, optimizer, buffer, device)

    env.close()

if __name__ == "__main__":
    main()
