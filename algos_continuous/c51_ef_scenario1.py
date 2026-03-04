import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import gym
import sys

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv

V_MIN, V_MAX = -100.0, 100.0
N_ATOMS = 51
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
LR = 3e-4
LR_C = 3e-2
GAMMA = 0.99
BATCH_SIZE = 256
TAU = 0.005
ALPHA = 0.2
K_ENSEMBLE = 5
EU_THRESHOLD = 200
STATE_DIM = 28
ACTION_DIM = 2

class DistributionalCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DistributionalCritic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.z_head = nn.Linear(256, N_ATOMS)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        return torch.softmax(self.z_head(x), dim=-1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

class FeedbackHead(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(FeedbackHead, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return torch.tanh(self.action_out(x))

def get_aleatoric_uncertainty(ensemble, state, action, support):
    probs = torch.stack([net(state, action) for net in ensemble])
    avg_probs = torch.mean(probs, dim=0)
    expected_val = torch.sum(avg_probs * support, dim=-1, keepdim=True)
    au = torch.sum(avg_probs * (support - expected_val)**2, dim=-1)
    return au.mean().item()

def projection_step(next_probs, rewards, dones, support):
    batch_size = next_probs.size(0)
    Tz = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * support.unsqueeze(0)
    Tz = torch.clamp(Tz, V_MIN, V_MAX)
    b = (Tz - V_MIN) / DELTA_Z
    l, u = b.floor().long(), b.ceil().long()
    projected_probs = torch.zeros(batch_size, N_ATOMS).to(next_probs.device)
    for i in range(batch_size):
        projected_probs[i].index_add_(0, l[i], next_probs[i] * (u[i] - b[i]))
        projected_probs[i].index_add_(0, u[i], next_probs[i] * (b[i] - l[i]))
    return projected_probs

def train(actor, ensemble, h_head, target_ensemble, actor_opt, critic_opts, h_opt, buffer, support, predictor):
    batch = random.sample(buffer, BATCH_SIZE)
    s, a, r, s_, d = [torch.FloatTensor(np.array(x)) for x in zip(*batch)]
    
    with torch.no_grad():
        a_next, log_p_next = actor(s_)
        for i in range(K_ENSEMBLE):
            next_probs = target_ensemble[i](s_, a_next)
            target_probs = projection_step(next_probs, r, d, support)
            current_probs = ensemble[i](s, a)
            z_loss = -torch.sum(target_probs * torch.log(current_probs + 1e-8), dim=-1).mean()
            critic_opts[i].zero_grad()
            z_loss.backward()
            critic_opts[i].step()

    curr_a, log_p = actor(s)
    q_vals = torch.stack([torch.sum(net(s, curr_a) * support, dim=-1) for net in ensemble])
    min_q = torch.min(q_vals, dim=0)[0].unsqueeze(-1)
    actor_loss = (ALPHA * log_p - min_q).mean()
    
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    h_pred = h_head(s)
    with torch.no_grad():
        h_target = predictor(s) 
    h_loss = F.mse_loss(h_pred, h_target)
    h_opt.zero_grad()
    h_loss.backward()
    h_opt.step()

    for i in range(K_ENSEMBLE):
        for param, target_param in zip(ensemble[i].parameters(), target_ensemble[i].parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

def observation_adapter(env_observation):
    ego = env_observation.ego_vehicle_state
    obs = np.array([ego.speed, ego.steering, ego.heading, ego.lane_index] + [0.0]*24, dtype=np.float32)
    return obs

def main():
    agent_id = "C51_EF_Agent"
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Full, max_episode_steps=1000),
        observation_adapter=observation_adapter,
        reward_adapter=lambda obs, reward: reward,
        action_adapter=lambda action: np.array([action[0], action[1]], dtype=np.float32),
    )

    env = HiWayEnv(scenarios=["scenarios/sumo/intersections/4lane"], agent_specs={agent_id: agent_spec}, headless=True)
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS)
    
    actor = Actor(STATE_DIM, ACTION_DIM)
    h_head = FeedbackHead(STATE_DIM, ACTION_DIM)
    ensemble = [DistributionalCritic(STATE_DIM, ACTION_DIM) for _ in range(K_ENSEMBLE)]
    target_ensemble = [DistributionalCritic(STATE_DIM, ACTION_DIM) for _ in range(K_ENSEMBLE)]
    
    for i in range(K_ENSEMBLE):
        target_ensemble[i].load_state_dict(ensemble[i].state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=LR)
    critic_opts = [optim.Adam(net.parameters(), lr=LR) for net in ensemble]
    h_opt = optim.Adam(h_head.parameters(), lr=LR)
    buffer = deque(maxlen=1000000)

    predictor = lambda s: torch.zeros((s.size(0), ACTION_DIM)) 

    epsilon = 1.0

    for ep in range(500):
        observations = env.reset()
        state = observations[agent_id]
        done = False
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                actor_a, _ = actor(state_t)
                h_a = h_head(state_t)
                
                q_vals_ensemble = torch.stack([torch.sum(net(state_t, actor_a) * support, dim=-1) for net in ensemble])
                eu = torch.var(q_vals_ensemble, dim=0).mean().item()
                
                if eu <= EU_THRESHOLD:
                    greedy_action = actor_a.detach().cpu().numpy()[0]
                else:
                    greedy_action = h_a.detach().cpu().numpy()[0]

                au = get_aleatoric_uncertainty(ensemble, state_t, torch.FloatTensor(greedy_action).unsqueeze(0), support)

            delta = 1.0 / (1.0 + au)
            epsilon = max(0.01, epsilon - delta)

            if random.random() < epsilon:
                action = np.random.uniform(-1, 1, ACTION_DIM)
            else:
                action = greedy_action

            next_obs, rewards, dones, _ = env.step({agent_id: action})
            buffer.append((state, action, rewards[agent_id], next_obs[agent_id], dones[agent_id]))
            state = next_obs[agent_id]
            done = dones[agent_id]

            if len(buffer) > BATCH_SIZE:
                train(actor, ensemble, h_head, target_ensemble, actor_opt, critic_opts, h_opt, buffer, support, predictor)

    env.close()

if __name__ == "__main__":
    main()