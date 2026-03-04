import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import sys

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv

V_MIN, V_MAX = -100.0, 100.0
N_ATOMS = 51
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
LR = 3e-4
GAMMA = 0.99
BATCH_SIZE = 256
TAU = 0.005
ALPHA = 0.2
STATE_DIM = 28
ACTION_DIM = 2
FIXED_BETA = 0.5

REWARD_MU = 100.0
REWARD_SIGMA = 20.0
FEEDBACK_ACCURACY = 0.95
FEEDBACK_BUDGET = 5000

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

def get_noisy_reward(reward):
    return reward + np.random.normal(REWARD_MU, REWARD_SIGMA)

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

def train(actor, critic, h_head, target_critic, actor_opt, critic_opt, h_opt, buffer, support, predictor, budget):
    batch = random.sample(buffer, BATCH_SIZE)
    s, a, r, s_, d = [torch.FloatTensor(np.array(x)) for x in zip(*batch)]
    noisy_r = torch.FloatTensor([get_noisy_reward(val.item()) for val in r]).to(s.device)
    
    with torch.no_grad():
        a_next, _ = actor(s_)
        next_probs = target_critic(s_, a_next)
        target_probs = projection_step(next_probs, noisy_r, d, support)
        
    current_probs = critic(s, a)
    z_loss = -torch.sum(target_probs * torch.log(current_probs + 1e-8), dim=-1).mean()
    critic_opt.zero_grad()
    z_loss.backward()
    critic_opt.step()

    curr_a, log_p = actor(s)
    q_val = torch.sum(critic(s, curr_a) * support, dim=-1).unsqueeze(-1)
    actor_loss = (ALPHA * log_p - q_val).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    if budget['remaining'] > 0:
        h_pred = h_head(s)
        with torch.no_grad():
            h_target = predictor(s)
            mask = (torch.rand(h_target.shape[0], 1) > (1 - FEEDBACK_ACCURACY)).float()
            h_target = h_target * mask + (h_target * -1) * (1 - mask)
        
        h_loss = F.mse_loss(h_pred, h_target)
        h_opt.zero_grad()
        h_loss.backward()
        h_opt.step()
        budget['remaining'] -= BATCH_SIZE

    for param, target_param in zip(critic.parameters(), target_critic.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

def main():
    agent_id = "Ablated_S2_Agent"
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Full, max_episode_steps=1000),
        observation_adapter=lambda obs: np.array([obs.ego_vehicle_state.speed, obs.ego_vehicle_state.steering, obs.ego_vehicle_state.heading, obs.ego_vehicle_state.lane_index] + [0.0]*24, dtype=np.float32),
        reward_adapter=lambda obs, reward: reward,
        action_adapter=lambda action: np.array([action[0], action[1]], dtype=np.float32),
    )

    env = HiWayEnv(scenarios=["scenarios/sumo/intersections/4lane"], agent_specs={agent_id: agent_spec}, headless=True)
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS)
    
    actor = Actor(STATE_DIM, ACTION_DIM)
    h_head = FeedbackHead(STATE_DIM, ACTION_DIM)
    critic = DistributionalCritic(STATE_DIM, ACTION_DIM)
    target_critic = DistributionalCritic(STATE_DIM, ACTION_DIM)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=LR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR)
    h_opt = optim.Adam(h_head.parameters(), lr=LR)
    buffer = deque(maxlen=1000000)
    budget = {'remaining': FEEDBACK_BUDGET}
    predictor = lambda s: torch.zeros((s.size(0), ACTION_DIM)) 

    for ep in range(500):
        observations = env.reset()
        state = observations[agent_id]
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                a_actor, _ = actor(state_t)
                a_feedback = h_head(state_t)
                
                action_t = (1 - FIXED_BETA) * a_actor + FIXED_BETA * a_feedback
                action = action_t.detach().cpu().numpy()[0]

            next_obs, rewards, dones, _ = env.step({agent_id: action})
            buffer.append((state, action, rewards[agent_id], next_obs[agent_id], dones[agent_id]))
            state = next_obs[agent_id]
            done = dones[agent_id]

            if len(buffer) > BATCH_SIZE:
                train(actor, critic, h_head, target_critic, actor_opt, critic_opt, h_opt, buffer, support, predictor, budget)
    env.close()

if __name__ == "__main__":
    main()
