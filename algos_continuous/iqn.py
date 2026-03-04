import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import sys

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv

LR = 3e-4
LR_C = 3e-2
GAMMA = 0.99
BATCH_SIZE = 32
TAU_SAMPLES = 8  
K_SAMPLES = 8    
N_PHI = 64       
STATE_DIM = 28
ACTION_DIM = 2

class IQNCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(IQNCritic, self).__init__()
        self.phi = nn.Linear(1, N_PHI)
        self.fc_state = nn.Linear(state_dim + action_dim, 256)
        self.fc_joint = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, state, action, tau):
        batch_size = state.size(0)
        num_samples = tau.size(1)
        
        sa = torch.cat([state, action], dim=1)
        x_state = F.relu(self.fc_state(sa)) 
        
        i_pi = np.pi * torch.arange(1, N_PHI + 1).to(state.device)
        phi = torch.cos(tau.unsqueeze(-1) * i_pi).view(batch_size * num_samples, N_PHI)
        phi = F.relu(self.phi(phi)).view(batch_size, num_samples, 256)
        
        x_state = x_state.unsqueeze(1) 
        x_joint = F.relu(self.fc_joint(x_state * phi)) 
        
        quantiles = self.fc_out(x_joint).squeeze(-1) 
        return quantiles

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
        dist = Normal(mu, torch.exp(log_std))
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

def huber_quantile_loss(current_quantiles, target_quantiles, taus):
    
    diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2) 
    loss = F.smooth_l1_loss(current_quantiles.unsqueeze(2), target_quantiles.unsqueeze(1), reduction='none')
    
    weight = torch.abs(taus.unsqueeze(2) - (diff < 0).float())
    return (weight * loss).mean()

def train_iqn(actor, critic, target_critic, actor_opt, critic_opt, buffer):
    batch = random.sample(buffer, BATCH_SIZE)
    s, a, r, s_, d = [torch.FloatTensor(np.array(x)) for x in zip(*batch)]

    with torch.no_grad():
        a_next, _ = actor(s_)
        tau_next = torch.rand(BATCH_SIZE, K_SAMPLES).to(s.device)
        target_quantiles = r.unsqueeze(1) + GAMMA * (1 - d).unsqueeze(1) * target_critic(s_, a_next, tau_next)
    
    taus = torch.rand(BATCH_SIZE, TAU_SAMPLES).to(s.device)
    current_quantiles = critic(s, a, taus)
    
    critic_loss = huber_quantile_loss(current_quantiles, target_quantiles, taus)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    curr_a, _ = actor(s)
    tau_actor = torch.rand(BATCH_SIZE, TAU_SAMPLES).to(s.device)
    q_val = critic(s, curr_a, tau_actor).mean(dim=1, keepdim=True)
    
    actor_loss = -q_val.mean() 
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

def main():
    agent_id = "IQN_Baseline_Agent"
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Full, max_episode_steps=1000),
        observation_adapter=lambda obs: np.array([obs.ego_vehicle_state.speed, obs.ego_vehicle_state.steering] + [0.0]*26, dtype=np.float32),
        reward_adapter=lambda obs, reward: reward,
        action_adapter=lambda action: np.array([action[0], action[1]], dtype=np.float32),
    )

    env = HiWayEnv(scenarios=["scenarios/sumo/intersections/4lane"], agent_specs={agent_id: agent_spec}, headless=True)
    
    critic = IQNCritic(STATE_DIM, ACTION_DIM)
    target_critic = IQNCritic(STATE_DIM, ACTION_DIM)
    actor = Actor(STATE_DIM, ACTION_DIM)
    
    target_critic.load_state_dict(critic.state_dict())
    critic_opt = optim.Adam(critic.parameters(), lr=LR)
    actor_opt = optim.Adam(actor.parameters(), lr=LR)
    buffer = deque(maxlen=1000000)

    for ep in range(500):
        observations = env.reset()
        state = observations[agent_id]
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, _ = actor(state_t)
            action_np = action.detach().cpu().numpy()[0]

            next_obs, rewards, dones, _ = env.step({agent_id: action_np})
            buffer.append((state, action_np, rewards[agent_id], next_obs[agent_id], dones[agent_id]))
            state = next_obs[agent_id]
            done = dones[agent_id]

            if len(buffer) > BATCH_SIZE:
                train_iqn(actor, critic, target_critic, actor_opt, critic_opt, buffer)

    env.close()

if __name__ == "__main__":
    main()
