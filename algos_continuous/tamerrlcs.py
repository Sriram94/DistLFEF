import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv

LR = 3e-4
LR_C = 3e-2
GAMMA = 0.99
BATCH_SIZE = 256
TAU = 0.005
ALPHA = 0.2
STATE_DIM = 28
ACTION_DIM = 2
OMEGA_CS = 0.4  

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        return self.q_out(x)

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

class TAMER_H(nn.Module):
    """The H-Model predicts human evaluative reinforcement."""
    def __init__(self, state_dim, action_dim):
        super(TAMER_H, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.h_out = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        return self.h_out(x)

def train_tamer_rl_cs(actor, critics, tamer_h, target_critics, actor_opt, critic_opts, tamer_opt, buffer, predictor):
    batch = random.sample(buffer, BATCH_SIZE)
    s, a, r, s_, d = [torch.FloatTensor(np.array(x)) for x in zip(*batch)]
    
    with torch.no_grad():
        h_target = predictor(s, a) 
    h_val = tamer_h(s, a)
    h_loss = F.mse_loss(h_val, h_target)
    tamer_opt.zero_grad()
    h_loss.backward()
    tamer_opt.step()

    with torch.no_grad():
        a_next, log_p_next = actor(s_)
        q_target = torch.min(target_critics[0](s_, a_next), target_critics[1](s_, a_next)) - ALPHA * log_p_next
        y = r + (1 - d) * GAMMA * q_target

    for i in range(2):
        q_current = critics[i](s, a)
        rl_loss = F.mse_loss(q_current, y)
        cs_loss = F.mse_loss(q_current, tamer_h(s, a))
        total_critic_loss = (1 - OMEGA_CS) * rl_loss + OMEGA_CS * cs_loss
        
        critic_opts[i].zero_grad()
        total_critic_loss.backward()
        critic_opts[i].step()

    curr_a, log_p = actor(s)
    q_val = torch.min(critics[0](s, curr_a), critics[1](s, curr_a))
    actor_loss = (ALPHA * log_p - q_val).mean()
    
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    for i in range(2):
        for param, target_param in zip(critics[i].parameters(), target_critics[i].parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

def main():
    agent_id = "TAMER_RL_CS_Agent"
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Full, max_episode_steps=1000),
        observation_adapter=lambda obs: np.array([obs.ego_vehicle_state.speed, obs.ego_vehicle_state.steering] + [0.0]*24, dtype=np.float32),
        reward_adapter=lambda obs, reward: reward,
        action_adapter=lambda action: np.array([action[0], action[1]], dtype=np.float32),
    )

    env = HiWayEnv(scenarios=["scenarios/sumo/intersections/4lane"], agent_specs={agent_id: agent_spec}, headless=True)
    
    actor = Actor(STATE_DIM, ACTION_DIM)
    tamer_h = TAMER_H(STATE_DIM, ACTION_DIM)
    critics = [Critic(STATE_DIM, ACTION_DIM) for _ in range(2)]
    target_critics = [Critic(STATE_DIM, ACTION_DIM) for _ in range(2)]
    
    actor_opt = optim.Adam(actor.parameters(), lr=LR)
    tamer_opt = optim.Adam(tamer_h.parameters(), lr=LR)
    critic_opts = [optim.Adam(c.parameters(), lr=LR) for c in critics]
    buffer = deque(maxlen=1000000)

    predictor = lambda s, a: torch.zeros((s.size(0), 1))

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
                train_tamer_rl_cs(actor, critics, tamer_h, target_critics, actor_opt, critic_opts, tamer_opt, buffer, predictor)

    env.close()

if __name__ == "__main__":
    main()
