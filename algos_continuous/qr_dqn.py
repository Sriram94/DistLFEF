import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec
from smarts.env.hiway_env import HiWayEnv

LR = 3e-4
LR_C = 3e-2
GAMMA = 0.99
BATCH_SIZE = 64
N_QUANTILES = 51  
TAU = 0.005
STATE_DIM = 28
ACTION_DIM = 2  

class QuantileNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QuantileNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.q_head = nn.Linear(256, N_QUANTILES)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        return self.q_head(x) 

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
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        return action

def quantile_huber_loss(current_quantiles, target_quantiles, cumulative_probabilities):
    
    diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2) 
    
    huber_loss = F.smooth_l1_loss(current_quantiles.unsqueeze(2), 
                                  target_quantiles.unsqueeze(1), 
                                  reduction='none')
    
    loss = torch.abs(cumulative_probabilities - (diff < 0).float()) * huber_loss
    return loss.mean()

def train_qrdqn(actor, q_net, target_q_net, q_opt, buffer, cum_probs):
    batch = random.sample(buffer, BATCH_SIZE)
    s, a, r, s_, d = [torch.FloatTensor(np.array(x)) for x in zip(*batch)]

    with torch.no_grad():
        a_next = actor(s_)
        target_quantiles = target_q_net(s_, a_next)
        y_quantiles = r.unsqueeze(1) + GAMMA * (1 - d).unsqueeze(1) * target_quantiles

    current_quantiles = q_net(s, a)
    loss = quantile_huber_loss(current_quantiles, y_quantiles, cum_probs)
    
    q_opt.zero_grad()
    loss.backward()
    q_opt.step()

    for param, target_param in zip(q_net.parameters(), target_q_net.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

def observation_adapter(env_obs):
    ego = env_obs.ego_vehicle_state
    return np.array([ego.speed, ego.steering, ego.heading, ego.lane_index] + [0.0]*24, dtype=np.float32)

def main():
    agent_id = "QR_DQN_Baseline"
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Full, max_episode_steps=1000),
        observation_adapter=observation_adapter,
        reward_adapter=lambda obs, reward: reward,
        action_adapter=lambda action: np.array([action[0], action[1]], dtype=np.float32),
    )

    env = HiWayEnv(scenarios=["scenarios/sumo/intersections/4lane"], agent_specs={agent_id: agent_spec}, headless=True)
    
    q_net = QuantileNetwork(STATE_DIM, ACTION_DIM)
    target_q_net = QuantileNetwork(STATE_DIM, ACTION_DIM)
    target_q_net.load_state_dict(q_net.state_dict())
    
    actor = Actor(STATE_DIM, ACTION_DIM)
    q_opt = optim.Adam(q_net.parameters(), lr=LR)
    buffer = deque(maxlen=1000000)

    cum_probs = (torch.arange(N_QUANTILES) + 0.5) / N_QUANTILES
    cum_probs = cum_probs.view(1, 1, -1) 

    for ep in range(500):
        obs = env.reset()
        state = obs[agent_id]
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = actor(state_t).detach().cpu().numpy()[0]

            next_obs, rewards, dones, _ = env.step({agent_id: action})
            buffer.append((state, action, rewards[agent_id], next_obs[agent_id], dones[agent_id]))
            state = next_obs[agent_id]
            done = dones[agent_id]

            if len(buffer) > BATCH_SIZE:
                train_qrdqn(actor, q_net, target_q_net, q_opt, buffer, cum_probs)

    env.close()

if __name__ == "__main__":
    main()
