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
SEGMENT_LEN = 25

class RewardModel(nn.Module):
    """Intrinsic reward model learned from preferences."""
    def __init__(self, state_dim, action_dim):
        super(RewardModel, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.r_out = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        return self.r_out(x)

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

def learn_reward(reward_model, reward_opt, pref_buffer):
    """Update reward model using Bradley-Terry preference labels."""
    if len(pref_buffer) < 10: return
    
    batch = random.sample(pref_buffer, min(len(pref_buffer), 16))
    seg1, seg2, labels = zip(*batch)
    
    def get_seg_reward(seg):
        s, a = zip(*seg)
        s, a = torch.FloatTensor(np.array(s)), torch.FloatTensor(np.array(a))
        return reward_model(s, a).sum()

    r1 = torch.stack([get_seg_reward(s) for s in seg1])
    r2 = torch.stack([get_seg_reward(s) for s in seg2])
    
    logits = torch.stack([r1, r2], dim=1)
    loss = F.cross_entropy(logits, torch.LongTensor(labels))
    
    reward_opt.zero_grad()
    loss.backward()
    reward_opt.step()

def train_pebble(actor, critics, target_critics, reward_model, actor_opt, critic_opts, buffer):
    batch = random.sample(buffer, BATCH_SIZE)
    s, a, _, s_, d = [torch.FloatTensor(np.array(x)) for x in zip(*batch)]
    
    with torch.no_grad():
        r_hat = reward_model(s, a)
        a_next, log_p_next = actor(s_)
        q_target = torch.min(target_critics[0](s_, a_next), target_critics[1](s_, a_next)) - ALPHA * log_p_next
        y = r_hat + (1 - d) * GAMMA * q_target

    for i in range(2):
        q_current = critics[i](s, a)
        loss = F.mse_loss(q_current, y)
        critic_opts[i].zero_grad()
        loss.backward()
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
    agent_id = "PEBBLE_Agent"
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Full, max_episode_steps=1000),
        observation_adapter=lambda obs: np.array([obs.ego_vehicle_state.speed, obs.ego_vehicle_state.steering] + [0.0]*24, dtype=np.float32),
        reward_adapter=lambda obs, reward: reward,
        action_adapter=lambda action: np.array([action[0], action[1]], dtype=np.float32),
    )

    env = HiWayEnv(scenarios=["scenarios/sumo/intersections/4lane"], agent_specs={agent_id: agent_spec}, headless=True)
    
    reward_model = RewardModel(STATE_DIM, ACTION_DIM)
    reward_opt = optim.Adam(reward_model.parameters(), lr=LR)
    actor = Actor(STATE_DIM, ACTION_DIM)
    critics = [Critic(STATE_DIM, ACTION_DIM) for _ in range(2)]
    target_critics = [Critic(STATE_DIM, ACTION_DIM) for _ in range(2)]
    
    actor_opt = optim.Adam(actor.parameters(), lr=LR)
    critic_opts = [optim.Adam(c.parameters(), lr=LR) for c in critics]
    
    buffer = deque(maxlen=1000000)
    pref_buffer = deque(maxlen=5000)
    current_seg = []

    oracle = lambda s1, s2: 0 if np.mean([x[0] for x in s1]) > np.mean([x[0] for x in s2]) else 1

    total_steps = 0
    for ep in range(500):
        observations = env.reset()
        state = observations[agent_id]
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, _ = actor(state_t)
            action_np = action.detach().cpu().numpy()[0]

            next_obs, _, dones, _ = env.step({agent_id: action_np})
            buffer.append((state, action_np, 0, next_obs[agent_id], dones[agent_id]))
            current_seg.append((state, action_np))
            
            state = next_obs[agent_id]
            done = dones[agent_id]
            total_steps += 1

            if len(buffer) > BATCH_SIZE:
                train_pebble(actor, critics, target_critics, reward_model, actor_opt, critic_opts, buffer)
            
            if total_steps % FEEDBACK_INTERVAL == 0 and len(pref_buffer) < 100:
                pass

    env.close()

if __name__ == "__main__":
    main()
