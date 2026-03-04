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

LR_ACTOR = 3e-4
LR_CRITIC = 3e-2
GAMMA = 0.99
BATCH_SIZE = 64
TAU = 0.001
STATE_DIM = 28
DISCRETE_ACTION_DIM = 3 
PARAM_ACTION_DIM = 2 

class ParameterActor(nn.Module):
    """Outputs continuous parameters for EACH discrete action."""
    def __init__(self, state_dim, discrete_dim, param_dim):
        super(ParameterActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, discrete_dim * param_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return torch.tanh(self.output(x))

class QNetwork(nn.Module):
    """Calculates Q-values for discrete actions given state and continuous parameters."""
    def __init__(self, state_dim, discrete_dim, param_dim):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + (discrete_dim * param_dim), 256)
        self.l2 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, discrete_dim)

    def forward(self, state, params):
        x = torch.cat([state, params], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.q_out(x)

def train_pdqn(q_net, p_actor, target_q, target_p, q_opt, p_opt, buffer):
    batch = random.sample(buffer, BATCH_SIZE)
    s, a_disc, a_param, r, s_, d = [torch.FloatTensor(np.array(x)) for x in zip(*batch)]
    a_disc = a_disc.long().unsqueeze(1)

    with torch.no_grad():
        next_params = target_p(s_)
        next_q_values = target_q(s_, next_params)
        max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
        y = r.unsqueeze(1) + (1 - d.unsqueeze(1)) * GAMMA * max_next_q

    current_q_values = q_net(s, a_param)
    current_q = current_q_values.gather(1, a_disc)
    loss_q = F.mse_loss(current_q, y)
    
    q_opt.zero_grad()
    loss_q.backward()
    q_opt.step()

    all_params = p_actor(s)
    loss_p = -q_net(s, all_params).mean()
    
    p_opt.zero_grad()
    loss_p.backward()
    p_opt.step()

    for p, target_p_param in zip(p_actor.parameters(), target_p.parameters()):
        target_p_param.data.copy_(TAU * p.data + (1 - TAU) * target_p_param.data)
    for q, target_q_param in zip(q_net.parameters(), target_q.parameters()):
        target_q_param.data.copy_(TAU * q.data + (1 - TAU) * target_q_param.data)

def main():
    agent_id = "PDQN_Baseline"
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Full, max_episode_steps=1000),
        observation_adapter=lambda obs: np.array([obs.ego_vehicle_state.speed, obs.ego_vehicle_state.steering] + [0.0]*26, dtype=np.float32),
        reward_adapter=lambda obs, reward: reward,
        action_adapter=lambda action: np.array([action[0], action[1]], dtype=np.float32),
    )

    env = HiWayEnv(scenarios=["scenarios/sumo/intersections/4lane"], agent_specs={agent_id: agent_spec}, headless=True)
    
    p_actor = ParameterActor(STATE_DIM, DISCRETE_ACTION_DIM, PARAM_ACTION_DIM)
    q_net = QNetwork(STATE_DIM, DISCRETE_ACTION_DIM, PARAM_ACTION_DIM)
    target_p = ParameterActor(STATE_DIM, DISCRETE_ACTION_DIM, PARAM_ACTION_DIM)
    target_q = QNetwork(STATE_DIM, DISCRETE_ACTION_DIM, PARAM_ACTION_DIM)
    
    q_opt = optim.Adam(q_net.parameters(), lr=LR_CRITIC)
    p_opt = optim.Adam(p_actor.parameters(), lr=LR_ACTOR)
    buffer = deque(maxlen=1000000)

    for ep in range(500):
        obs = env.reset()
        state = obs[agent_id]
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            
            all_params_t = p_actor(state_t)
            q_values = q_net(state_t, all_params_t)
            disc_action = torch.argmax(q_values).item()
            
            start_idx = disc_action * PARAM_ACTION_DIM
            chosen_params = all_params_t[0, start_idx:start_idx + PARAM_ACTION_DIM].detach().cpu().numpy()

            next_obs, rewards, dones, _ = env.step({agent_id: chosen_params})
            
            buffer.append((state, disc_action, all_params_t.detach().cpu().numpy()[0], rewards[agent_id], next_obs[agent_id], dones[agent_id]))
            
            state = next_obs[agent_id]
            done = dones[agent_id]

            if len(buffer) > BATCH_SIZE:
                train_pdqn(q_net, p_actor, target_q, target_p, q_opt, p_opt, buffer)

    env.close()

if __name__ == "__main__":
    main()
