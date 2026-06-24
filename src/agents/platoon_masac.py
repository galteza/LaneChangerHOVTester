import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from configs.configs import RLArgs

# Import your custom wrapper!
# from your_wrapper_file import HighwayMARLWrapper

# =============================================================================
# REPLAY BUFFER
# =============================================================================
class MultiAgentReplayBuffer:
    def __init__(self, buffer_size, num_agents, obs_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.device = device
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.observations[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.actions[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.rewards[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_observations[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.dones[idxs], dtype=torch.float32).to(self.device)
        )


# =============================================================================
# CENTRALIZED CRITIC (Evaluates the entire 10-car setup)
# =============================================================================
class CentralizedSoftQNetwork(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim):
        super().__init__()
        # Input is the flattened joint observation + flattened joint action
        input_dim = (num_agents * obs_dim) + (num_agents * action_dim) # 140
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, joint_x, joint_a):
        x = torch.cat([joint_x, joint_a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =============================================================================
# SHARED ACTOR (The "Universal Handbook" for all cars)
# =============================================================================
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class SharedActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Takes ONLY a single car's observation
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        
        # Highway-env continuous actions are naturally bounded [-1, 1]
        self.register_buffer("action_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

# =============================================================================
# MASAC-RL AGENT
# =============================================================================

class MASACRL():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Core Networks
        self.actor = SharedActor(args.obs_dim, args.action_dim).to(device)
        self.qf1 = CentralizedSoftQNetwork(args.num_agents, args.obs_dim, args.action_dim).to(device)
        self.qf2 = CentralizedSoftQNetwork(args.num_agents, args.obs_dim, args.action_dim).to(device)
        
        # Target Networks
        self.qf1_target = CentralizedSoftQNetwork(args.num_agents, args.obs_dim, args.action_dim).to(device)
        self.qf2_target = CentralizedSoftQNetwork(args.num_agents, args.obs_dim, args.action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)
        
        # Temperature (Entropy Autotune)
        self.autotune = args.autotune
        if self.autotune:
            # target entropy is -dim(A) for the whole joint team
            self.target_entropy = -float(args.num_agents * args.action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=args.q_lr)
        else:
            self.alpha = args.alpha

    def get_action(self, obs):
        """Accepts raw numpy observations matrix (10, 12) and returns numpy actions."""
        # 1. Convert the entire (10, 12) numpy matrix to a PyTorch tensor at once
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # 2. Pass it directly.
            # Output shape: (10, 2)
            actions_tensor, _, _ = self.actor.get_action(obs_tensor)
            
        # 3. Move back to CPU and convert directly to a numpy array for the environment wrapper
        return actions_tensor.cpu().numpy()

    def update(self, data, global_step):
        """Runs the entire multi-agent centralized critic optimization pipeline."""
        # 1. Prepare flat representations for the centralized critics
        # (Batch, 10, 12) -> (Batch, 120)
        # [obs, action, reward, next obs, done]
    
        flat_joint_obs = data[0].view(self.args.batch_size, -1)
        flat_joint_next_obs = data[3].view(self.args.batch_size, -1)
        flat_joint_actions = data[1].view(self.args.batch_size, -1)
        
        joint_reward = data[2].sum(dim=1)
        joint_done = data[4].any(dim=1, keepdim=True).float()

        # =====================================================================
        # CRITIC UPDATE
        # =====================================================================
        with torch.no_grad():
            flat_next_obs_for_actor = data[3].view(-1, self.args.obs_dim)
            next_state_actions_flat, next_state_log_pi_flat, _ = self.actor.get_action(flat_next_obs_for_actor)
            
            next_state_actions_joint = next_state_actions_flat.view(self.args.batch_size, -1)
            next_state_log_pi_joint = next_state_log_pi_flat.view(self.args.batch_size, self.args.num_agents, 1).sum(dim=1)
            
            qf1_next_target = self.qf1_target(flat_joint_next_obs, next_state_actions_joint)
            qf2_next_target = self.qf2_target(flat_joint_next_obs, next_state_actions_joint)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi_joint
            
            next_q_value = joint_reward + (1 - joint_done) * self.args.gamma * min_qf_next_target

        qf1_a_values = self.qf1(flat_joint_obs, flat_joint_actions)
        qf2_a_values = self.qf2(flat_joint_obs, flat_joint_actions)
        
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # =====================================================================
        # SHARED ACTOR & ALPHA UPDATE
        # =====================================================================
        actor_loss_val = 0.0
        if global_step % self.args.policy_frequency == 0:
            for _ in range(self.args.policy_frequency):

                # Route current states through the actor

                flat_obs_for_actor = data[0].view(-1, self.args.obs_dim)
                pi_flat, log_pi_flat, _ = self.actor.get_action(flat_obs_for_actor)
                
                # Reconstruct joint action and joint log_pi

                pi_joint = pi_flat.view(self.args.batch_size, -1)
                log_pi_joint = log_pi_flat.view(self.args.batch_size, self.args.num_agents, 1).sum(dim=1)
                
                # Centralize critic evaluates new joint action

                qf1_pi = self.qf1(flat_joint_obs, pi_joint)
                qf2_pi = self.qf2(flat_joint_obs, pi_joint)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                
                actor_loss = ((self.alpha * log_pi_joint) - min_qf_pi).mean()
                actor_loss_val = actor_loss.item()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Alpha update

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi_flat, _ = self.actor.get_action(flat_obs_for_actor)
                        log_pi_joint = log_pi_flat.view(self.args.batch_size, self.args.num_agents, 1).sum(dim=1)
                    
                    alpha_loss = (-self.log_alpha.exp() * (log_pi_joint + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # =====================================================================
        # TARGET NETWORK UPDATE (Soft Update)
        # =====================================================================
        if global_step % self.args.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                
        return qf_loss.item() / 2.0, actor_loss_val