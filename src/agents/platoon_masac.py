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

class MultiAgentReplayBuffer:
    def __init__(self, rlargs):

        # Prepare variables from RL arguments config file
        self.rlargs = rlargs
        self.buffer_size = self.rlargs.buffer_size
        self.num_agents = self.rlargs.num_agents
        self.device = self.rlargs.device
        self.obs_dim = self.rlargs.obs_dim
        self.action_dim = self.rlargs.action_dim
        self.ptr = 0 # Rolling counter for the replay buffer's current address
        self.size = 0 # Current number of transitions in buffer
        self.batch_size = self.rlargs.batch_size # Maximum number of transitions

        # Preallocate memory for the replay buffer

        self.observations = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32) # Each agent has its own observation vector
        self.actions = np.zeros((self.buffer_size, self.num_agents, self.action_dim), dtype=np.float32) # Each agent has its own action vector
        self.rewards = np.zeros((self.buffer_size, self.num_agents, 1), dtype=np.float32) # Each agent has its own reward scalar
        self.next_observations = np.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=np.float32) # Each agent has its own next-observatin vector
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32) # Each episode is either the done point or not, scalar for the whole snapshot of the env

    def add(self, obs, action, reward, next_obs, done):
        
        # Given the calculated transition (obs, action, reward, next_obs, done), store it in the replay buffer's ptr location
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = np.array(reward).reshape(-1, 1)
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done

        # If pointer reaches end of buffer, wrap around to beginning and replace old trans
        # Update the current size of the buffer

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):

        # Draw random indices (amount of indicated batch size) from the replay buffer, submit as torch tensors

        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return (
            torch.tensor(self.observations[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.actions[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.rewards[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_observations[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.dones[idxs], dtype=torch.float32).to(self.device)
        )

class CentralizedSoftQNetwork(nn.Module):

    # Modified from neural network architecture

    def __init__(self, rlargs):
        super().__init__()

        # Initialize RL arguments from config file

        self.rlargs = rlargs

        # Input dimension for critic is joint observation and joint action space for one snapshot (of all the vehicles on the highway)

        input_dim = (self.rlargs.num_agents * self.rlargs.obs_dim) + (self.rlargs.num_agents * self.rlargs.action_dim) # For case of 10 agents, 670 input features
        
        # Building the critic network with 3 fully connected layers and 2 hidden layers of 256 neurons

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, joint_x, joint_a):

        # Take in joint observations (x) and joint actions (a), perform forward pass, returning a single Q-value
        # Joint x is a horiz

        x = torch.cat([joint_x, joint_a], 1) # Concatenate along feature di
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5

class SharedActor(nn.Module):
    def __init__(self, rlargs):
        super().__init__()
        self.rlargs = rlargs

        self.fc1 = nn.Linear(self.rlargs.obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        self.fc_mean = nn.Linear(256, self.rlargs.action_dim)
        self.fc_logstd = nn.Linear(256, self.rlargs.action_dim)
        
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
        mean, log_std = self(x) # calling forward() to get mean and log_std
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick

        y_t = torch.tanh(x_t) # action squashing to naturally bounded range of [-1, 1] in highway-env continuous action space
        action = y_t * self.action_scale + self.action_bias # [0, 1]

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6) # jacobian correction for tanh squashing
        log_prob = log_prob.sum(1, keepdim=True) # sum over action dimensions to get joint log_prob

        mean = torch.tanh(mean) * self.action_scale + self.action_bias # [0, 1]
        return action, log_prob, mean

# ==== MASAC-RL AGENT ====

class MASACRL():
    def __init__(self, rlargs):
        self.rlargs = rlargs
        self.device = self.rlargs.device
        
        # Core Networks
        self.actor = SharedActor(rlargs).to(self.device)
        self.qf1 = CentralizedSoftQNetwork(rlargs).to(self.device)
        self.qf2 = CentralizedSoftQNetwork(rlargs).to(self.device)
        
        # Target Networks
        self.qf1_target = CentralizedSoftQNetwork(rlargs).to(self.device)
        self.qf2_target = CentralizedSoftQNetwork(rlargs).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=rlargs.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=rlargs.policy_lr)
        
        # Temperature (Entropy Autotune)
        self.autotune = rlargs.autotune
        if self.autotune:
            # target entropy is -dim(A) for the whole joint team
            self.target_entropy = -float(rlargs.action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=rlargs.q_lr)
        else:
            self.alpha = rlargs.alpha

    def get_action(self, obs):
        """Accepts raw numpy observations matrix (10, 12) and returns numpy actions."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            actions_tensor, _, _ = self.actor.get_action(obs_tensor)
            
        return actions_tensor.cpu().numpy()

    def update(self, data, global_step):
        """
        The critic pipeline is to be run 10 times, once for each agent.
        Observation is created for each agent, arranged such that self observations come first, then other agents's observations, and then everyone's actions.
        The critic is then run on this joint observation and joint action, and the Q-value is returned for that agent.

        The actor pipeline is to be run 10 times 
        
        """

        joint_obs = data[0] # originally (256, n, 6n+5)
        joint_actions = data[1] # originally (256, n, 2)
        joint_rewards = data[2] # originally (256, n, 1)
        joint_next_obs = data[3] # originally (256, n, 6n+5)
        joint_done = data[4].any(dim=1, keepdim=True).float() # originally (256, 1)
        
        per_agent_rewards = joint_rewards.view(self.rlargs.batch_size, -1) # originally (256, n, 1) --> (256, n)

        # =====================================================================
        # CRITIC UPDATE
        # =====================================================================

        # Grab the next observations and run them through actor to get next actions and log probabilities for next state

        with torch.no_grad():

            flat_next_obs_for_actor = joint_next_obs.view(-1, self.rlargs.obs_dim) # originally (256, n, 6n+5) --> (256*n, 6n+5)
            next_state_actions_flat, next_state_log_pi_flat, _ = self.actor.get_action(flat_next_obs_for_actor) # action, log_prob, mean

        # Flatten the joint actions for use in critic evaluation

            next_actions_reshaped = next_state_actions_flat.view(self.rlargs.batch_size, self.rlargs.num_agents, -1) # originally (256*n, 2) --> (256, n, 2)
            next_log_pi_reshaped = next_state_log_pi_flat.view(self.rlargs.batch_size, self.rlargs.num_agents, 1) # originally (256*n, 1) --> (256, n, 1)

        # Premake the inputs into critic network for each agent prior to passing through the network!
        # Create list to store those inputs

        all_agent_obs = []
        all_agent_next_obs = []
        all_agent_actions = []
        all_agent_next_actions = []
        all_agents_rewards = []
        all_agents_log_pi = []

        for agent_idx in range(self.rlargs.num_agents):

            # Isolate info for agent in question

            self_obs = joint_obs[:, agent_idx, :] # (256, 6n+5)
            self_next_obs = joint_next_obs[:, agent_idx, :] # (256, 6n+5)
            self_actions = joint_actions[:, agent_idx, :] # (256, 2)
            self_next_actions = next_actions_reshaped[:, agent_idx, :] # (256, 2)

            # Remove questioned agent's observation from the joint observation

            other_obs = torch.cat([joint_obs[:, i, :] for i in range(self.rlargs.num_agents) if i != agent_idx], dim=1) # (256, (n-1)*(6n+5))
            other_next_obs = torch.cat([joint_next_obs[:, i, :] for i in range(self.rlargs.num_agents) if i != agent_idx], dim=1) # (256, (n-1)*(6n+5))
            other_actions = torch.cat([joint_actions[:, i, :] for i in range(self.rlargs.num_agents) if i != agent_idx], dim=1) # (256, (n-1)*2)
            other_next_actions = torch.cat([next_actions_reshaped[:, i, :] for i in range(self.rlargs.num_agents) if i != agent_idx], dim=1) # (256, (n-1)*2)

            # Create joint input for this agent --> each one to be fed into the critic

            all_agent_obs.append(torch.cat([self_obs, other_obs], dim=1)) # (256, 6n^2 + 5n)
            all_agent_next_obs.append(torch.cat([self_next_obs, other_next_obs], dim=1)) # (256, 6n^2 + 5n)
            all_agent_actions.append(torch.cat([self_actions, other_actions], dim=1)) # (256, 2n)
            all_agent_next_actions.append(torch.cat([self_next_actions, other_next_actions], dim=1)) # (256, 2n)

            # Grab agent's rewards
            all_agents_rewards.append(per_agent_rewards[:, agent_idx].unsqueeze(-1)) # (256, 1)
            all_agents_log_pi.append(next_log_pi_reshaped[:, agent_idx, :]) # (256, 1)

        # Stack vertically and prepare to send to tensor

        batch_obs = torch.cat(all_agent_obs, dim=0) # (256*n, 6n^2 + 5n)
        batch_next_obs = torch.cat(all_agent_next_obs, dim=0) # (256*n, 6n^2 + 5n)
        batch_actions = torch.cat(all_agent_actions, dim=0) # (256*n, 2n)
        batch_next_actions = torch.cat(all_agent_next_actions, dim=0) # (256*n, 2n)
        batch_rewards = torch.cat(all_agents_rewards, dim=0) # (256*n, 1)
        batch_log_pi = torch.cat(all_agents_log_pi, dim=0) # (256*n, 1)

        batch_done = joint_done.repeat_interleave(self.rlargs.num_agents, dim=0) # (256*n, 1)

        # UPDATE BASED ON REWARD (Q(r)): Feeding into critic networks to get Q-values for each agent

        with torch.no_grad():

            # Update target networks 
            qf1_next_target = self.qf1_target(batch_next_obs, batch_next_actions) # (256, 1)
            qf2_next_target = self.qf2_target(batch_next_obs, batch_next_actions) # (256, 1)
            
            # Compute the min of the two critics, and get next Q-values from target networks
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * batch_log_pi
            next_q_value = batch_rewards + (1 - batch_done) * self.rlargs.gamma * min_qf_next_target # (256*n, 1) Each unique experience/perspective gets a Q value

        # Update critic networks by computing the loss between current Q-values and next Q-values

        qf1_a_values = self.qf1(batch_obs, batch_actions) # (256*n, 1)
        qf2_a_values = self.qf2(batch_obs, batch_actions) # (256*n, 1)
        
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value) # (256*n, 1)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value) # (256*n, 1)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # =====================================================================
        # SHARED ACTOR & ALPHA UPDATE
        # =====================================================================
        actor_loss_val = 0.0
        if global_step % self.rlargs.policy_frequency == 0:
            for _ in range(self.rlargs.policy_frequency):

                # Grab the desired actions based on the current actor policy
                # Returned is the action, log probability of that action, and the mean of the action distribution

                flat_obs_for_actor = joint_obs.view(-1, self.rlargs.obs_dim) # originally (256, n, 6n+5) --> (256*n, 6n+5)
                pi_flat, log_pi_flat, _ = self.actor.get_action(flat_obs_for_actor) # originally (256*n, 2), (256*n, 1), (256*n, 2)

                # Have actions be separated per agent per episode
                
                pi_reshaped = pi_flat.view(self.rlargs.batch_size, self.rlargs.num_agents, -1) # originally (256*n, 2) --> (256, n, 2)
                
                # Build critic input space per agent using their desired actions 

                all_agents_pi = []

                for agent_idx in range(self.rlargs.num_agents):
                    self_pi = pi_reshaped[:, agent_idx, :] # (256, 2)
                    other_pi = torch.cat([pi_reshaped[:, i, :] for i in range(self.rlargs.num_agents) if i != agent_idx], dim=1) # (256, (n-1)*2)
                    all_agents_pi.append(torch.cat([self_pi, other_pi], dim=1)) # (256, 2n)

                batch_pi_actions = torch.cat(all_agents_pi, dim=0) # (256*n, 2n)
                
                # Critic evaluates the desired actions

                qf1_pi = self.qf1(batch_obs, batch_pi_actions) # (256*n, 1) from (256*n, 6n^2 + 5n) and (256*n, 2n)
                qf2_pi = self.qf2(batch_obs, batch_pi_actions) # (256*n, 1) from (256*n, 6n^2 + 5n) and (256*n, 2n)
                min_qf_pi = torch.min(qf1_pi, qf2_pi) # (256*n, 1)
                
                # Calculate the actor loss (direction of policy improvement) using the min Q-value and the log probability of the action, scaled by alpha

                actor_loss = ((self.alpha * log_pi_flat) - min_qf_pi).mean() # (256*n, 1) --> scalar

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Updating alpha (temperature parameter) if autotune is enabled, using the log probability of the action and the target entropy

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi_flat, _ = self.actor.get_action(flat_obs_for_actor)
                    
                    alpha_loss = (-self.log_alpha.exp() * (log_pi_flat + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # =====================================================================
        # TARGET NETWORK UPDATE (Soft Update)
        # =====================================================================
        if global_step % self.rlargs.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.rlargs.tau * param.data + (1 - self.rlargs.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.rlargs.tau * param.data + (1 - self.rlargs.tau) * target_param.data)
                
        return qf_loss.item() / 2.0, actor_loss_val