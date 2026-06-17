import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Universal Handbook (Shared Actor)
class SharedActor(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 inputs -> Hidden Layers -> 2 outputs (mean), 2 outputs (std for SAC)
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # [mu_throttle, mu_steer, log_std_throttle, log_std_steer]
        )
    def forward(self, x):
        return self.net(x)

# 2. Define the Centralized Critic
class CentralizedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs: (10 cars * 8 features) + (10 cars * 2 actions) = 100 inputs
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Outputs a single global Q-value
        )
    def forward(self, obs_flat, actions_flat):
        x = torch.cat([obs_flat, actions_flat], dim=-1)
        return self.net(x)

# --- THE CRITICAL TRAINING LOOP FRAME ---
# Inside your replay buffer sampling loop:

# Flat vectors for the Centralized Critic
obs_flat = batch_obs.view(-1, 80)       # Flattened batch to (Batch_Size, 80)
actions_flat = batch_actions.view(-1, 20) # Flattened batch to (Batch_Size, 20)

# Step A: Update Centralized Critic
current_Q = critic(obs_flat, actions_flat)
critic_loss = nn.MSELoss()(current_Q, target_Q_from_buffer)

critic_optimizer.zero_grad()
critic_loss.backward()
critic_optimizer.step()

# Step B: Update Shared Actor (Loop over the 10 experiences explicitly)
actor_loss = 0
for i in range(10):
    # Extract data for car i across the batch
    car_obs = batch_obs[:, i, :] # Shape: (Batch_Size, 8)
    
    # Sample new actions using the current actor policy
    new_action, log_prob = sample_action_from_actor(actor(car_obs))
    
    # Construct a counterfactual joint action vector where only car_i updates its action
    modified_actions_flat = actions_flat.clone()
    modified_actions_flat[:, i*2 : (i+1)*2] = new_action
    
    # Evaluate how much this specific action choice impacts the global Q-value
    actor_loss += (-critic(obs_flat, modified_actions_flat) + alpha * log_prob).mean()

actor_optimizer.zero_grad()
actor_loss.backward()
actor_optimizer.step()