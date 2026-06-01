"""
SAC (Soft Actor-Critic) for Continuous Action Spaces

This is the original SAC algorithm designed for continuous control tasks like robotic
manipulation and locomotion. Unlike discrete SAC (for Atari), this version uses:

1. Gaussian policies with reparameterization trick
2. Tanh squashing to bound actions
3. Q-networks that take actions as input (not output Q for each action)
4. Delayed policy updates (TD3-style)

Key Components:
- Actor: Outputs mean and log_std for a Gaussian distribution
- Twin Critics: Q(s,a) networks that take both state and action as input
- Target Networks: Delayed copies for stable learning
- Automatic Entropy Tuning: Adjusts exploration dynamically

SAC Objective: Maximize expected return + entropy
  J(π) = E[∑ γ^t (r_t + α H(π(·|s_t)))]

The continuous version is the "canonical" SAC from the original paper:
"Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"
https://arxiv.org/abs/1801.01290

CleanRL docs: https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
"""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    """
    Configuration for SAC training on continuous control tasks.
    
    This dataclass uses tyro for command-line argument parsing.
    Example: python script.py --env_id HalfCheetah-v4 --seed 42 --total_timesteps 2000000
    """
    
    # Experiment metadata
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """Name of this experiment (defaults to script filename without .py)"""
    
    seed: int = 1
    """Random seed for reproducibility across PyTorch, NumPy, and Python."""
    
    torch_deterministic: bool = True
    """If True, sets torch.backends.cudnn.deterministic=True for reproducibility.
    Note: May slightly reduce performance on GPU."""
    
    cuda: bool = True
    """Enable CUDA for GPU acceleration. GPU is highly recommended for faster training."""
    
    track: bool = False
    """If True, track experiment with Weights & Biases for remote monitoring and comparison."""
    
    wandb_project_name: str = "cleanRL"
    """W&B project name for organizing related experiments."""
    
    wandb_entity: str = None
    """W&B team/entity name. None uses your personal account."""
    
    capture_video: bool = False
    """If True, record videos of agent performance. Saved to videos/{run_name}/"""

    # Environment configuration
    env_id: str = "Hopper-v4"
    """Gymnasium environment ID. Should be a continuous control task.
    Popular environments: Hopper-v4, HalfCheetah-v4, Walker2d-v4, Ant-v4, Humanoid-v4.
    MuJoCo environments require: pip install gymnasium[mujoco]"""
    
    total_timesteps: int = 1000000
    """Total number of environment steps to train for. 1M is typical for MuJoCo tasks."""
    
    num_envs: int = 1
    """Number of parallel environments. SAC typically uses 1 (unlike PPO which benefits from many).
    More envs = faster data collection but higher memory usage."""
    
    # Replay buffer
    buffer_size: int = int(1e6)
    """Maximum number of transitions to store. 1M is standard for continuous control.
    Memory usage: ~buffer_size * (obs_dim + action_dim + 3) * 4 bytes."""
    
    # Core RL hyperparameters
    gamma: float = 0.99
    """Discount factor for future rewards. 0.99 is standard for tasks with long horizons."""
    
    tau: float = 0.005
    """Soft target update coefficient for target networks.
    target = tau * current + (1-tau) * target
    Smaller tau = slower, more stable updates. 0.005 is typical for continuous control.
    Compare to discrete SAC which uses tau=1.0 (hard updates) with less frequent updates."""
    
    batch_size: int = 256
    """Number of transitions sampled per training step. 256 is standard for SAC.
    Larger batches = more stable gradients but slower iteration."""
    
    learning_starts: int = 5e3
    """Number of steps to collect with random actions before training begins.
    Ensures replay buffer has diverse initial data. 5k is typical for continuous control."""
    
    # Learning rates
    policy_lr: float = 3e-4
    """Learning rate for actor (policy) network. 3e-4 is Adam's default and works well."""
    
    q_lr: float = 1e-3
    """Learning rate for Q-networks (critics). Often slightly higher than policy_lr.
    Helps critics learn faster to provide better guidance to the policy."""
    
    # Training frequency
    policy_frequency: int = 2
    """Train policy every N critic updates (delayed policy updates, from TD3).
    Allows critics to stabilize before updating policy. 2 is typical.
    Set to 1 for standard SAC (no delay)."""
    
    target_network_frequency: int = 1
    """Update target networks every N training steps. 
    With tau=0.005, we do soft updates every step (very gradual).
    Compare to discrete SAC: tau=1.0, frequency=8000 (hard updates, infrequent)."""
    
    # SAC-specific: Entropy regularization
    alpha: float = 0.2
    """Entropy regularization coefficient. Higher alpha = more exploration.
    If autotune=True, this is just the initial value before automatic adjustment."""
    
    autotune: bool = True
    """If True, automatically adjust alpha to maintain target entropy.
    Highly recommended - eliminates need to manually tune exploration."""


def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Create a Gymnasium environment with optional video recording.
    
    For continuous control, we don't need the extensive Atari preprocessing.
    Just wrap for episode statistics tracking and optional video recording.
    
    Args:
        env_id: Gymnasium environment name (e.g., "Hopper-v4")
        seed: Random seed for this environment
        idx: Environment index (for parallel environments)
        capture_video: Whether to record videos
        run_name: Unique name for this run (for video folder)
    
    Returns:
        A function that creates the environment (thunk pattern for vectorization)
    """
    def thunk():
        # Create base environment with optional video recording
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        
        # Track episode statistics (return, length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Seed the action space for reproducibility
        env.action_space.seed(seed)
        return env

    return thunk


class SoftQNetwork(nn.Module):
    """
    Soft Q-Network (Critic) for SAC with continuous actions.
    
    Architecture: MLP that takes both state and action as input, outputs scalar Q-value
    
    Key difference from discrete SAC:
    - Discrete: Q(s) -> Q-values for all actions [vector output]
    - Continuous: Q(s,a) -> single Q-value [scalar output]
    
    This is because continuous action spaces are infinite - we can't enumerate all actions.
    Instead, we evaluate Q(s,a) for specific action values.
    
    Network structure:
      Input: [state, action] concatenated
      Hidden: 256 -> 256 (ReLU activations)
      Output: scalar Q-value
    """
    
    def __init__(self, env):
        """
        Initialize the Q-network.
        
        Args:
            env: Vectorized environment to extract observation/action space dimensions
        """
        super().__init__()
        
        # Calculate input dimension: observation_dim + action_dim
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        input_dim = obs_dim + action_dim
        
        # Three-layer MLP: input -> 256 -> 256 -> 1
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        """
        Forward pass: (state, action) -> Q-value.
        
        Args:
            x: State/observation tensor, shape (batch, obs_dim)
            a: Action tensor, shape (batch, action_dim)
        
        Returns:
            Q-value for (state, action) pair, shape (batch, 1)
        """
        # Concatenate state and action
        x = torch.cat([x, a], 1)
        
        # Pass through network with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Bounds for log standard deviation
# These prevent the policy from becoming too deterministic or too random
LOG_STD_MAX = 2   # exp(2) ≈ 7.4, allows significant exploration
LOG_STD_MIN = -5  # exp(-5) ≈ 0.007, prevents collapse to deterministic


class Actor(nn.Module):
    """
    Actor (Policy) Network for SAC with continuous actions.
    
    Outputs parameters of a Gaussian distribution: mean and log_std.
    The policy samples actions from this distribution using the reparameterization trick.
    
    Architecture:
      Input: state
      Hidden: 256 -> 256 (ReLU)
      Output heads:
        - mean: Linear(256 -> action_dim)
        - log_std: Linear(256 -> action_dim), clamped to [LOG_STD_MIN, LOG_STD_MAX]
    
    Action sampling process:
      1. Sample from Normal(mean, std): x_t ~ N(μ, σ)
      2. Squash with tanh: y_t = tanh(x_t) ∈ [-1, 1]
      3. Rescale to action bounds: a = y_t * scale + bias
    
    Tanh squashing:
      - Bounds actions to valid range (required for most continuous control)
      - Creates a smooth, differentiable policy
      - Requires log probability correction (Jacobian of tanh transformation)
    """
    
    def __init__(self, env):
        """
        Initialize the policy network.
        
        Args:
            env: Vectorized environment to extract observation/action space info
        """
        super().__init__()
        
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        
        # Shared trunk: state -> features
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Output heads: features -> distribution parameters
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        
        # Action rescaling: map from [-1, 1] to [action_low, action_high]
        # scale = (high - low) / 2
        # bias = (high + low) / 2
        # action = tanh(x) * scale + bias
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        """
        Forward pass: state -> distribution parameters (mean, log_std).
        
        Args:
            x: State/observation tensor, shape (batch, obs_dim)
        
        Returns:
            mean: Mean of action distribution, shape (batch, action_dim)
            log_std: Log standard deviation, shape (batch, action_dim)
        """
        # Shared feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Distribution parameters
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        
        # Constrain log_std to prevent extreme values
        # tanh maps to [-1, 1], then we scale to [LOG_STD_MIN, LOG_STD_MAX]
        # This technique is from SpinningUp and Denis Yarats' implementation
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        """
        Sample an action from the policy and compute log probability.
        
        This method implements the full action sampling pipeline:
        1. Get distribution parameters (mean, std)
        2. Sample from Gaussian using reparameterization trick
        3. Apply tanh squashing to bound actions
        4. Rescale to environment's action space
        5. Compute log probability with correction for tanh
        
        The reparameterization trick is critical for SAC:
          Instead of: a ~ π(·|s)  [non-differentiable sampling]
          We use: a = μ(s) + σ(s) * ε, where ε ~ N(0,1)
          This makes the action differentiable w.r.t. policy parameters.
        
        Args:
            x: State/observation tensor, shape (batch, obs_dim)
        
        Returns:
            action: Sampled action rescaled to env bounds, shape (batch, action_dim)
            log_prob: Log probability of the action, shape (batch, 1)
            mean: Deterministic action (for evaluation), shape (batch, action_dim)
        """
        # Get distribution parameters
        mean, log_std = self(x)
        std = log_std.exp()
        
        # Create Gaussian distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Sample action using reparameterization trick
        # rsample() = mean + std * eps, where eps ~ N(0,1)
        # This is differentiable w.r.t. mean and std
        x_t = normal.rsample()
        
        # Apply tanh squashing to bound to [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Rescale to environment's action bounds
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        
        # Apply change of variables formula for tanh transformation
        # When we transform x -> y = tanh(x), we need to account for the Jacobian:
        # log π(y) = log π(x) - log|dy/dx| = log π(x) - log|1 - tanh²(x)|
        # 
        # Derivation:
        #   d/dx tanh(x) = 1 - tanh²(x) = 1 - y²
        #   log|dy/dx| = log(1 - y²)
        #   log π(y) = log π(x) - log(1 - y²)
        #
        # We also account for action rescaling by dividing by action_scale
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        
        # Sum across action dimensions (assume independence)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Deterministic action for evaluation (no noise)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


if __name__ == "__main__":
    # Parse command-line arguments
    args = tyro.cli(Args)
    
    # Create unique run name with timestamp
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Initialize Weights & Biases tracking if requested
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,  # Sync TensorBoard logs to W&B
            config=vars(args),
            name=run_name,
            monitor_gym=True,       # Monitor gym environments
            save_code=True,         # Save code for reproducibility
        )
    
    # Initialize TensorBoard writer for local logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # =============================================================================
    # SEEDING: Ensure reproducibility
    # =============================================================================
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set device (prefer GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # =============================================================================
    # ENVIRONMENT SETUP
    # =============================================================================
    # Create vectorized environments
    # Even with num_envs=1, vectorization provides a consistent interface
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    
    # Verify continuous action space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Get maximum action value (for logging/debugging)
    max_action = float(envs.single_action_space.high[0])

    # =============================================================================
    # NETWORK INITIALIZATION
    # =============================================================================
    
    # Actor: Stochastic policy that outputs action distribution parameters
    actor = Actor(envs).to(device)
    
    # Critics: Two Q-networks for double Q-learning (reduces overestimation)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    
    # Target networks: Delayed copies for stable TD targets
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())  # Initialize with same weights
    qf2_target.load_state_dict(qf2.state_dict())
    
    # Optimizers
    # Note: Q-networks often use higher learning rate than policy
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # =============================================================================
    # AUTOMATIC ENTROPY TUNING
    # =============================================================================
    # Alpha controls exploration vs exploitation trade-off
    # Automatic tuning adjusts alpha to maintain target entropy
    if args.autotune:
        # Target entropy heuristic: -dim(A)
        # This means we target a policy with entropy slightly less than uniform
        # For a d-dimensional action space with Gaussian policy:
        #   H_uniform ≈ d * log(sqrt(2πe)) ≈ d * 2.05
        #   H_target = -d (approximately half of uniform)
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        
        # Log-space parameterization ensures alpha > 0
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        
        # Optimizer for alpha (separate from networks)
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # =============================================================================
    # REPLAY BUFFER
    # =============================================================================
    # Stores transitions for off-policy learning
    envs.single_observation_space.dtype = np.float32  # Ensure float32 for efficiency
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,  # We handle this manually
    )
    
    start_time = time.time()

    # =============================================================================
    # MAIN TRAINING LOOP
    # =============================================================================
    obs, _ = envs.reset(seed=args.seed)
    
    for global_step in range(args.total_timesteps):
        
        # =========================================================================
        # ACTION SELECTION
        # =========================================================================
        if global_step < args.learning_starts:
            # Warmup phase: Random actions to fill replay buffer
            # Important for having diverse initial data before learning
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Use learned policy to select actions
            # During training, we sample from the stochastic policy (exploration)
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # =========================================================================
        # ENVIRONMENT STEP
        # =========================================================================
        # Execute action in environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # =========================================================================
        # LOGGING
        # =========================================================================
        # Log episodic returns when episodes complete
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # =========================================================================
        # STORE TRANSITION IN REPLAY BUFFER
        # =========================================================================
        # Handle truncated episodes properly
        # When episode is truncated (time limit), the next_obs returned is from the
        # reset, not the actual next state. We need the actual next state for TD target.
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                # Use the "final" observation before reset
                real_next_obs[idx] = infos["final_observation"][idx]
        
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # Update current observation for next step
        obs = next_obs

        # =========================================================================
        # TRAINING
        # =========================================================================
        if global_step > args.learning_starts:
            # Sample a batch of transitions from replay buffer
            data = rb.sample(args.batch_size)
            
            # =====================================================================
            # CRITIC (Q-FUNCTION) UPDATE
            # =====================================================================
            # Compute TD target for Q-functions
            with torch.no_grad():
                # Sample actions from current policy for next states
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                
                # Get Q-values for next state from target networks
                # We use target networks for stability (delayed updates)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                
                # Take minimum Q-value to reduce overestimation (double Q-learning)
                # Subtract entropy bonus: Q - α*log(π)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                
                # Bellman backup: r + γ * (1 - done) * [Q(s',a') - α*log(π(a'|s'))]
                # This is the TD target we want our Q-networks to match
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            # Get current Q-values for the actions that were actually taken
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            
            # Compute MSE loss between predicted Q and target Q
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # Update Q-networks
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # =====================================================================
            # ACTOR (POLICY) UPDATE - Delayed (TD3-style)
            # =====================================================================
            # Update policy less frequently than Q-functions
            # This gives Q-functions time to improve before using them to train policy
            if global_step % args.policy_frequency == 0:
                # Compensate for delayed updates by doing multiple policy updates
                # If policy_frequency=2, we do 2 policy updates per critic update cycle
                for _ in range(args.policy_frequency):
                    # Sample actions from current policy for the batch states
                    pi, log_pi, _ = actor.get_action(data.observations)
                    
                    # Get Q-values for these actions from current Q-networks
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    
                    # Actor loss: maximize E[Q(s,a) - α*log(π(a|s))]
                    # Equivalently, minimize E[α*log(π(a|s)) - Q(s,a)]
                    # This encourages:
                    #   1. Actions with high Q-values (exploitation)
                    #   2. High entropy policy (exploration, weighted by α)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    # Update policy network
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # =============================================================
                    # ENTROPY TEMPERATURE (ALPHA) UPDATE
                    # =============================================================
                    if args.autotune:
                        # Resample actions to get fresh log_pi (with no_grad for efficiency)
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        
                        # Alpha loss: adjust α to match target entropy
                        # We want entropy H[π(·|s)] ≈ target_entropy
                        # Since H = -E[log π], we want: -E[log π] ≈ target_entropy
                        # Equivalently: E[log π] ≈ -target_entropy
                        # 
                        # Loss: -α * (E[log π(a|s)] + target_entropy)
                        # When entropy is too low (log_pi too negative):
                        #   loss is positive → increase α → more exploration
                        # When entropy is too high (log_pi less negative):
                        #   loss is negative → decrease α → less exploration
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        # Update alpha (in log space)
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()  # Convert back from log-space

            # =====================================================================
            # TARGET NETWORK UPDATE - Soft update every step
            # =====================================================================
            # Slowly blend current network weights into target networks
            # target = τ * current + (1-τ) * target
            # With τ=0.005, target networks change very gradually
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            # =====================================================================
            # LOGGING TRAINING METRICS
            # =====================================================================
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                
                # Log training speed (steps per second)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    # Cleanup
    envs.close()
    writer.close()