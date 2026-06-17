# General imports

import time
import numpy as np
import random
import os

# Imports for RL

import torch
from torch.utils.tensorboard import SummaryWriter
import tyro

# Imports for self-made objects (agents)

from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment, Wrapper_MergeExitLaneHighway_Environment
from src.agents.platoon_masac import MASACRL, MultiAgentReplayBuffer

# Imports for configs    

from configs.configs import RLArgs
    
if __name__ == "__main__":

    RLargs = tyro.cli(RLArgs)

    run_name = f"{RLargs.env_id}__{RLargs.exp_name}__{RLargs.seed}__{int(time.time())}"
    save_dir = f"runs/{run_name}"

    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(save_dir)
    
    random.seed(RLargs.seed)
    np.random.seed(RLargs.seed)
    torch.manual_seed(RLargs.seed)
    torch.backends.cudnn.deterministic = RLargs.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and RLargs.cuda else "cpu")
    print(f"Using device: {device}")

    base_env = MergeExitLaneHighway_Environment()

    env = Wrapper_MergeExitLaneHighway_Environment(base_env) # observation is (n x 12)

    agent = MASACRL(RLargs, device)

    rb = MultiAgentReplayBuffer(RLargs.buffer_size, RLargs.num_agents, RLargs.obs_dim, RLargs.action_dim, device)
    
    start_time = time.time()
    obs, _ = env.reset(seed=RLargs.seed) # already flattened observation matrix
    
    for global_step in range(RLargs.total_timesteps):

        if global_step < RLargs.learning_starts:
            actions = np.random.uniform(-1, 1, (RLargs.num_agents, RLargs.action_dim))
        else:
            actions = agent.get_action(obs)     # actor input / obs is (n x 12), actor output / action is (n x 2)

        # ENVIRONMENT STEP
        next_obs, reward, done, truncated, info = env.step(actions)     # fed (n x 2), outputs (n x 12) next_obs
        
        # STORE TRANSITION
        rb.add(obs, actions, reward, next_obs, done)
        obs = next_obs if not (done or truncated) else env.reset()[0]

        # TRAINING
        if global_step > RLargs.learning_starts:
            data = rb.sample(RLargs.batch_size)
            # data is composed of 5 tensors: observations, action, reward, next observations, done
            
            qf_loss_val, actor_loss_val = agent.update(data, global_step)

            # Logging
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf_loss", qf_loss_val, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_val, global_step)
                writer.add_scalar("losses/alpha", agent.alpha, global_step)
                print(f"Step: {global_step} | SPS: {int(global_step / (time.time() - start_time))}")

        # SAVING
        if global_step % (RLargs.total_timesteps/10) == 0:
            torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"checkpoint_actor_{global_step}.pth"))
            print(f"Checkpoint at {global_step} saved in {save_dir}")

    writer.close()
    
    torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_final.pth"))
    torch.save(agent.qf1.state_dict(), os.path.join(save_dir, "criticqf1_final.pth"))
    torch.save(agent.qf2.state_dict(), os.path.join(save_dir, "criticqf2_final.pth"))

    print(f"Final model weights successfully saved in {save_dir}")
