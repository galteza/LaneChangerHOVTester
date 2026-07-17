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

# Import for video recording

from gymnasium.wrappers import RecordVideo
    
if __name__ == "__main__":

    RLargs = tyro.cli(RLArgs) # Load up the RL arguments from the configs

    formatted_time = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{RLargs.env_id}__{RLargs.exp_name}__{RLargs.seed}__{formatted_time}" # Name the run according to time
    save_dir = f"runs/{run_name}" # Create a directory to save the upcoming training results
    os.makedirs(save_dir, exist_ok=True) 

    # Initialize the SummaryWriter for TensorBoard logging

    writer = SummaryWriter(save_dir)
    
    # Set random seeds for reproducibility

    random.seed(RLargs.seed)
    np.random.seed(RLargs.seed)
    torch.manual_seed(RLargs.seed)
    torch.backends.cudnn.deterministic = RLargs.torch_deterministic # Slows down training but ensures reproducibility
    
    # Set device for PyTorch; use GPU if available otherwise fallback to CPU

    device = torch.device(RLargs.device)
    print(f"Using device: {device}")

    # Initialize environment, agent, and replay buffer    

    training_env = MergeExitLaneHighway_Environment()
    wrapped_training_env = Wrapper_MergeExitLaneHighway_Environment(training_env) # For changing up the observation space

    raw_viewing_env = MergeExitLaneHighway_Environment(render_mode="rgb_array") # For rendering the environment during training
    wrapped_viewing_env = Wrapper_MergeExitLaneHighway_Environment(raw_viewing_env) # For changing up the observation space
    viewing_env = RecordVideo(
                    wrapped_viewing_env, 
                    video_folder=os.path.join(save_dir, "checkpoint_videos"), 
                    episode_trigger=lambda episode_id: True, 
                    name_prefix=f"checkpoint"
                )

    # Setting camera focus to be LC
    unwrapped_viewing_env = viewing_env.unwrapped
    unwrapped_viewing_env.observation_type.observer_vehicle = unwrapped_viewing_env.ego 

    agent = MASACRL(RLargs) # The learning agent

    rb = MultiAgentReplayBuffer(RLargs) # The replay buffer for storing transitions

    # Start training loop
    
    start_time = time.time()

    flat_obs, _ = wrapped_training_env.reset(seed=RLargs.seed) # already flattened observation matrix

    try:

        reward_cumulative = np.zeros(RLargs.num_agents)
        step_counter = 0
        average_reward = np.zeros(RLargs.num_agents)
    
        for global_step in range(RLargs.total_timesteps):

            # If still in the initial exploration phase, take random actions, otherwise use agent's prepared policy

            if global_step < RLargs.learning_starts:
                actions = np.random.uniform(-1, 1, (RLargs.num_agents, RLargs.action_dim))
            else:
                actions = agent.get_action(flat_obs)

            # ENV STEP: Take next step in environment using chosen actions

            next_flat_obs, reward, done, truncated, info = wrapped_training_env.step(actions)

            reward_cumulative += reward
            step_counter += 1
            average_reward = reward_cumulative / step_counter

            if done or truncated:
                print(f"==== EP FINISHED AFTER {step_counter} STEPS /// AVERAGE REWARD: {average_reward} ====")
                reward_cumulative = np.zeros(RLargs.num_agents)
                step_counter = 0
            
            # STORAGE IN BUFFER: Store all these in the buffer for later training
            
            rb.add(flat_obs, actions, reward, next_flat_obs, done)
            flat_obs = next_flat_obs if not (done or truncated) else wrapped_training_env.reset()[0] # Move to next states
            
            # TRAINING: Sample buffer and train agent!

            if global_step > RLargs.learning_starts:
                data = rb.sample() # data composed of 5 tensors: observations, action, reward, next observations, done
                
                qf_loss_val, actor_loss_val = agent.update(data, global_step) # run pipeline using given data

                # LOGGING: Logging losses regularly to TensorBoard and console for monitoring training progress
                if global_step % RLargs.logging_frequency == 0:
                    writer.add_scalar("losses/qf_loss", qf_loss_val, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss_val, global_step)
                    writer.add_scalar("losses/alpha", agent.alpha, global_step)
                    for idx in range(RLargs.num_agents):
                        writer.add_scalar(f"average reward_{idx}", average_reward[idx], global_step)
                    print(f"Step: {global_step} | SPS: {int(global_step / (time.time() - start_time))} | Reward: {np.mean(reward)} {reward} ")

                    writer.flush() # Ensure that all pending events have been written to disk

            # SAVING: Save model weights regularly and generate video of current performance
            if global_step % int(RLargs.total_timesteps/RLargs.checkpoints_num) == 0:
                
                torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"checkpoint_actor_{global_step}.pth"))
                print(f"Checkpoint at {global_step} saved in {save_dir}")

                print(f"Generating video for checkpoint at {global_step}...")
                try:
                    viewing_env.name_prefix = f"checkpoint_{global_step}"

                    obs, _ = viewing_env.reset(seed=RLargs.seed)
                    done = truncated = False

                    while not (done or truncated):
                        action = agent.get_action(obs)
                        obs, reward, done, truncated, info = viewing_env.step(action)
                        viewing_env.render()

                    print(f"Checkpoint video for {global_step} saved in {os.path.join(save_dir, 'checkpoint_videos')}")
                except Exception as video_error:
                    print(f"Checkpoint video skipped at step {global_step}: {video_error}")
                    print("Training will continue without interrupting the run.")

        torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_final.pth"))
        torch.save(agent.qf1.state_dict(), os.path.join(save_dir, "criticqf1_final.pth"))
        torch.save(agent.qf2.state_dict(), os.path.join(save_dir, "criticqf2_final.pth"))

        print(f"Final model weights successfully saved in {save_dir}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model weights...")

        torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_interrupted.pth"))
        torch.save(agent.qf1.state_dict(), os.path.join(save_dir, "criticqf1_interrupted.pth"))
        torch.save(agent.qf2.state_dict(), os.path.join(save_dir, "criticqf2_interrupted.pth"))

        print(f"Interrupted model weights saved in {save_dir}")

    finally:
        
        writer.close()
        wrapped_training_env.close()
        viewing_env.close()