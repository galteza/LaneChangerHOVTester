import gymnasium as gym
import torch
import time
import os
import re
from pathlib import Path

from src.agents.platoon_masac import MASACRL
from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment, Wrapper_MergeExitLaneHighway_Environment
from configs.configs import RLArgs

# IMPORTANT: You need to import or recreate the 'args' object you used during training
# so that MASACRL initializes with the correct dimensions (obs_dim, action_dim, etc.)
# from src.config import get_args  <-- Adjust this to match your codebase

if __name__ == "__main__":
    # 1. Setup Environment
    base_env = MergeExitLaneHighway_Environment(render_mode="human")
    env = Wrapper_MergeExitLaneHighway_Environment(base_env)

    # 2. Setup Device (Render on CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Initialize Model Architecture
    print("Initializing MASACRL architecture...")
    args = RLArgs() # Uncomment and set this up so your class gets its parameters
    model = MASACRL(args) 

    # 4. Find the newest run directory and pick the best available actor checkpoint

    # --- Configuration ---
    # Set to your specific folder path, or None to skip straight to the fallback.
    specific_run_dir = None # "runs/merge_exit_highway__configs__1__20260716-183711"

    # Set to a specific episode integer (e.g., 500), or None to use the default fallback logic.
    specific_episode = 910000 
    # ---------------------

    # Determine the run directory (try specific first, fallback to newest)
    run_dir = None

    if specific_run_dir:
        candidate_dir = Path(specific_run_dir)
        if candidate_dir.exists() and candidate_dir.is_dir():
            run_dir = candidate_dir
        else:
            print(f"Warning: Specific folder '{candidate_dir}' not found. Falling back to newest run.")

    if run_dir is None:
        runs_root = Path(__file__).resolve().parent / "runs"
        
        if not runs_root.exists():
            raise FileNotFoundError(f"Runs root directory not found: {runs_root}")
            
        run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found in {runs_root}")

        run_dir = max(run_dirs, key=lambda path: path.stat().st_mtime)

    print(f"Loading checkpoints from: {run_dir}")

    # Determine the actor checkpoint
    actor_path = None

    # Attempt to load the specific episode if requested
    if specific_episode is not None:
        candidate_actor = run_dir / f"checkpoint_actor_{specific_episode}.pth"
        if candidate_actor.exists():
            actor_path = candidate_actor
            print(f"Found specific checkpoint: {actor_path.name}")
        else:
            print(f"Warning: {candidate_actor.name} not found. Falling back to default checkpoints.")

    # Fallback logic if no specific episode was requested (or if it was missing)
    if actor_path is None:
        actor_final_path = run_dir / "actor_final.pth"
        actor_interrupted_path = run_dir / "actor_interrupted.pth"
        
        # Find the latest checkpoint as a last resort
        checkpoint_candidates = list(run_dir.glob("checkpoint_actor_*.pth"))
        latest_checkpoint = None
        if checkpoint_candidates:
            def _checkpoint_step(path: Path) -> int:
                match = re.search(r"checkpoint_actor_(\d+)\.pth$", path.name)
                return int(match.group(1)) if match else -1

            latest_checkpoint = max(checkpoint_candidates, key=_checkpoint_step)

        # Pick the best available file
        if actor_final_path.exists():
            actor_path = actor_final_path
        elif actor_interrupted_path.exists():
            actor_path = actor_interrupted_path
        elif latest_checkpoint is not None:
            actor_path = latest_checkpoint
        else:
            raise FileNotFoundError(
                f"No actor checkpoint found in {run_dir}. Expected `actor_final.pth`, `actor_interrupted.pth`, or `checkpoint_actor_<episode>.pth`."
            )

    print(f"Selected actor model: {actor_path}")

    # You only strictly need the actor for rendering, but you can load critics if you want to log Q-values!
    
    # 5. Load the trained weights into your custom actor
    print(f"Loading trained weights from {actor_path}...")
    model.actor.load_state_dict(torch.load(actor_path, map_location=device))
    model.actor.eval() # Set network to evaluation mode (turns off dropout/batchnorm updates if any)

    obs, info = env.reset()

    # env.unwrapped.vehicle = env.unwrapped.road.vehicles[0]

    if hasattr(base_env.unwrapped, "viewer") and base_env.unwrapped.viewer is not None:
        # Manually point the viewer's target to the victim vehicle object directly
        # assuming you defined 'victim' somewhere accessible, or from road vehicles:
        base_env.unwrapped.viewer.observer_vehicle = base_env.unwrapped.road.vehicles[0]

    print("Running model through environment... press Ctrl+C to stop.")
    try:
        while True:
            # 6. Use your custom get_action method with the deterministic flag we added!
            # This ensures the swarm executes the master strategy without exploration noise.
            action = model.get_action(obs) 
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            # Adjust sleep time if the simulation renders too fast or too slow
            time.sleep(0.02)
            
            if terminated or truncated:
                print("Episode finished. Resetting environment...")
                obs, info = env.reset()

                if hasattr(base_env.unwrapped, "viewer") and base_env.unwrapped.viewer is not None:
                    base_env.unwrapped.viewer.observer_vehicle = base_env.unwrapped.road.vehicles[0]
                # env.unwrapped.vehicle = env.unwrapped.road.vehicles[0]
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        env.close()