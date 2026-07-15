# Copilot Instructions — LaneChangerHOVTester

## Project Overview
A highway lane-change adversarial testing framework. The system trains a **Multi-Agent SAC (MASAC) adversarial platoon** that tries to maximize crash-risk metrics (TTC, DRAC) against a **System Under Test (SUT)** ego vehicle traversing a merge-and-exit highway scenario built on top of `highway-env`.

## Architecture

```
src/env/highway_env_mergeexit.py   ← Custom Gymnasium env (MergeExitLaneHighway_Environment)
src/env/risk_calculators.py        ← PolygonTTCCalculator for TTC/DRAC metrics
src/agents/platoon_masac.py        ← MASAC RL agent + MultiAgentReplayBuffer (adversarial platoon)
src/agents/platoon_sac.py          ← Single-agent SAC variant
src/agents/lane_changer.py         ← SUT lane changer (IDM + MOBIL)
src/agents/LC_*.py / mpc_*.py      ← MPC-based longitudinal/lateral controllers
configs/configs.py                 ← All config dataclasses (EnvArgs, RLArgs, VehicleModelParams, etc.)
configs/params_main.yaml           ← YAML params for MPC-based simulation
```

**Data flow:** `RLArgs` (tyro CLI) → `MergeExitLaneHighway_Environment` → `Wrapper_MergeExitLaneHighway_Environment` (flattens obs) → `MASACRL` agent ↔ `MultiAgentReplayBuffer` → TensorBoard logs in `runs/<run_name>/`.

## Role of Each Main Script
| Script | Purpose |
|---|---|
| `main-trainer.py` | Train adversarial MASAC platoon; saves checkpoints + videos to `runs/` |
| `main_sim.py` | MPC-based SUT simulation with SAC platoon (stable-baselines3) |
| `main_trial.py` | Quick SB3 SAC trial with flattened wrapper |
| `main-viewer.py` | Replay/render saved runs |
| `main-ttc-playground.py` | Isolated TTC metric experimentation |

## Key Config Patterns

All configuration uses **`dataclasses` + `tyro` CLI** — never edit hardcoded values directly.

```python
# Entrypoint pattern:
RLargs = tyro.cli(RLArgs)  # CLI overrides any dataclass default

# Hierarchy: RLArgs → EnvArgs → EnvObsConfigArgs / EnvActionConfigArgs / EnvRewardArgs
```

- `RLArgs.__post_init__` derives `num_agents` and `obs_dim` automatically from `EnvArgs`.
- `controlled_vehicles = vehicles_count - 1` (1 ego SUT + N adversaries; set in `EnvArgs.__post_init__`).
- `obs_dim = num_agents * feature_dim + (feature_dim - 1)` — centralized critic sees joint observations.

## Environment Conventions

- **Road topology:** 7 longitudinal sections with merge ramp (SineLane) and exit ramp; nodes labeled `(a)-(b)-(c)-(d)-(e)-(f)` plus `(j-k)` and `(l-m)` branches.
- **Observation:** `(n_vehicles, n_vehicles, 6)` kinematics matrix; `Wrapper_MergeExitLaneHighway_Environment` flattens it.
- **Agents:** `vehicles_count=20` ambient cars; `observation.vehicles_count=11` → 1 SUT ego + 10 adversaries.
- The ego vehicle (`self.ego`) uses `IDMVehicle`; adversarial platoon uses `MDPVehicle` controlled by MASAC.

## Developer Workflows

**Run training (uv):**
```bash
uv run main-trainer.py
# Override CLI args:
uv run main-trainer.py --env.vehicles-count 15 --total-timesteps 500000
```

**Monitor training:**
```bash
tensorboard --logdir runs/
```

**Add/change reward components:** edit `EnvRewardArgs` in `configs/configs.py` and the reward method in `src/env/highway_env_mergeexit.py`.

**MPC tuning:** edit weights in `SUTLaneChangerMPCParams` (configs.py) or `configs/params_main.yaml` for YAML-driven simulation (`main_sim.py`).

## Dependency Notes
- Managed via `uv` (`pyproject.toml`); Python ≥ 3.11 required.
- Key packages: `highway-env`, `torch`, `stable-baselines3`, `do-mpc`, `tyro`, `gymnasium`.
- `setuptools<81` is pinned — do not bump it.
