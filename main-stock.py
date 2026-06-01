import numpy as np
import time

from matplotlib import pyplot as plt
import yaml

from src.agents.decoupled_lanechanger_mpc import DecoupledLaneChanger_MPC
from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment
from src.agents.lc_corridor_generator import LaneChangePlanner
from src.agents.mpc_lateral import MPC_Lateral
from src.agents.mpc_longitudinal import MPC_Longitudinal
from src.agents.maneuver_coordinator import Maneuver_Coordinator

# === 0. PREPARING CONFIGS & ARRAYS FOR GRAPH DISPLAY ===

with open("configs/simenv_params.yaml", "r") as f:
    simenv_params = yaml.safe_load(f)

history_time = []
history_actual_d = []
history_actual_s = []

t = 0.0


# === 1. INITIALIZE CUSTOM ENVIRONMENT ===
highway_environment = MergeExitLaneHighway_Environment()
highway_environment.configure(simenv_params['env_params'])
highway_environment.render_mode = "human"

mpc_lateral = MPC_Lateral(
    simenv_params['env_params'],
    simenv_params['vehiclemodel_params'],
    simenv_params['sutlanechanger_mpc_params'],
)

mpc_longitudinal = MPC_Longitudinal(
    simenv_params['env_params'],
    simenv_params['vehiclemodel_params'],
    simenv_params['sutlanechanger_mpc_params'],
)

global_planner = LaneChangePlanner(
    simenv_params['env_params'],
    simenv_params['vehiclemodel_params'],
    simenv_params['sutlanechanger_mpc_params'],
)

maneuver_coordinator = Maneuver_Coordinator(
    mpc_lateral,
    mpc_longitudinal,
    global_planner,
    simenv_params['env_params'],
    simenv_params['vehiclemodel_params'],
    simenv_params['sutlanechanger_mpc_params'],
)

obs, info = highway_environment.reset()

# === 2. RUN THE SIMULATION / EPISODES ===

"""

Submitting the positions of the front and back cars in the leading lane to the longitudinal MPC through the global planner.
At each given time, observe the current states of the lane changer and its surrounding vehicles.
Calculate the best lane to go into by calling functions in the global planner.
Repeat for next 5 time steps.
If for the next 5 time steps, the desired lane remains same, call the find gap function in the global planner. Receive the boundaries for the peri region, as well as the time instance to go for the merge, t.
(Global planner and corridor work interchangeably with a lock-safe observing funciton within them to determine when to hand over to the other.)
Separate corridor generator takes the gap, finds out which vehicles to track for the next t seconds and calculates the middle of that gap, passes it to MPC. If corridor sees car can't reach in time, failsafe activated (activate lateral MPC to head back to original lane center, and goes back to global planner.
Corridor keeps calculating the peri region and its middle for the MPC to use in the next step.
Once reaches within acceptable error from the gap center and if still before or on the chosen time-instance, activate MPC lateral as well and once reaches within acceptable error from the gap center, change the longitudinal gap info to be that of the new gap. Reactivate global planner.


"""

done = False
truncated = False

try:
    while not (done or truncated):

        multi_agent_actions = []
        
        for i, agent_obs_matrix in enumerate(obs):
            agent_self_state = agent_obs_matrix[0]
            current_agent_s = agent_self_state[1]       # longitudinal distance
            current_agent_d = agent_self_state[2]       # lateral distance
            current_agent_vx = agent_self_state[3]      # longitudinal velocity
            current_agent_vy = agent_self_state[4]      # lateral velocity
            current_agent_psi = agent_self_state[5]     # heading angle

            current_agent_v = current_agent_vx * np.cos(current_agent_psi) + current_agent_vy * np.sin(current_agent_psi)
            
            obstacle_matrix = agent_obs_matrix[1:]

            # CAR 0: SUT LANE CHANGER (Heading to exit)
            if i == 0:
                x0_extracted_structure = sutlanechanger_mpc.mpc.x0

                x0_extracted_structure['s'] = current_agent_s
                x0_extracted_structure['d'] = current_agent_d
                x0_extracted_structure['psi'] = current_agent_psi
                x0_extracted_structure['v'] = current_agent_v

                sutlanechanger_mpc.update_targets_and_obstacles_in_sut_mpc(
                    target_d=-4.0,
                    target_v=simenv_params['sutlanechanger_mpc_params']['target_v'],
                    observation_matrix=obstacle_matrix
                )

                # CALCULATE NEXT OPTIMAL STEP USING MPC

                u0 = sutlanechanger_mpc.make_step(x0_extracted_structure)
                u0_extracted_structure = sutlanechanger_mpc.mpc.u0

                # EXTRACTING STEPS FOR RENDERING
                steering_rad = float(u0_extracted_structure['steering'])
                accel_ms2 = float(u0_extracted_structure['accel'])

                # RECORDING DISTANCE FOR LATER PLOTTING

                history_actual_d.append(current_agent_d)
                history_actual_s.append(current_agent_s)

            # ADVERSARIAL PLATOON OF CRUISERS (Staying on highway)

            else:
                x0_extracted_structure = sutlanechanger_mpc.mpc.x0 # change this later
                
                x0_extracted_structure['s'] = current_agent_s
                x0_extracted_structure['d'] = current_agent_d
                x0_extracted_structure['psi'] = current_agent_psi
                x0_extracted_structure['v'] = current_agent_v

                sutlanechanger_mpc.update_targets_and_obstacles_in_sut_mpc( # change this later too
                    target_d = current_agent_d, # Tell them to stay in whatever lane they spawned in!
                    target_v = 25.0,          # Standard highway cruising speed
                    observation_matrix = obstacle_matrix
                )

                u0 = sutlanechanger_mpc.make_step(x0_extracted_structure) # change this one as well
                u0_extracted_structure = sutlanechanger_mpc.mpc.u0 # change this one last

                steering_rad = float(u0_extracted_structure['steering'])
                accel_ms2 = float(u0_extracted_structure['accel'])


            normalized_accel = accel_ms2 / simenv_params['vehiclemodel_params']['max_long_accel_ms2']
            normalized_steer = steering_rad / simenv_params['vehiclemodel_params']['max_steer_rad']

            agent_action = np.array([normalized_accel, normalized_steer], dtype=np.float32)
            multi_agent_actions.append(agent_action)
        
        obs, reward, done, truncated, info = highway_environment.step(tuple(multi_agent_actions)) # converting into tuple

        highway_environment.render()
        #time.sleep(0.05)

        history_time.append(t)
        t += 0.1

except KeyboardInterrupt:
    print("\n[INFO] You pressed Ctrl+C! Stopping the simulation and generating the graph...")


# === 3. PLOTTING RESULTS ===

# plt.title('Highway Environment')
# plt.imshow(highway_environment.render())

plt.figure(figsize=(10, 5))
plt.plot(history_time, history_actual_d, 'b-', label='Lateral Distance (d)')
plt.title('Lateral Distance vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Lateral Distance (m)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history_time, history_actual_s, 'r-', label='Longitudinal Distance (s)')
plt.title('Longitudinal Distance vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Longitudinal Distance (m)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history_actual_s, history_actual_d, 'g-', label='Trajectory')
plt.title('Trajectory')
plt.xlabel('Longitudinal Distance (m)')
plt.ylabel('Lateral Distance (m)')
plt.legend()
plt.grid(True)
plt.show()
