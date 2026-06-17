# === LIBRARIES ===
import numpy as np
import time
from matplotlib import pyplot as plt
import yaml

import stable_baselines3 import SAC

# === CUSTOM FILES ===

from src.agents.decoupled_lanechanger_mpc import DecoupledLaneChanger_MPC
from src.env.highway_env_mergeexit import MergeExitLaneHighway_Environment
from src.agents.lc_corridor_generator import LaneChangePlanner
from src.agents.mpc_lateral import MPC_Lateral
from src.agents.mpc_longitudinal import MPC_Longitudinal
from src.agents.maneuver_coordinator import Maneuver_Coordinator

# === 0. PREPARING CONFIGS & ARRAYS FOR GRAPH DISPLAY ===

params_path = "configs/params_main.yaml"

with open(params_path, "r") as f:
    params = yaml.safe_load(f)

history_time = []
history_actual_d = []
history_actual_s = []

t = 0.0

# === 1. INITIALIZE CUSTOM ENVIRONMENT ===

highway_environment = MergeExitLaneHighway_Environment()
highway_environment.configure(params['env_params'])
highway_environment.render_mode = "human"

mpc_lateral = MPC_Lateral(
    params['env_params'],
    params['vehiclemodel_params'],
    params['sutlanechanger_mpc_params'],
)

mpc_longitudinal = MPC_Longitudinal(
    params['env_params'],
    params['vehiclemodel_params'],
    params['sutlanechanger_mpc_params'],
)

planner = LaneChangePlanner(
    params['env_params'],
    params['vehiclemodel_params'],
    params['sutlanechanger_mpc_params'],
)

mapper = MapService(
    highway_environment
)

maneuver_coordinator = Maneuver_Coordinator(
    mpc_lateral,
    mpc_longitudinal,
    planner,
    params['env_params'],
    params['vehiclemodel_params'],
    params['sutlanechanger_mpc_params'],
)

model = SAC("MlpPolicy", highway_environment, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_highwayenv")

model = SAC.load("sac_highwayenv")

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

target_lane_idx, ego_observation, extracted_vehicle_observations, current_v_ref = global_planner.determine_desired_lane(obs, v_ref, desired_time_gap)

maneuver_coordinator.update_state(target_lane_idx)

if maneuver_coordinator.state == "ALIGNING":
    gap_follower, gap_leader, t_peri = global_planner.calculate_desired_gap(target_lane_idx, ego_observation, extracted_vehicle_observations, current_v_ref)

if maneuver_coordinator.state == "ALIGNING" OR maneuver_coordinator.state == "INTRUDING":
    if current_sim_time > (maneu)



try:
    while not (done or truncated):

        multi_agent_actions = []

        """
        Obs. matrix is organized as follows:

        Vehicle             x       y       vx      vy
        ego-vehicle (0)     5.0     4.0     15.0    0
        vehicle 1           -10.0   4.0     12.0    0
        vehicle 2           13.0    8.0     13.5    0
        ...
        vehicle V           22.2    10.5    18.0    0.5

        """
        

        # Looping for the ego vehicle

        ego_obsermat = obs[0]
        ego_selfobser = ego_obsermat[0]
        ego_obstacmat = ego_obsermat[1:]

        global_planner.determine_desired_lane(obs, v_ref, desired_time_gap)
        # check the lane utility (average speed in each lane, average time gap in each lane, average travellable distance till end of the road, how far away lane is from target)
        
        # check each of the available gaps in target lane is good
        # take the longitudinal position of the followers and find the one that limits the range
        # take the longitudinal position of the leaders and find the one that limits the range
        # limits define gap a
        # take the maximum reachable position and minimum reachable position of the ego --> gap b
        # take the intersection of gap a and gap b 


        # LONGITUDINAL MOVEMENT (necessary)
        long_x0templ = mpc_longitudinal.x0
        long_x0templ = mpc_longitudinal.u0
        long_tvptempl = mpc_longitudinal.get_tvp_template()


        # CONDITIONAL LATERAL MOVEMENT
        if maneuver_coordinator.state == "FOLLOWING":
            # follow leading vehicle safely as usual

            # calculate the safe distance and velocity to follow below 
            long_x0templ['s'] = velocity 
            long_x0templ['v'] = velocity of leading vehicle

            if 5 time instances have passed and still exists,
                maneuver_coordinator.state = "ALIGNING"
            
        elif maneuver_coordinator.state == "ALIGNING":
            # means that the vehicle has overridden the following of the leading vehicle and is currently or about to
            # be positioning itself within the peri region longitudinal range in the prev lane

            long_x0templ = mpc_lateral.mpc.x0
            lat_x0templ = mpc_lateral.mpc.x0


            lat_x0templ['d'] = 
            lat_x0templ['psi'] = 

            lat_u0 = mpc_lateral.make_step(lat_x0templ)
            if suddenly gap is gone
                maneuver_coordinator.state = "ABORTING"
        elif maneuver_coordinator.state == "INTRUDING":
            # means that the vehicle has reached within the peri region range and can now begin lateral movement
            if suddenly
        elif maneuver_coordinator.state == "ABORTING":
             take previous lane and realign there, try to keep safe distance 


        cur_agent_s


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
                    target_v=params['sutlanechanger_mpc_params']['target_v'],
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
                # use the pure pursuit controller provided by highway_env
                
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


            normalized_accel = accel_ms2 / params['vehiclemodel_params']['max_long_accel_ms2']
            normalized_steer = steering_rad / params['vehiclemodel_params']['max_steer_rad']

            agent_action = np.array([normalized_accel, normalized_steer], dtype=np.float32)
            multi_agent_actions.append(agent_action)
        
        # step

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
