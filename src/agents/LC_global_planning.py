import numpy as np

class LaneChangePlanner:
    # Ego makes predictions on trajectories of platoon and where they'll each be at every time instance of the pred. horizon
    def __init__(self, env_params, vehiclemodel_params, sutlanechanger_mpc_params):

        # ==== ENVIRONMENT PARAMETERS ====

        self.observed_vehicles_count = env_params['observation']['observation_config']['vehicles_count']
        self.lane_width_m = env_params['lane_width_m']
        self.lanes_count = env_params['lanes_count']
        self.ends_m = env_params['ends_m']

        self.target_s = sum(self.ends_m) - self.ends_m[-1]

        # ==== LC VEHICLE CONTROLLER PARAMS ====

        self.horizon_N = sutlanechanger_mpc_params['horizon_N']
        self.dt = sutlanechanger_mpc_params['dt']
        self.time_horizon =  self.horizon_N * self.dt

        self.nonzerodivparam_avetravtimeperlane = sutlanechanger_mpc_params['nonzerodivparam_avetravtimeperlane']
        self.scalingparam_avetimegapdensityperlane = sutlanechanger_mpc_params['scalingparam_avetimegapdensityperlane']
        self.urgencyfactor_urgency = sutlanechanger_mpc_params['urgencyfactor_urgency'] 

        # quantifying the benefit of a lane change into the next lane (i.e., higher utility in the next lane means initiation!)
        self.utilweight_avetravtimeperlane = sutlanechanger_mpc_params['utilweight_avetravtimeperlane']
        self.utilweight_avetimegapdensityperlane = sutlanechanger_mpc_params['utilweight_avetimegapdensityperlane']
        self.utilweight_remainingtravtime = sutlanechanger_mpc_params['utilweight_remainingtravtime']                  
        self.utilweight_urgency = sutlanechanger_mpc_params['utilweight_urgency']

        self.final_utilweight_factor = sutlanechanger_mpc_params['final_utilweight_factor']

        self.min_time_gap = sutlanechanger_mpc_params['min_time_gap']
        self.safety_dist_buffer = sutlanechanger_mpc_params['safety_dist_buffer']

        # ==== LC VEHICLE MODEL PARAMS ====

        self.max_long_accel_ms2 = vehiclemodel_params['max_long_accel_ms2']
        self.min_long_accel_ms2 = vehiclemodel_params['min_long_accel_ms2']
        self.max_long_vel_ms = vehiclemodel_params['max_long_vel_ms']


    # First, find out the desired lane at each instance, and output the desired lane. If desired lane persists for five consecutive instances, pass onto calculating the desired gap. (This will be done in the main file. Where to get current_v_ref and current_desiredtimegap values, still unknown.)

    def determine_desired_lane(self, observation_matrix, current_v, current_v_ref, current_desiredtimegap):
        lane_utilities, ego_observation, extracted_vehicle_observations = self._calculate_neighboring_lane_utilities(observation_matrix, current_v_ref, current_desiredtimegap)

        final_lane_utilities = [lane_utilities[u_idx] - (1 + self.final_utilweight_factor * abs(u_idx - 1)) * abs(lane_utilities[1]) for u_idx in range(len(lane_utilities))]

        target_lane_idx = final_lane_utilities.index(max(final_lane_utilities))

        return target_lane_idx, ego_observation, extracted_vehicle_observations, current_v, current_v_ref


    def calculate_desired_gap(self, target_lane_idx, ego_observation, extracted_vehicle_observations, current_v, current_v_ref):

        ego_current_dist = ego_observation[1]
        ego_current_v = ego_observation[3]

        target_lane_vehicle_observations = extracted_vehicle_observations[target_lane_idx]

        target_inf_leader = [0, float('inf'), (int(ego_observation[2] // self.lane_width_m) + (target_lane_idx - 1)) * self.lane_width_m, ego_observation[3], 0, 0]
        target_inf_follower = [0, -float('inf'), (int(ego_observation[2] // self.lane_width_m) + (target_lane_idx - 1)) * self.lane_width_m, ego_observation[3], 0, 0]

        target_lane_vehicle_observations.extend([target_inf_follower, target_inf_leader])

        ego_lane_vehicle_observations = extracted_vehicle_observations[1]

        sorted_target_lane_vehicle_observations_by_pos = sorted(target_lane_vehicle_observations, key=lambda vehicle_observation:vehicle_observation[1])
    
        ego_lane_leader_observations = [vehicle_observation for vehicle_observation in ego_lane_vehicle_observations if vehicle_observation[1] > ego_observation[1]]
        ego_lane_follower_observations = [vehicle_observation for vehicle_observation in ego_lane_vehicle_observations if vehicle_observation[1] < ego_observation[1]]

        ego_inf_leader = [0, float('inf'), ego_observation[2], ego_observation[3], 0, 0]
        ego_inf_follower = [0, -float('inf'), ego_observation[2], ego_observation[3], 0, 0]

        ego_nearest_leader = min(ego_lane_leader_observations, key=lambda vehicle_observation:vehicle_observation[1], default=ego_inf_leader)
        ego_nearest_follower = max(ego_lane_follower_observations, key=lambda vehicle_observation:vehicle_observation[1], default=ego_inf_follower)

        if target_lane_idx == 1:
            return ego_nearest_follower[1], ego_nearest_leader[1], None

        target_farthest_follower_in_range_idx = 0
        for i in range(len(sorted_target_lane_vehicle_observations_by_pos)):
            if sorted_target_lane_vehicle_observations_by_pos[i][1] <= ego_nearest_follower[1]:
                target_farthest_follower_in_range_idx = i
            else:
                break

        target_farthest_leader_in_range_idx = len(sorted_target_lane_vehicle_observations_by_pos) - 1
        for i in range(target_farthest_follower_in_range_idx, len(sorted_target_lane_vehicle_observations_by_pos)):
            if sorted_target_lane_vehicle_observations_by_pos[i][1] > ego_nearest_leader[1]:
                target_farthest_leader_in_range_idx = i - 1
                break

        gap_info = []

        for i in range(target_farthest_follower_in_range_idx, target_farthest_leader_in_range_idx):
            gap_info.append(
                self._calculate_gap_utility(
                    sorted_target_lane_vehicle_observations_by_pos[i], 
                    sorted_target_lane_vehicle_observations_by_pos[i + 1],
                    ego_nearest_leader,
                    ego_nearest_follower,
                    ego_current_dist,
                    current_v,
                    current_v_ref
                )
            )
        
        best_gap_info = max(gap_info, key=lambda gap:gap[0])
        best_time_instance = best_gap_info[1]
        gap_info_idx = gap_info.index(best_gap_info)
        best_gap_follower_pos = sorted_target_lane_vehicle_observations_by_pos[target_farthest_follower_in_range_idx + gap_info_idx][1]
        best_gap_leader_pos = sorted_target_lane_vehicle_observations_by_pos[target_farthest_follower_in_range_idx + gap_info_idx + 1][1]

        return best_gap_follower_pos, best_gap_leader_pos, best_time_instance

    def _calculate_gap_utility(self, target_following_vehicle_obs, target_leading_vehicle_obs, ego_following_vehicle_obs, ego_leading_vehicle_obs, ego_initial_dist, current_v, current_v_ref):

        gap_utility = 0
        best_time_instance = None
        min_displacement_err = float('inf')

        for i in range(self.horizon_N):

            # INTRA: Ego vehicle max and min physical reach

            t = i * self.dt

            if current_v + self.max_long_accel_ms2 * t <= self.max_long_vel_ms:
                max_physical_reach_m = ego_initial_dist + current_v * t + 0.5 * self.max_long_accel_ms2 * t ** 2
            else:
                t_to_max_vel = (self.max_long_vel_ms - current_v) / self.max_long_accel_ms2
                max_physical_reach_m = (ego_initial_dist + current_v * t_to_max_vel + 0.5 * self.max_long_accel_ms2 * t_to_max_vel ** 2) + (t - t_to_max_vel) * self.max_long_vel_ms

            if current_v - self.min_long_accel_ms2 * t >= 0:
                min_physical_reach_m = ego_initial_dist + current_v * t + 0.5 * self.min_long_accel_ms2 * t ** 2
            else:
                t_to_zero_vel = current_v / abs(self.min_long_accel_ms2)
                min_physical_reach_m = (ego_initial_dist + current_v * t_to_zero_vel + 0.5 * self.min_long_accel_ms2 * t_to_zero_vel ** 2)

            # INTER: Ego vehicle allowable reach based on prediction of surrounding cars 
            
            target_max_allowable_reach_m = t * target_leading_vehicle_obs[3] + target_leading_vehicle_obs[1] - self.min_time_gap * min(target_leading_vehicle_obs[3], current_v_ref) + self.safety_dist_buffer
            
            target_min_allowable_reach_m = t * target_following_vehicle_obs[3] + target_following_vehicle_obs[1] + self.min_time_gap * target_following_vehicle_obs[3] + self.safety_dist_buffer

            ego_max_allowable_reach_m = t * ego_leading_vehicle_obs[3] + ego_leading_vehicle_obs[1] - self.min_time_gap * min(ego_leading_vehicle_obs[3], current_v_ref) + self.safety_dist_buffer
            
            ego_min_allowable_reach_m = t * ego_following_vehicle_obs[3] + ego_following_vehicle_obs[1] + self.min_time_gap * ego_following_vehicle_obs[3] + self.safety_dist_buffer

            max_allowable_reach_m = min(target_max_allowable_reach_m, ego_max_allowable_reach_m)
            min_allowable_reach_m = max(target_min_allowable_reach_m, ego_min_allowable_reach_m)

            # COMBINED: Final reach

            physically_limited_max_reach_m = min(max_physical_reach_m, max_allowable_reach_m)
            physically_limited_min_reach_m = max(min_physical_reach_m, min_allowable_reach_m)

            utility_increment = max(0, physically_limited_max_reach_m - physically_limited_min_reach_m)

            gap_utility = gap_utility + utility_increment

            # Check if it would be at this instance that the control error is minimized
            # i.e., When the distance between the gap center and steady cruising position of the LC is minimized (warrants less change in acceleration)

            if utility_increment > 0:
                gap_center_m = (physically_limited_max_reach_m + physically_limited_min_reach_m) / 2
                steady_vel_crusing_pos_m = ego_initial_dist + (current_v * t)

                displacement_err_m = abs(gap_center_m - steady_vel_crusing_pos_m)

                if displacement_err_m < min_displacement_err:
                    min_displacement_err = displacement_err_m
                    best_time_instance = t

        return [gap_utility, best_time_instance]
            

    # ==== HELPER FUNCTIONS ====

    def _calculate_neighboring_lane_utilities(self, observation_matrix, current_v_ref, current_desiredtimegap):

        lane_utilities = []

        ego_vehicle_observation, extracted_vehicle_observations = self._separate_vehicle_observations(observation_matrix)

        same_lane_vehicle_observations = extracted_vehicle_observations[1]
        next_lane_vehicle_observations = extracted_vehicle_observations[0]
        prev_lane_vehicle_observations = extracted_vehicle_observations[2]

        ego_s = observation_matrix[0][1]

        for lane_vehicle_observations in [next_lane_vehicle_observations,
                                            same_lane_vehicle_observations,
                                            prev_lane_vehicle_observations]:
            if lane_vehicle_observations == [] or lane_vehicle_observations is None:
                lane_utilities.append(None)
            else:
                lane_idx = int(lane_vehicle_observations[0][2] // self.lane_width_m)

                utility_avetravtimeperlane = self._calculate_utility_avetravtimeperlane(lane_vehicle_observations, current_v_ref)
                utility_avetimegapdensityperlane = self._calculate_utility_avetimegapdensityperlane(lane_vehicle_observations, current_desiredtimegap)
                utility_remainingtravtime = self._calculate_utility_remainingtravtime(lane_vehicle_observations, ego_s, current_v_ref, self.target_s)
                utility_urgency = self._calculate_utility_urgency(ego_s, self.target_s, lane_idx)

                total_utility = (
                    self.utilweight_avetravtimeperlane * utility_avetravtimeperlane + 
                    self.utilweight_avetimegapdensityperlane * utility_avetimegapdensityperlane + 
                    self.utilweight_remainingtravtime * utility_remainingtravtime + 
                    self.utilweight_urgency * utility_urgency
                )

                lane_utilities.append(total_utility)

        return lane_utilities, ego_vehicle_observation, extracted_vehicle_observations

    def _separate_vehicle_observations(self, observation_matrix):
        same_lane_vehicle_observations = []
        next_lane_vehicle_observations = []
        prev_lane_vehicle_observations = []

        ego_lane_idx = int(observation_matrix[0][2] // self.lane_width_m)
        next_lane_idx = ego_lane_idx - 1
        prev_lane_idx = ego_lane_idx + 1

        if next_lane_idx < 0:
            next_lane_idx = None
            next_lane_vehicle_observations = None

        if prev_lane_idx >= self.lanes_count:
            prev_lane_idx = None
            prev_lane_vehicle_observations = None

        for i in range(1, self.observed_vehicles_count):
            observed_vehicle_lane_idx = int(observation_matrix[i][2] // self.lane_width_m)
            if ego_lane_idx == observed_vehicle_lane_idx:
                same_lane_vehicle_observations.append(observation_matrix[i])
            elif next_lane_idx == observed_vehicle_lane_idx:
                next_lane_vehicle_observations.append(observation_matrix[i])
            elif prev_lane_idx == observed_vehicle_lane_idx:
                prev_lane_vehicle_observations.append(observation_matrix[i])
            
        extracted_vehicle_observations = [next_lane_vehicle_observations, same_lane_vehicle_observations, prev_lane_vehicle_observations] # does not include ego
        ego_vehicle_observation = observation_matrix[0]
        
        return ego_vehicle_observation, extracted_vehicle_observations

    # ==== CALCULATING NORMALIZED UTILITY VALUES PER LANE ====

    # discretionary and anticipatory
    def _calculate_utility_avetravtimeperlane(self, vehicle_observations, current_v_ref):
        # utility of the lane accounts for the mean velocity of all the cars in a lane in seeing if the desired average velocity matches

        if len(vehicle_observations) == 0 or not vehicle_observations:
            return 0
        else:
            vehicle_velocities = [vehicle_observation[3] for vehicle_observation in vehicle_observations]
            avevelocityperlane = sum(vehicle_velocities) / len(vehicle_velocities)

            utility_avetravtimeperlane = -abs(
                self.time_horizon - (self.time_horizon * current_v_ref / max([avevelocityperlane, self.nonzerodivparam_avetravtimeperlane]))
            ) / abs(
                self.time_horizon - (self.time_horizon * current_v_ref / self.nonzerodivparam_avetravtimeperlane)
            )
            
            return utility_avetravtimeperlane

    # discretionary and anticipatory
    def _calculate_utility_avetimegapdensityperlane(self, vehicle_observations, current_desiredtimegap):
        # utility calculates the time gaps for each of the gaps in the lane and takes their average
        
        if len(vehicle_observations) < 2:
            return 1
        else:
            sorted_vehicle_observations_by_pos = sorted(vehicle_observations, key=lambda vehicle_observation:vehicle_observation[1])
            
            time_gaps = []

            for i in range(len(sorted_vehicle_observations_by_pos) - 1):
                follower_s = sorted_vehicle_observations_by_pos[i][1]
                follower_v = sorted_vehicle_observations_by_pos[i][3]
                lead_s = sorted_vehicle_observations_by_pos[i+1][1]

                
                time_gaps.append(min([current_desiredtimegap * self.scalingparam_avetimegapdensityperlane, (lead_s - follower_s) / max(follower_v, 0.1)]))
            
            avetimegapperlane = sum(time_gaps) / len(time_gaps)

            utility_avetimegapdensityperlane = min([current_desiredtimegap * self.scalingparam_avetimegapdensityperlane, avetimegapperlane]) / (current_desiredtimegap * self.scalingparam_avetimegapdensityperlane)
            
            return utility_avetimegapdensityperlane

    # mandatory
    def _calculate_utility_remainingtravtime(self, vehicle_observations, current_s, current_v_ref, target_s):
        # utility calculates how much time is left travellable
        
        utility_remainingtravtime = 0

        if len(vehicle_observations) == 0 or not vehicle_observations:
            utility_remainingtravtime = min(
                self.time_horizon * current_v_ref - current_s, 
                target_s - current_s,
                ) / max(current_v_ref, 0.1) / self.time_horizon
        else:
            utility_remainingtravtime = min(
                self.time_horizon * current_v_ref - current_s, 
                target_s - current_s, 
                max(vehicle_observations, key=lambda vehicle_observation:vehicle_observation[1])[1] - current_s
                ) / max(current_v_ref, 0.1) / self.time_horizon
        return utility_remainingtravtime

    # mandatory
    def _calculate_utility_urgency(self, current_s, target_s, lane_idx):
        dist_to_exit = target_s - current_s

        if dist_to_exit <= 0:
            return 0

        k = (lane_idx + 1)/(self.lanes_count + 1) * self.urgencyfactor_urgency

        utility_urgency = np.exp(-k * dist_to_exit)
        
        return utility_urgency


    def plan_next_step(self, target_d, target_v, observation_matrix):
        observation matrix
        tvp_template_longitudinal + mpc_longitudinal.get.tvp
        tvp_template = self.mpc.get_tvp_template()

        for k in range(self.horizon_N + 1):

            # target distance and velocity do not change (ASSUMED)
            tvp_template['_tvp', k, 'target_d'] = target_d
            tvp_template['_tvp', k, 'target_v'] = target_v

            for i in range(self.observed_vehicles_count):
                # gives no presence (0) to the excess observed vehicles count  
                presence = observation_matrix[i][0] if len(observation_matrix) > i else 0

                if presence == 1:
                    observed_x_current = observation_matrix[i][1]
                    observed_y_current = observation_matrix[i][2]
                    observed_vx_current = observation_matrix[i][3]
                    observed_vy_current = observation_matrix[i][4]

                    # POSITION PREDICTION FOR EACH CAR

                    predicted_x = observed_x_current + observed_vx_current * (k * self.dt)
                    predicted_y = observed_y_current + observed_vy_current * (k * self.dt)

                    tvp_template['_tvp', k, f'obs_{i}_x'] = predicted_x
                    tvp_template['_tvp', k, f'obs_{i}_y'] = predicted_y

                else:
                    # set safety ellipse really far back if car is of no presence
                    tvp_template['_tvp', k, f'obs_{i}_x'] = -1000.0
                    tvp_template['_tvp', k, f'obs_{i}_y'] = -1000.0

        self.current_tvp = tvp_template