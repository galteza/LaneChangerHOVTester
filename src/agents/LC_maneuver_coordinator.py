class Maneuver_Coordinator:
    def __init__(self, mpc_lat, mpc_long, planner, env_params, vehiclemodel_params, sutlanechanger_mpc_params):
        self.mpc_long = mpc_long
        self.mpc_lat = mpc_lat
        self.planner = planner

        self.persistence_counter = 0 
        self.persistence_quota_N = sutlanechanger_mpc_params['persistence_quota_N']
        
        self.target_lane_idx = None
        
        self.state = "FOLLOWING" # FOLLOWING, ALIGNING, INTRUDING, ABORTING


    def run_step(self, observation):
        # Update persistence and check if we should start a move
        self._update_strategic_intent(observation)
        
        # Execute physics based on the current state
        if self.state == "FOLLOWING":
            return self.handle_following(observation)
        elif self.state == "ALIGNING":
            return self.handle_alignment(observation)
        elif self.state == "INTRUDING":
            return self.handle_intrusion(observation)

    def _update_strategic_intent(self, observation):
        
        if self.state == "ALIGNING" or self.state == "INTRUDING":
            # if gap is gone, self.state = "ABORT"  

        target_lane_idx, ego_observation, extracted_vehicle_observations, current_v_ref = self.planner.determine_desired_lane(observation_matrix, current_v_ref, current_desiredtimegap)
        
        ego_lane = int(ego_observation[2] // self.planner.lane_width_m)

        if target_lane_idx == self.target_lane_idx:
            self.persistence_quota_N = self.persistence_quota_N + 1
        else:
            self.target_lane_idx = target_lane_idx
            self.persistence_counter = 0
            self.state = "FOLLOWING"

        if self._check_persistence() and self.state == "FOLLOWING":
            self.state = "ALIGNING"
            self.active_target_lane_idx = self.target_lane_idx
            print(f"Lane intrusion in motion, moving to lane {self.active_target_lane_idx}")
        
        if self.state == "ALIGNING":
            # if within the peri region already (with buffer), self.state = "INTRUDING"
    
    def _check_persistence():
        if self.persistence_counter >= self.persistence_quota_N:
            return True
        elif self.persistence_counter < self.persistence_quota_N:
            return False

    def _handle_following(self):
        # pass the position of the front car and the back to do lane keeping in the middle
        return

    def _handle_alignment(self):
        # pass the borders of the peri region
        return

    def _handle_intruding(self):
        # begin lateral movement, aligning
        return

    def _handle_aborting(self):
        # call previous to go back to original lane or slow down
        return
    