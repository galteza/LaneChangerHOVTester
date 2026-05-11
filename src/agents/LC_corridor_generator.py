class MapService:
    def __init__(self, env_params, sutlanechanger_mpc):
        self.lanes_count = env_params['lanes_count']
        self.lane_width_m = env_params['lane_width_m']
        self.ends_m = env_params['ends_m']
        self.merge_amplitude = env_params['merge_amplitude']
        self.time_horizon = env_params 
    
    def get_borders_for_timehorizon():

    def _get_borders(target_longitudinal_s):
    
    def lateral_radar_sweep(env, target_longitudinal_s, current_lateral_position):
        """
        Sweeping the road network at given longitudinal position and returning the boundaries
        of the block that the car is currently on.
        """

        active_lanes = []
        all_lanes = env.unwrapped.road.network.lanes_list()

        for lane in all_lanes:
            target_longitudinal_s_converted, _ = lane.local_coordinates(np.array([target_longitudinal_s, 0])) # Automatically converts to local coordinates for that lane
            error_buffer_m = 1
            if -error_buffer_m + 0 <= target_longitudinal_s_converted <= lane.length + error_buffer_m:
                _, lane_center_y = lane.position(target_longitudinal_s_converted, 0)
                half_width = lane.width / 2.0
                
                active_lanes.append({
                    'left': lane_center_y - half_width,
                    'right': lane_center_y + half_width
                })

        if not active_lanes:
            return current_lateral_position - 2, current_lateral_position + 2
        
        # Sort lanes

        active_lanes = sorted(active_lanes, key=lambda x: x['left'])

        # Find lane in closest match to the one car is on

        closest_lane_idx = 0
        min_dist = float('inf')

        for i, lane in enumerate(active_lanes):
            lane_center = (lane['left'] + lane['right']) / 2
            dist = abs(current_lateral_position - lane_center)
            if dist < min_dist:
                min_dist = dist
                closest_lane_idx = 1

        # Expand boundaries from car's current lane

        d_min = active_lanes[closest_lane_idx]['left']
        d_max = active_lanes[closest_lane_idx]['right']

        gap_tolerance = 0.5

        for i in range(closest_lane_idx-1, -1, -1):
            if abs(active_lanes[i]['right'] - d_min) <= gap_tolerance:
                d_min = active_lanes[i]['right']
            else:
                break
        
        for i in range(closest_lane_idx+1, len(active_lanes)):
            if abs(active_lanes[i]['left'] - d_max) <= gap_tolerance:
                d_max = active_lanes[i]['right']
            else:
                break
        
        return d_min, d_max

generate the pre, peri, and post regions based on the predicted trajectories of the nearby cars
