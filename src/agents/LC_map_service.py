from env.highway_env_mergeexit import MergeExitLaneHighway_Environment
from agents.LC_maneuver_coordinator import Maneuver_Coordinator

class MapService:
    """
    The role of the map service is to survey the environment for its boundaries and movable areas, and taking
    its intersection with the pre, peri, and post regions of the predicted trajectories
    """

    def __init__(self, environment, sutlanechanger_mpc):

        self.env = environment
    
    def get_borders_for_timehorizon(target_pos, 
        self.env.get_borders_for_timehorizon()

        

    def _get_borders(target_longitudinal_s):

generate the pre, peri, and post regions based on the predicted trajectories of the nearby cars

take the borders for each of predicted distances for each instance in the time horizon
