from highway_env.vehicle.controller import ControlledVehicle
from highway_env.utils import Vector

class LaneChanger():
    """
    A lane changing vehicle modeled off of 
    """

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0, predition_type: str = "constant_steering"):
        super().__init__(road, position, heading, speed)
        self.prediction_type = predition_type
        self.action = {"steering": 0, "acceleration": 0}
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)