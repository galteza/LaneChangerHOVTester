import numpy as np

"""
These pieces of code were lifted from the work of Yiru Jiao (Department of Transport & Planning, Delft University of Technology)
Code was modified to switch from the use of ==== to numpy arrays for compatibility with highway-env vehicles.

"""

class PolygonTTCCalculator:
    """
    A high-performance NumPy utility class to compute the precise Time-to-Collision (TTC)
    between two oriented rectangular vehicle bounding boxes.
    """
    
    @staticmethod
    def _line(p0, p1):
        a = p0[1] - p1[1]
        b = p1[0] - p0[0]
        c = p0[0] * p1[1] - p1[0] * p0[1]
        return a, b, c

    @staticmethod
    def _intersect(line0, line1):
        a0, b0, c0 = line0
        a1, b1, c1 = line1
        D = a0 * b1 - a1 * b0
        if abs(D) < 1e-5:
            return np.array([np.nan, np.nan])
        x = (b0 * c1 - b1 * c0) / D
        y = (a1 * c0 - a0 * c1) / D
        return np.array([x, y])

    @staticmethod
    def _ison(line_start, line_end, point):
        if np.isnan(point[0]):
            return False
        crossproduct = (point[1] - line_start[1]) * (line_end[0] - line_start[0]) - (point[0] - line_start[0]) * (line_end[1] - line_start[1])
        if abs(crossproduct) > 1e-5:
            return False
        dotproduct = (point[0] - line_start[0]) * (line_end[0] - line_start[0]) + (point[1] - line_start[1]) * (line_end[1] - line_start[1])
        squaredlength = (line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2
        return (dotproduct >= 0) and (dotproduct <= squaredlength)

    @classmethod
    def get_bounding_box_corners(cls, x, y, heading, length, width):
        """Calculates the 4 absolute corner points of a vehicle."""
        h_vec = np.array([np.cos(heading), np.sin(heading)])
        perp_h_vec = np.array([-h_vec[1], h_vec[0]])
        
        point_up = np.array([x, y]) + h_vec * (length / 2.0)
        point_down = np.array([x, y]) - h_vec * (length / 2.0)
        
        return [
            point_up + perp_h_vec * (width / 2.0),
            point_up - perp_h_vec * (width / 2.0),
            point_down + perp_h_vec * (width / 2.0),
            point_down - perp_h_vec * (width / 2.0)
        ]

    @classmethod
    def _pairwise_ttc(cls, params_i, params_j):
        """Computes directional ray-cast TTC from Pack I to Pack J."""
        corners_i = cls.get_bounding_box_corners(params_i[0], params_i[1], params_i[4], params_i[5], params_i[6])
        corners_j = cls.get_bounding_box_corners(params_j[0], params_j[1], params_j[4], params_j[5], params_j[6])
        
        v_i = np.array([params_i[2], params_i[3]])
        v_j = np.array([params_j[2], params_j[3]])
        direct_v = v_i - v_j
        
        rel_speed = np.linalg.norm(direct_v)
        if rel_speed < 1e-5:
            return np.inf

        min_dist_ist = np.inf
        valid_collision_course = False
        edges_j = [(corners_j[0], corners_j[1]), (corners_j[2], corners_j[3]), 
                   (corners_j[0], corners_j[2]), (corners_j[1], corners_j[3])]

        for p_start in corners_i:
            p_end = p_start + direct_v
            ray_line = cls._line(p_start, p_end)
            
            for edge_start, edge_end in edges_j:
                edge_line = cls._line(edge_start, edge_end)
                ist = cls._intersect(ray_line, edge_line)
                
                if cls._ison(edge_start, edge_end, ist):
                    leaving_check = direct_v[0] * (ist[0] - p_start[0]) + direct_v[1] * (ist[1] - p_start[1])
                    if leaving_check >= 0:
                        valid_collision_course = True
                        dist_ist = np.linalg.norm(ist - p_start)
                        if dist_ist < min_dist_ist:
                            min_dist_ist = dist_ist

        if not valid_collision_course or min_dist_ist == np.inf:
            return np.inf
            
        return min_dist_ist / rel_speed

    @classmethod
    def compute_ttc(cls, veh_i, veh_j) -> float:
        """
        Public API: Call this to get the symmetrical polygon TTC between two highway-env vehicles.
        """
        # Format: (x, y, vx, vy, heading, length, width)
        # Safely handle environments that use velocity vector arrays vs raw scalar speeds
        v_i = getattr(veh_i, 'velocity', np.array([veh_i.speed * np.cos(veh_i.heading), veh_i.speed * np.sin(veh_i.heading)]))
        v_j = getattr(veh_j, 'velocity', np.array([veh_j.speed * np.cos(veh_j.heading), veh_j.speed * np.sin(veh_j.heading)]))

        params_i = (veh_i.position[0], veh_i.position[1], v_i[0], v_i[1], veh_i.heading, veh_i.LENGTH, veh_i.WIDTH)
        params_j = (veh_j.position[0], veh_j.position[1], v_j[0], v_j[1], veh_j.heading, veh_j.LENGTH, veh_j.WIDTH)
        
        return min(cls._pairwise_ttc(params_i, params_j), cls._pairwise_ttc(params_j, params_i))
    