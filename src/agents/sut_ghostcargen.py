import numpy as np

def generate_ghost_car_target(current_s, start_s, end_s, start_d, end_d):
    """
    Generates a perfectly smooth S-curve using a Cosine function.
    Returns the target lateral position (d) and target heading (psi).
    """
    # If we haven't started the lane change yet, stay in current lane
    if current_s <= start_s:
        return start_d, 0.0 
        
    # If we finished the lane change, lock into the target lane
    if current_s >= end_s:
        return end_d, 0.0 

    # Calculate how far along the lane change we are (0.0 to 1.0)
    progress = (current_s - start_s) / (end_s - start_s)
    
    # Cosine S-Curve formula for lateral position
    target_d = start_d + (end_d - start_d) * 0.5 * (1 - np.cos(np.pi * progress))
    
    # The derivative of the curve gives us the perfect heading angle (psi)
    # We use arctan to convert the lateral slope into a steering angle
    slope = (end_d - start_d) * 0.5 * np.pi * np.sin(np.pi * progress) / (end_s - start_s)
    target_psi = np.arctan(slope)
    
    return target_d, target_psi