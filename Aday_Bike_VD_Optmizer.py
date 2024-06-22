import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define initial hardpoints for the bicycle (example values)
initial_hardpoints = {
    'front_hub': [0.0, 0.0, 0.0],
    'rear_hub_left': [0.05, -1.0, 0.0],
    'rear_hub_right': [-0.05, -1.0, 0.0],
    'bottom_bracket': [0.0, -0.5, 0.3],
    'seat_stay_intersection': [0.0, -0.5, 0.8],
    'chain_stay_intersection': [0.0, -0.5, 0.3],
    'top_tube_intersection': [0.0, -0.1, 0.8],
    'head_tube': [0.0, 0.0, 0.8],
    'crank': [0.0, -0.5, 0.3]
}

# Initial tube thickness (example values in meters)
initial_thickness = {
    'top_tube': 0.01,
    'down_tube': 0.01,
    'seat_stay': 0.01,
    'chain_stay': 0.01,
    'head_tube': 0.01
}

# Bicycle and component dimensions
tire_radius = 0.35  # meters
tire_width = 0.1  # meters (wheel width)
crank_length = 0.175  # meters

# Material properties
material_properties = {
    'youngs_modulus': 210e9,  # Pa
    'density': 7850,  # kg/m^3
    'yield_strength': 250e6  # Pa
}

# Define bounds for the optimization (example values)
bounds = [
    (0.0, 0.0), (-0.1, 0.1), (-0.1, 0.1),   # front_hub
    (0.04, 0.06), (-1.1, -0.9), (0.0, 0.0),  # rear_hub_left
    (-0.06, -0.04), (-1.1, -0.9), (0.0, 0.0),  # rear_hub_right
    (0.0, 0.0), (-0.6, -0.4), (0.2, 0.4),   # bottom_bracket
    (0.0, 0.0), (-0.6, -0.4), (0.7, 0.9),   # seat_stay_intersection
    (0.0, 0.0), (-0.6, -0.4), (0.2, 0.4),   # chain_stay_intersection
    (0.0, 0.0), (-0.2, 0.0), (0.7, 0.9),    # top_tube_intersection
    (0.0, 0.0), (-0.2, 0.0), (0.7, 0.9),    # head_tube
    (0.0, 0.0), (-0.6, -0.4), (0.2, 0.4),   # crank
    (0.005, 0.02),  # top_tube thickness
    (0.005, 0.02),  # down_tube thickness
    (0.005, 0.02),  # seat_stay thickness
    (0.005, 0.02),  # chain_stay thickness
    (0.005, 0.02)   # head_tube thickness
]

# Flatten the initial guess for the optimizer
initial_guess = np.array([value for points in initial_hardpoints.values() for value in points] +
                         list(initial_thickness.values()))

# Function to calculate frame deflection and stress based on hardpoints and tube thickness
def calculate_deflection_and_stress(hardpoints, thickness, material_properties):
    youngs_modulus = material_properties['youngs_modulus']
    density = material_properties['density']
    yield_strength = material_properties['yield_strength']
    
    # Calculate lengths of frame members
    top_tube_length = np.linalg.norm(np.array(hardpoints['head_tube']) - np.array(hardpoints['top_tube_intersection']))
    down_tube_length = np.linalg.norm(np.array(hardpoints['head_tube']) - np.array(hardpoints['bottom_bracket']))
    seat_stay_length_left = np.linalg.norm(np.array(hardpoints['seat_stay_intersection']) - np.array(hardpoints['rear_hub_left']))
    seat_stay_length_right = np.linalg.norm(np.array(hardpoints['seat_stay_intersection']) - np.array(hardpoints['rear_hub_right']))
    chain_stay_length_left = np.linalg.norm(np.array(hardpoints['chain_stay_intersection']) - np.array(hardpoints['rear_hub_left']))
    chain_stay_length_right = np.linalg.norm(np.array(hardpoints['chain_stay_intersection']) - np.array(hardpoints['rear_hub_right']))
    head_tube_length = np.linalg.norm(np.array(hardpoints['head_tube']) - np.array(hardpoints['front_hub']))
    
    # Calculate moment of inertia for each tube
    I_top_tube = np.pi * (thickness['top_tube'] / 2)**4 / 4
    I_down_tube = np.pi * (thickness['down_tube'] / 2)**4 / 4
    I_seat_stay = np.pi * (thickness['seat_stay'] / 2)**4 / 4
    I_chain_stay = np.pi * (thickness['chain_stay'] / 2)**4 / 4
    I_head_tube = np.pi * (thickness['head_tube'] / 2)**4 / 4

    # Calculate deflection (simplified model)
    deflection_top_tube = (top_tube_length**3) / (3 * youngs_modulus * I_top_tube)
    deflection_down_tube = (down_tube_length**3) / (3 * youngs_modulus * I_down_tube)
    deflection_seat_stay_left = (seat_stay_length_left**3) / (3 * youngs_modulus * I_seat_stay)
    deflection_seat_stay_right = (seat_stay_length_right**3) / (3 * youngs_modulus * I_seat_stay)
    deflection_chain_stay_left = (chain_stay_length_left**3) / (3 * youngs_modulus * I_chain_stay)
    deflection_chain_stay_right = (chain_stay_length_right**3) / (3 * youngs_modulus * I_chain_stay)
    deflection_head_tube = (head_tube_length**3) / (3 * youngs_modulus * I_head_tube)

    total_deflection = (deflection_top_tube + deflection_down_tube + 
                        deflection_seat_stay_left + deflection_seat_stay_right + 
                        deflection_chain_stay_left + deflection_chain_stay_right + deflection_head_tube)

    # Calculate stress (simplified model)
    stress_top_tube = (youngs_modulus * deflection_top_tube) / (yield_strength * top_tube_length)
    stress_down_tube = (youngs_modulus * deflection_down_tube) / (yield_strength * down_tube_length)
    stress_seat_stay_left = (youngs_modulus * deflection_seat_stay_left) / (yield_strength * seat_stay_length_left)
    stress_seat_stay_right = (youngs_modulus * deflection_seat_stay_right) / (yield_strength * seat_stay_length_right)
    stress_chain_stay_left = (youngs_modulus * deflection_chain_stay_left) / (yield_strength * chain_stay_length_left)
    stress_chain_stay_right = (youngs_modulus * deflection_chain_stay_right) / (yield_strength * chain_stay_length_right)
    stress_head_tube = (youngs_modulus * deflection_head_tube) / (yield_strength * head_tube_length)

    total_stress = (stress_top_tube + stress_down_tube + 
                    stress_seat_stay_left + stress_seat_stay_right + 
                    stress_chain_stay_left + stress_chain_stay_right + stress_head_tube)

    return total_deflection, total_stress

# Objective function to minimize
def objective(params):
    # Extract hardpoints and thickness from the parameter list
    hardpoints_flat = params[:len(initial_guess)-5]
    thickness_flat = params[len(initial_guess)-5:]
    
    # Reshape the flat lists back into dictionaries
    hardpoints = {
        'front_hub': hardpoints_flat[0:3],
        'rear_hub_left': hardpoints_flat[3:6],
        'rear_hub_right': hardpoints_flat[6:9],
        'bottom_bracket': hardpoints_flat[9:12],
        'seat_stay_intersection': hardpoints_flat[12:15],
        'chain_stay_intersection': hardpoints_flat[15:18],
        'top_tube_intersection': hardpoints_flat[18:21],
        'head_tube': hardpoints_flat[21:24],
        'crank': hardpoints_flat[24:27]
    }
    thickness = {
        'top_tube': thickness_flat[0],
        'down_tube': thickness_flat[1],
        'seat_stay': thickness_flat[2],
        'chain_stay': thickness_flat[3],
        'head_tube': thickness_flat[4]
    }
    
    # Calculate deflection and stress
    deflection, stress = calculate_deflection_and_stress(hardpoints, thickness, material_properties)
    
    # Objective to minimize deflection and stress
    return deflection + stress

# Perform the optimization
result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

# Reshape the optimized hardpoints and thickness back into dictionaries
optimized_hardpoints = {
    'front_hub': result.x[0:3],
    'rear_hub_left': result.x[3:6],
    'rear_hub_right': result.x[6:9],
    'bottom_bracket': result.x[9:12],
    'seat_stay_intersection': result.x[12:15],
    'chain_stay_intersection': result.x[15:18],
    'top_tube_intersection': result.x[18:21],
    'head_tube': result.x[21:24],
    'crank': result.x[24:27]
}
optimized_thickness = {
    'top_tube': result.x[27],
    'down_tube': result.x[28],
    'seat_stay': result.x[29],
    'chain_stay': result.x[30],
    'head_tube': result.x[31]
}

# Function to plot bicycle frame geometry
def plot_bicycle_frame(ax, hardpoints, title):
    ax.plot([hardpoints['front_hub'][0], hardpoints['head_tube'][0]],
            [hardpoints['front_hub'][1], hardpoints['head_tube'][1]],
            [hardpoints['front_hub'][2], hardpoints['head_tube'][2]], 'b-o', label='Head Tube')
    
    ax.plot([hardpoints['head_tube'][0], hardpoints['top_tube_intersection'][0]],
            [hardpoints['head_tube'][1], hardpoints['top_tube_intersection'][1]],
            [hardpoints['head_tube'][2], hardpoints['top_tube_intersection'][2]], 'g-o', label='Top Tube')
    
    ax.plot([hardpoints['seat_stay_intersection'][0], hardpoints['rear_hub_left'][0]],
            [hardpoints['seat_stay_intersection'][1], hardpoints['rear_hub_left'][1]],
            [hardpoints['seat_stay_intersection'][2], hardpoints['rear_hub_left'][2]], 'c-o', label='Seat Stay Left')
    
    ax.plot([hardpoints['seat_stay_intersection'][0], hardpoints['rear_hub_right'][0]],
            [hardpoints['seat_stay_intersection'][1], hardpoints['rear_hub_right'][1]],
            [hardpoints['seat_stay_intersection'][2], hardpoints['rear_hub_right'][2]], 'm-o', label='Seat Stay Right')
    
    ax.plot([hardpoints['chain_stay_intersection'][0], hardpoints['rear_hub_left'][0]],
            [hardpoints['chain_stay_intersection'][1], hardpoints['rear_hub_left'][1]],
            [hardpoints['chain_stay_intersection'][2], hardpoints['rear_hub_left'][2]], 'y-o', label='Chain Stay Left')
    
    ax.plot([hardpoints['chain_stay_intersection'][0], hardpoints['rear_hub_right'][0]],
            [hardpoints['chain_stay_intersection'][1], hardpoints['rear_hub_right'][1]],
            [hardpoints['chain_stay_intersection'][2], hardpoints['rear_hub_right'][2]], 'y-o', label='Chain Stay Right')
    
    ax.plot([hardpoints['bottom_bracket'][0], hardpoints['chain_stay_intersection'][0]],
            [hardpoints['bottom_bracket'][1], hardpoints['chain_stay_intersection'][1]],
            [hardpoints['bottom_bracket'][2], hardpoints['chain_stay_intersection'][2]], 'orange', label='Down Tube')
    
    ax.plot([hardpoints['bottom_bracket'][0], hardpoints['crank'][0]],
            [hardpoints['bottom_bracket'][1], hardpoints['crank'][1]],
            [hardpoints['bottom_bracket'][2], hardpoints['crank'][2]], 'purple', label='Crank')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)

print("Initial Hardpoints:")
for key, value in initial_hardpoints.items():
    print(f"{key}: {[f'{v:.8f}' for v in value]}")

print("Initial Tube Thickness:")
for key, value in initial_thickness.items():
    print(f"{key}: {value:.8f}")

print("\nOptimized Hardpoints:")
for key, value in optimized_hardpoints.items():
    print(f"{key}: {[f'{v:.8f}' for v in value]}")

print("Optimized Tube Thickness:")
for key, value in optimized_thickness.items():
    print(f"{key}: {value:.8f}")

# Plot initial and optimized bicycle frame geometries
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
plot_bicycle_frame(ax1, initial_hardpoints, 'Initial Bicycle Frame Geometry')

ax2 = fig.add_subplot(122, projection='3d')
plot_bicycle_frame(ax2, optimized_hardpoints, 'Optimized Bicycle Frame Geometry')

plt.tight_layout()
plt.show()
