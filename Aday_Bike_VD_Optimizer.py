print("Ryan Aday\nBike Optimizer\n")
print("Version 1.0\n")

print("""Provides optimized tube dimensions and angles for a standard bike frame
while maximizing torsional rigidity and fatigue life (Basquin's Law). 

Also prints out tube lengths, torsional rigidity, and fatigue life for
individual tubes and for the bike overall.

NOTE: This uses material properties for 4130 steel. Modify to suit
your own needs.

Provide:
# Parameters (constants):
wheel_diameter # in mm
wheel_thickness  # in mm
rear_center  # in mm (horizontal distance)
front_center # in mm (horizontal distance)
material_properties (constants):
    shear_modulus # MPa for G
    fatigue_strength_coefficient # MPa for sigma_f_prime
    fatigue_strength_exponent # dimensionless for b

# Optimized variables:
Tube dimensions (outer diameter, thickness) in mm
    seat_tube
    top_tube
    head_tube
    down_tube
    seatstay
    fork
seat_tube_length # in mm
top_tube_length # in mm
head_tube_length # in mm
seat_tube_angle # in degrees (referenced from vertical)
head_tube_angle  # in degrees (referenced from vertical)
fork_angle # in degrees (referenced from vertical)
top_tube_angle  # in degrees (referenced from horizontal)
bottom_bracket_drop  # in mm
    
Requires libraries:
    numpy
    matplotlib   
    mpl_toolkits
    scipy
    sys
    \n""")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D, art3d
    from scipy.optimize import differential_evolution
    import sys
except ImportError:
    sys.exit("""
        You need the numpy, mpl_toolkits, scipy, and matplotlib libraries.
        To install these libraries, please enter:
        pip install numpy scipy matplotlib mpl_toolkits
        """)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.optimize import differential_evolution

def calculate_hardpoints(wheel_diameter, wheel_thickness, seat_tube_length, top_tube_length, head_tube_length, seat_tube_angle, head_tube_angle, fork_angle, top_tube_angle, bottom_bracket_drop, rear_center, front_center):
    # Basic parameters
    wheel_radius = wheel_diameter / 2

    # Rear wheel position
    rear_wheel_y = 0
    rear_wheel_z = wheel_radius

    # Bottom bracket position
    bottom_bracket_y = rear_center
    bottom_bracket_z = rear_wheel_z - bottom_bracket_drop

    # Front wheel position
    front_wheel_y = bottom_bracket_y + front_center
    front_wheel_z = wheel_radius

    # Seat tube top
    seat_tube_top_y = bottom_bracket_y + seat_tube_length * np.sin(np.radians(seat_tube_angle))
    seat_tube_top_z = bottom_bracket_z + seat_tube_length * np.cos(np.radians(seat_tube_angle))

    # Head tube bottom
    head_tube_bottom_y = bottom_bracket_y + top_tube_length * np.cos(np.radians(top_tube_angle))
    head_tube_bottom_z = bottom_bracket_z + top_tube_length * np.sin(np.radians(top_tube_angle))

    # Head tube top
    head_tube_top_y = head_tube_bottom_y + head_tube_length * np.sin(np.radians(head_tube_angle))
    head_tube_top_z = head_tube_bottom_z + head_tube_length * np.cos(np.radians(head_tube_angle))

    # Fork starts from the bottom of the head tube
    fork_top_y = head_tube_bottom_y
    fork_top_z = head_tube_bottom_z

    # Adjust fork bottom based on fork angle and front wheel position
    fork_bottom_y = front_wheel_y
    fork_bottom_z = front_wheel_z

    # Wheel thickness offset
    wheel_offset = wheel_thickness / 2

    # Hardpoints
    hardpoints = {
        'rear_wheel_center': np.array([rear_wheel_y, 0, rear_wheel_z]),
        'front_wheel_center': np.array([front_wheel_y, 0, front_wheel_z]),
        'rear_wheel_left': np.array([rear_wheel_y, -wheel_offset, rear_wheel_z]),
        'rear_wheel_right': np.array([rear_wheel_y, wheel_offset, rear_wheel_z]),
        'front_wheel_left': np.array([front_wheel_y, -wheel_offset, front_wheel_z]),
        'front_wheel_right': np.array([front_wheel_y, wheel_offset, front_wheel_z]),
        'bottom_bracket': np.array([bottom_bracket_y, 0, bottom_bracket_z]),
        'seat_tube_top': np.array([seat_tube_top_y, 0, seat_tube_top_z]),
        'head_tube_bottom': np.array([head_tube_bottom_y, 0, head_tube_bottom_z]),
        'head_tube_top': np.array([head_tube_top_y, 0, head_tube_top_z]),
        'fork_bottom_left': np.array([fork_bottom_y, -wheel_offset, fork_bottom_z]),
        'fork_bottom_right': np.array([fork_bottom_y, wheel_offset, fork_bottom_z]),
        'fork_top': np.array([fork_top_y, 0, fork_top_z]),
    }

    return hardpoints

def calculate_moment_of_inertia(outer_diameter, thickness):
    inner_diameter = outer_diameter - 2 * thickness
    I = (np.pi / 64) * (outer_diameter**4 - inner_diameter**4)
    J = (np.pi / 32) * (outer_diameter**4 - inner_diameter**4)
    return I, J

def calculate_tube_length(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)

def plot_bicycle_frame_3d(hardpoints, wheel_diameter, seat_tube_angle, head_tube_angle, fork_angle, top_tube_angle, rear_center, front_center, ax):
    # Draw frame
    # Chainstay and Seat Stays
    for side in ['left', 'right']:
        ax.plot([hardpoints[f'rear_wheel_{side}'][0], hardpoints['bottom_bracket'][0]], 
                [hardpoints[f'rear_wheel_{side}'][1], hardpoints['bottom_bracket'][1]], 
                [hardpoints[f'rear_wheel_{side}'][2], hardpoints['bottom_bracket'][2]], 'k-')
        
        ax.plot([hardpoints['bottom_bracket'][0], hardpoints['seat_tube_top'][0]], 
                [hardpoints['bottom_bracket'][1], hardpoints['seat_tube_top'][1]], 
                [hardpoints['bottom_bracket'][2], hardpoints['seat_tube_top'][2]], 'k-')
        
        ax.plot([hardpoints['seat_tube_top'][0], hardpoints['head_tube_top'][0]], 
                [hardpoints['seat_tube_top'][1], hardpoints['head_tube_top'][1]], 
                [hardpoints['seat_tube_top'][2], hardpoints['head_tube_top'][2]], 'k-')
        
        ax.plot([hardpoints['head_tube_top'][0], hardpoints['head_tube_bottom'][0]], 
                [hardpoints['head_tube_top'][1], hardpoints['head_tube_bottom'][1]], 
                [hardpoints['head_tube_top'][2], hardpoints['head_tube_bottom'][2]], 'k-')
        
        ax.plot([hardpoints['head_tube_bottom'][0], hardpoints['bottom_bracket'][0]], 
                [hardpoints['head_tube_bottom'][1], hardpoints['bottom_bracket'][1]], 
                [hardpoints['head_tube_bottom'][2], hardpoints['bottom_bracket'][2]], 'k-')
        
        ax.plot([hardpoints[f'rear_wheel_{side}'][0], hardpoints['seat_tube_top'][0]], 
                [hardpoints[f'rear_wheel_{side}'][1], hardpoints['seat_tube_top'][1]], 
                [hardpoints[f'rear_wheel_{side}'][2], hardpoints['seat_tube_top'][2]], 'k-')
        
        ax.plot([hardpoints[f'fork_bottom_{side}'][0], hardpoints['head_tube_bottom'][0]], 
                [hardpoints[f'fork_bottom_{side}'][1], hardpoints['head_tube_bottom'][1]], 
                [hardpoints[f'fork_bottom_{side}'][2], hardpoints['head_tube_bottom'][2]], 'k-')

    # Draw wheels in the yz plane
    for wheel in ['front_wheel', 'rear_wheel']:
        wheel_circle = plt.Circle((hardpoints[f'{wheel}_center'][0], hardpoints[f'{wheel}_center'][2]), wheel_diameter / 2, color='grey', fill=False)
        ax.add_patch(wheel_circle)
        art3d.pathpatch_2d_to_3d(wheel_circle, z=hardpoints[f'{wheel}_center'][1], zdir="y")

    # Set labels and limits
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('X (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Bicycle Frame Geometry')
    ax.set_xlim([-200, front_center + rear_center + 200])
    ax.set_ylim([-100, 100])
    ax.set_zlim([0, wheel_diameter + 200])

def calculate_torsional_rigidity_and_fatigue_life(tube_dimensions, material_properties, tube_lengths, sigma_a=100):
    torsional_rigidity = {}
    fatigue_life = {}

    total_torsional_rigidity_inv = 0  # for parallel springs calculation
    total_fatigue_life = 0  # for averaging fatigue life
    num_tubes = len(tube_dimensions)

    for tube, dims in tube_dimensions.items():
        outer_diameter, thickness = dims
        I, J = calculate_moment_of_inertia(outer_diameter, thickness)
        G = material_properties['shear_modulus']
        sigma_f_prime = material_properties['fatigue_strength_coefficient']
        b = material_properties['fatigue_strength_exponent']
        L = tube_lengths[tube]

        torsional_stiffness = G * J / L

        if tube in ['seatstay', 'fork']:
            if tube not in torsional_rigidity:
                torsional_rigidity[tube] = 0
            torsional_rigidity[tube] += torsional_stiffness
        else:
            torsional_rigidity[tube] = torsional_stiffness

        # Fatigue life calculation with provided sigma_a
        N_f = (sigma_a / (sigma_f_prime / 2))**(1 / b)
        fatigue_life[tube] = N_f

        total_fatigue_life += N_f

    # Calculate total torsional rigidity for parallel components
    equivalent_torsional_rigidity = {
        'seatstay': 1 / (1 / torsional_rigidity['seatstay'] + 1 / torsional_rigidity['seatstay']),
        'fork': 1 / (1 / torsional_rigidity['fork'] + 1 / torsional_rigidity['fork'])
    }

    total_torsional_rigidity = sum([torsional_rigidity[tube] for tube in torsional_rigidity if tube not in equivalent_torsional_rigidity])
    total_torsional_rigidity += sum(equivalent_torsional_rigidity.values())

    # Calculate average fatigue life
    overall_fatigue_life = total_fatigue_life / num_tubes

    return torsional_rigidity, fatigue_life, total_torsional_rigidity, overall_fatigue_life

def objective_function(x, *args):
    wheel_diameter, wheel_thickness, rear_center, front_center, tube_names, material_properties = args
    
    # Extract parameters from x
    seat_tube_length, top_tube_length, head_tube_length, seat_tube_angle, head_tube_angle, fork_angle, top_tube_angle, bottom_bracket_drop = x[:8]
    
    tube_dimensions = {
        tube_names[i]: (x[8 + 2*i], x[9 + 2*i]) for i in range(len(tube_names))
    }
    
    hardpoints = calculate_hardpoints(wheel_diameter, wheel_thickness, seat_tube_length, top_tube_length, head_tube_length, seat_tube_angle, head_tube_angle, fork_angle, top_tube_angle, bottom_bracket_drop, rear_center, front_center)

    tube_lengths = {
        'seat_tube': calculate_tube_length(hardpoints['bottom_bracket'], hardpoints['seat_tube_top']),
        'top_tube': calculate_tube_length(hardpoints['seat_tube_top'], hardpoints['head_tube_top']),
        'head_tube': calculate_tube_length(hardpoints['head_tube_top'], hardpoints['head_tube_bottom']),
        'down_tube': calculate_tube_length(hardpoints['head_tube_bottom'], hardpoints['bottom_bracket']),
        'seatstay': calculate_tube_length(hardpoints['rear_wheel_left'], hardpoints['seat_tube_top']),
        'fork': calculate_tube_length(hardpoints['fork_bottom_left'], hardpoints['head_tube_bottom']),
    }

    _, _, total_torsional_rigidity, overall_fatigue_life = calculate_torsional_rigidity_and_fatigue_life(tube_dimensions, material_properties, tube_lengths, sigma_a=100)

    # We want to maximize torsional rigidity and fatigue life
    # Minimize the negative of these values to convert to a minimization problem
    return -total_torsional_rigidity - overall_fatigue_life

# Parameters
wheel_diameter = 700  # in mm
wheel_thickness = 25  # in mm
rear_center = 405  # in mm (horizontal distance)
front_center = 650  # in mm (horizontal distance)

# Tube names
tube_names = ['seat_tube', 'top_tube', 'head_tube', 'down_tube', 'seatstay', 'fork']

# Material properties for 4130 steel (example values)
material_properties = {
    'shear_modulus': 80e3,  # MPa for G
    'fatigue_strength_coefficient': 700,  # MPa for sigma_f_prime
    'fatigue_strength_exponent': -0.08  # dimensionless for b
}

# Initial parameters (example values based on the provided image)
initial_params = [
    500,  # seat_tube_length
    600,  # top_tube_length
    75,   # head_tube_length
    -10,  # seat_tube_angle
    -10,  # head_tube_angle
    30,   # fork_angle
    45,   # top_tube_angle
    70,   # bottom_bracket_drop
    28.6, 1.2,  # seat_tube dimensions (outer_diameter, thickness)
    25.4, 1.2,  # top_tube dimensions (outer_diameter, thickness)
    31.8, 1.5,  # head_tube dimensions (outer_diameter, thickness)
    34.9, 1.2,  # down_tube dimensions (outer_diameter, thickness)
    19.0, 1.0,  # seatstay dimensions (outer_diameter, thickness)
    28.6, 1.2   # fork dimensions (outer_diameter, thickness)
]

# Bounds for the parameters to be optimized
bounds = [
    (400, 600),  # seat_tube_length
    (500, 700),  # top_tube_length
    (50, 100),   # head_tube_length
    (-20, 0),    # seat_tube_angle
    (-20, 0),    # head_tube_angle
    (20, 40),    # fork_angle
    (30, 60),    # top_tube_angle
    (50, 90),    # bottom_bracket_drop
    (20, 40), (1, 2),  # seat_tube dimensions (outer_diameter, thickness)
    (20, 40), (1, 2),  # top_tube dimensions (outer_diameter, thickness)
    (25, 40), (1, 2),  # head_tube dimensions (outer_diameter, thickness)
    (30, 40), (1, 2),  # down_tube dimensions (outer_diameter, thickness)
    (15, 25), (1, 2),  # seatstay dimensions (outer_diameter, thickness)
    (20, 40), (1, 2)   # fork dimensions (outer_diameter, thickness)
]

# Perform optimization
result = differential_evolution(objective_function, bounds, args=(wheel_diameter, wheel_thickness, rear_center, front_center, tube_names, material_properties))

# Extract optimized parameters
optimized_params = result.x

# Calculate initial and optimized hardpoints and properties
initial_hardpoints = calculate_hardpoints(wheel_diameter, wheel_thickness, *initial_params[:8], rear_center, front_center)
optimized_hardpoints = calculate_hardpoints(wheel_diameter, wheel_thickness, *optimized_params[:8], rear_center, front_center)

initial_tube_dimensions = {tube_names[i]: (initial_params[8 + 2*i], initial_params[9 + 2*i]) for i in range(len(tube_names))}
optimized_tube_dimensions = {tube_names[i]: (optimized_params[8 + 2*i], optimized_params[9 + 2*i]) for i in range(len(tube_names))}

initial_tube_lengths = {
    'seat_tube': calculate_tube_length(initial_hardpoints['bottom_bracket'], initial_hardpoints['seat_tube_top']),
    'top_tube': calculate_tube_length(initial_hardpoints['seat_tube_top'], initial_hardpoints['head_tube_top']),
    'head_tube': calculate_tube_length(initial_hardpoints['head_tube_top'], initial_hardpoints['head_tube_bottom']),
    'down_tube': calculate_tube_length(initial_hardpoints['head_tube_bottom'], initial_hardpoints['bottom_bracket']),
    'seatstay': calculate_tube_length(initial_hardpoints['rear_wheel_left'], initial_hardpoints['seat_tube_top']),
    'fork': calculate_tube_length(initial_hardpoints['fork_bottom_left'], initial_hardpoints['head_tube_bottom']),
}

optimized_tube_lengths = {
    'seat_tube': calculate_tube_length(optimized_hardpoints['bottom_bracket'], optimized_hardpoints['seat_tube_top']),
    'top_tube': calculate_tube_length(optimized_hardpoints['seat_tube_top'], optimized_hardpoints['head_tube_top']),
    'head_tube': calculate_tube_length(optimized_hardpoints['head_tube_top'], optimized_hardpoints['head_tube_bottom']),
    'down_tube': calculate_tube_length(optimized_hardpoints['head_tube_bottom'], optimized_hardpoints['bottom_bracket']),
    'seatstay': calculate_tube_length(optimized_hardpoints['rear_wheel_left'], optimized_hardpoints['seat_tube_top']),
    'fork': calculate_tube_length(optimized_hardpoints['fork_bottom_left'], optimized_hardpoints['head_tube_bottom']),
}

initial_torsional_rigidity, initial_fatigue_life, initial_total_torsional_rigidity, initial_overall_fatigue_life = calculate_torsional_rigidity_and_fatigue_life(initial_tube_dimensions, material_properties, initial_tube_lengths, sigma_a=100)
optimized_torsional_rigidity, optimized_fatigue_life, optimized_total_torsional_rigidity, optimized_overall_fatigue_life = calculate_torsional_rigidity_and_fatigue_life(optimized_tube_dimensions, material_properties, optimized_tube_lengths, sigma_a=100)

# Plot initial and optimized frames
fig = plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(121, projection='3d')
plot_bicycle_frame_3d(initial_hardpoints, wheel_diameter, initial_params[3], initial_params[4], initial_params[5], initial_params[6], rear_center, front_center, ax1)
ax1.set_title('Initial Bicycle Frame Geometry')

ax2 = fig.add_subplot(122, projection='3d')
plot_bicycle_frame_3d(optimized_hardpoints, wheel_diameter, optimized_params[3], optimized_params[4], optimized_params[5], optimized_params[6], rear_center, front_center, ax2)
ax2.set_title('Optimized Bicycle Frame Geometry')

plt.tight_layout()

# Print initial and optimized values
print("Initial Parameters:")
for param_name, param_value in zip(["Seat Tube Length", "Top Tube Length", "Head Tube Length", "Seat Tube Angle", "Head Tube Angle", "Fork Angle", "Top Tube Angle", "Bottom Bracket Drop"], initial_params[:8]):
    print(f"{param_name}: {param_value}")

print("\nOptimized Parameters:")
for param_name, param_value in zip(["Seat Tube Length", "Top Tube Length", "Head Tube Length", "Seat Tube Angle", "Head Tube Angle", "Fork Angle", "Top Tube Angle", "Bottom Bracket Drop"], optimized_params[:8]):
    print(f"{param_name}: {param_value:.2f}")

print("\nInitial Tube Dimensions:")
for tube, dims in initial_tube_dimensions.items():
    print(f"{tube.capitalize()} dimensions: outer diameter {dims[0]:.2f} mm, thickness {dims[1]:.2f} mm")

print("\nOptimized Tube Dimensions:")
for tube, dims in optimized_tube_dimensions.items():
    print(f"{tube.capitalize()} dimensions: outer diameter {dims[0]:.2f} mm, thickness {dims[1]:.2f} mm")

print("\nInitial Torsional Rigidity and Fatigue Life:")
for tube, rigidity in initial_torsional_rigidity.items():
    print(f"{tube.capitalize()} torsional rigidity: {rigidity:.2f} Nmm/rad")
for tube, life in initial_fatigue_life.items():
    print(f"{tube.capitalize()} fatigue life: {life:.2e} cycles")
print(f"Total torsional rigidity of the bicycle: {initial_total_torsional_rigidity:.2f} Nmm/rad")
print(f"Overall fatigue life of the bicycle: {initial_overall_fatigue_life:.2e} cycles")

print("\nOptimized Torsional Rigidity and Fatigue Life:")
for tube, rigidity in optimized_torsional_rigidity.items():
    print(f"{tube.capitalize()} torsional rigidity: {rigidity:.2f} Nmm/rad")
for tube, life in optimized_fatigue_life.items():
    print(f"{tube.capitalize()} fatigue life: {life:.2e} cycles")
print(f"Total torsional rigidity of the bicycle: {optimized_total_torsional_rigidity:.2f} Nmm/rad")
print(f"Overall fatigue life of the bicycle: {optimized_overall_fatigue_life:.2e} cycles")

# Plot the initial and hardpoint 3D wireframes
plt.show()