print("Ryan Aday\nBike Properties\n")
print("Version 1.0\n")

print("""Provides tube dimensions and angles for a standard bike frame
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

Tube dimensions (outer diameter, thickness) in mm
tube_dimensions
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
    \n""")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D, art3d
except ImportError:
    sys.exit("""
        You need the numpy, mpl_toolkits, scipy, and matplotlib libraries.
        To install these libraries, please enter:
        pip install numpy matplotlib mpl_toolkits
        """)

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

def plot_bicycle_frame_3d(hardpoints, wheel_diameter, seat_tube_angle, head_tube_angle, fork_angle, top_tube_angle, rear_center, front_center):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

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
    plt.show()

def calculate_torsional_rigidity_and_fatigue_life(tube_dimensions, material_properties, tube_lengths):
    torsional_rigidity = {}
    fatigue_life = {}

    total_torsional_rigidity = 0  # Sum of torsional rigidities
    total_fatigue_life = 0  # Sum of fatigue lives
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

        # Assume stress amplitude for fatigue life calculation
        sigma_a = 100  # MPa
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

# Parameters (example values based on the provided image)
wheel_diameter = 700  # in mm
wheel_thickness = 25  # in mm
seat_tube_length = 500  # in mm
top_tube_length = 600  # in mm
head_tube_length = 75  # in mm
seat_tube_angle = -10  # in degrees (referenced from vertical)
head_tube_angle = -10  # in degrees (referenced from vertical)
fork_angle = 30  # in degrees (referenced from vertical)
top_tube_angle = 45  # in degrees (referenced from horizontal)
bottom_bracket_drop = 70  # in mm
rear_center = 405  # in mm (horizontal distance)
front_center = 650  # in mm (horizontal distance)

# Tube dimensions (outer diameter, thickness) in mm
tube_dimensions = {
    'seat_tube': (28.6, 1.2),
    'top_tube': (25.4, 1.2),
    'head_tube': (31.8, 1.5),
    'down_tube': (34.9, 1.2),
    'seatstay': (19.0, 1.0),
    'fork': (28.6, 1.2)
}

# Material properties for 4130 steel (example values)
material_properties = {
    'shear_modulus': 80e3,  # MPa for G
    'fatigue_strength_coefficient': 700,  # MPa for sigma_f_prime
    'fatigue_strength_exponent': -0.08  # dimensionless for b
}

# Calculate moments of inertia for each tube
moments_of_inertia = {tube: calculate_moment_of_inertia(dims[0], dims[1])[0] for tube, dims in tube_dimensions.items()}

# Print moments of inertia
for tube, moi in moments_of_inertia.items():
    print(f"{tube.capitalize()} moment of inertia: {moi:.2f} mm^4")

hardpoints = calculate_hardpoints(wheel_diameter, wheel_thickness, seat_tube_length, top_tube_length, head_tube_length, seat_tube_angle, head_tube_angle, fork_angle, top_tube_angle, bottom_bracket_drop, rear_center, front_center)

# Calculate and print lengths of each tube
tube_lengths = {
    'seat_tube': calculate_tube_length(hardpoints['bottom_bracket'], hardpoints['seat_tube_top']),
    'top_tube': calculate_tube_length(hardpoints['seat_tube_top'], hardpoints['head_tube_top']),
    'head_tube': calculate_tube_length(hardpoints['head_tube_top'], hardpoints['head_tube_bottom']),
    'down_tube': calculate_tube_length(hardpoints['head_tube_bottom'], hardpoints['bottom_bracket']),
    'seatstay': calculate_tube_length(hardpoints['rear_wheel_left'], hardpoints['seat_tube_top']),
    'fork': calculate_tube_length(hardpoints['fork_bottom_left'], hardpoints['head_tube_bottom']),
}

for tube, length in tube_lengths.items():
    print(f"{tube.capitalize()} length: {length:.2f} mm")

# Calculate torsional rigidity and fatigue life
torsional_rigidity, fatigue_life, total_torsional_rigidity, overall_fatigue_life = calculate_torsional_rigidity_and_fatigue_life(tube_dimensions, material_properties, tube_lengths)

# Print torsional rigidity and fatigue life
for tube, rigidity in torsional_rigidity.items():
    print(f"{tube.capitalize()} torsional rigidity: {rigidity:.2f} Nmm/rad")

for tube, life in fatigue_life.items():
    print(f"{tube.capitalize()} fatigue life: {life:.2e} cycles")

print(f"Total torsional rigidity of the bicycle: {total_torsional_rigidity:.2f} Nmm/rad")
print(f"Overall fatigue life of the bicycle: {overall_fatigue_life:.2e} cycles")

plot_bicycle_frame_3d(hardpoints, wheel_diameter, seat_tube_angle, head_tube_angle, fork_angle, top_tube_angle, rear_center, front_center)
