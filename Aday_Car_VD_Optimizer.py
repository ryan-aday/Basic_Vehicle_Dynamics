print("Ryan Aday\nCar Vehicle Dynamics Optimizer\n")
print("Version 1.0\n")

print("""Optimizes basic front suspension geometry 
with the L-BFGS-B minimization algorithm
and provides a plot to demonstrate the 
differences between the initial and optimized
hardpoints.

Initial Hardpoints:
    UCA_inboard (average location of mounting points)
    UCA_outboard (average location of mounting points)
    LCA_inboard  (average location of mounting points)
    LCA_outboard (average location of mounting points)
    steering_rack (center location of mounting)
    wheel_center

Provided Constants:
# Vehicle and suspension dimensions
    wheelbase
    track_width
    steering_rack_length
    wheel_radius
    wheel_width

# Spring and damper properties
    spring_constant
    damping_coefficient

# Mass properties
    sprung_mass
    unsprung_mass

Vehicle Dynamics Properties to Optimize With:
    camber_angle
    toe_angle
    caster_angle
    ackermann_angle
    steer_angle
    roll_angle

Requires libraries:
    numpy
    scipy
    matplotlib   
    \n""")

try:
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    sys.exit("""
        You need the numpy, scipy, and matplotlib libraries.
        To install these libraries, please enter:
        pip install numpy scipy matplotlib
        """)

# Define hardpoints (example values)
initial_hardpoints = {
    'UCA_inboard': [0.5, 0.4, 0.6],
    'UCA_outboard': [1.0, 0.8, 0.5],
    'LCA_inboard': [0.5, 0.2, 0.3],
    'LCA_outboard': [1.0, 0.8, 0.2],
    'steering_rack': [0.6, 0.2, 0.4],
    'wheel_center': [1.0, 0.8, 0.4]
}

# Vehicle and suspension dimensions
wheelbase = 2.7  # meters
track_width = 1.725  # meters
steering_rack_length = 0.15  # meters (example value)
wheel_radius = 0.381  # meters
wheel_width = 0.185  # meters

# Spring and damper properties
spring_constant = 30000  # N/m
damping_coefficient = 1500  # Ns/m

# Mass properties
sprung_mass = 1250  # kg
unsprung_mass = 11  # kg

# Desired ranges for suspension parameters (example values)
desired_ranges = {
    'camber_angle': (-1, 1),    # degrees
    'toe_angle': (-0.5, 0.5),   # degrees
    'caster_angle': (3, 5),     # degrees
    'ackermann_angle': (10, 20),# degrees
    'steer_angle': (-30, 30),   # degrees
    'roll_angle': (-2, 2)       # degrees
}

# Define bounds for the optimization
bounds = [
    (0.4, 0.6), (0.3, 0.5), (0.5, 0.7),   # UCA_inboard
    (0.9, 1.1), (0.7, 0.9), (0.4, 0.6),   # UCA_outboard
    (0.4, 0.6), (0.1, 0.3), (0.2, 0.4),   # LCA_inboard
    (0.9, 1.1), (0.7, 0.9), (0.1, 0.3),   # LCA_outboard
    (0.5, 0.7), (0.1, 0.3), (0.3, 0.5),   # steering_rack
    (0.9, 1.1), (0.7, 0.9), (0.3, 0.5)    # wheel_center
]

# Flatten the initial hardpoints for the optimizer
initial_guess = np.array([value for points in initial_hardpoints.values() for value in points])

# Function to calculate wheel travel based on spring and damper properties
def calculate_wheel_travel(spring_constant, damping_coefficient, displacement, velocity):
    spring_force = spring_constant * displacement
    damping_force = damping_coefficient * velocity
    total_force = spring_force + damping_force
    wheel_travel = displacement - total_force / spring_constant
    return wheel_travel

# Function to calculate camber angle
def calculate_camber_angle(LCA_outboard, UCA_outboard, wheel_travel):
    camber_angle = np.arctan2(UCA_outboard[2] - LCA_outboard[2], UCA_outboard[1] - LCA_outboard[1])
    return np.degrees(camber_angle + wheel_travel)

# Function to calculate toe angle
def calculate_toe_angle(steering_rack, wheel_travel):
    toe_angle = np.arctan2(steering_rack[1], steering_rack[0])
    return np.degrees(toe_angle + wheel_travel)

# Function to calculate caster angle
def calculate_caster_angle(UCA_inboard, LCA_inboard, wheel_travel):
    caster_angle = np.arctan2(UCA_inboard[2] - LCA_inboard[2], UCA_inboard[0] - LCA_inboard[0])
    return np.degrees(caster_angle + wheel_travel)

# Function to calculate Ackermann angle
def calculate_ackermann_angle(wheelbase, track_width, steer_angle):
    ackermann_angle = np.arctan(wheelbase / (track_width + steer_angle))
    return np.degrees(ackermann_angle)

# Function to calculate steering angle based on steering rack length
def calculate_steer_angle(steering_rack_length, wheel_travel):
    steer_angle = np.arctan(steering_rack_length / (track_width / 2))
    return np.degrees(steer_angle + wheel_travel)

# Function to calculate roll angle based on sprung mass, unsprung mass, spring properties, and lateral acceleration
def calculate_roll_angle(sprung_mass, unsprung_mass, spring_constant, track_width, lateral_acceleration):
    roll_stiffness = spring_constant * (track_width / 2)**2
    roll_angle = (lateral_acceleration * 9.81 * (sprung_mass + unsprung_mass) * track_width) / (2 * roll_stiffness)
    return np.degrees(roll_angle)

# Objective function to minimize
def objective(hardpoints_flat):
    # Reshape the flat list back into a dictionary of hardpoints
    hardpoints = {
        'UCA_inboard': hardpoints_flat[0:3],
        'UCA_outboard': hardpoints_flat[3:6],
        'LCA_inboard': hardpoints_flat[6:9],
        'LCA_outboard': hardpoints_flat[9:12],
        'steering_rack': hardpoints_flat[12:15],
        'wheel_center': hardpoints_flat[15:18]
    }
    
    # Generate data for demonstration
    displacement = np.linspace(-0.1, 0.1, 100)  # Displacement in meters
    velocity = np.gradient(displacement, edge_order=2)  # Velocity in meters/second
    lateral_acceleration = np.linspace(-1, 1, 100)  # Lateral acceleration in g

    # Calculate wheel travel
    wheel_travel = calculate_wheel_travel(spring_constant, damping_coefficient, displacement, velocity)
    camber_angle = calculate_camber_angle(hardpoints['LCA_outboard'], hardpoints['UCA_outboard'], wheel_travel)
    toe_angle = calculate_toe_angle(hardpoints['steering_rack'], wheel_travel)
    caster_angle = calculate_caster_angle(hardpoints['UCA_inboard'], hardpoints['LCA_inboard'], wheel_travel)
    ackermann_angle = calculate_ackermann_angle(wheelbase, track_width, wheel_travel)
    roll_angle = calculate_roll_angle(sprung_mass, unsprung_mass, spring_constant, track_width, lateral_acceleration)
    steer_angle = calculate_steer_angle(steering_rack_length, wheel_travel)
    
    # Calculate the sum of squared errors for the desired ranges
    error = 0
    error += np.sum((np.clip(camber_angle, *desired_ranges['camber_angle']) - camber_angle) ** 2)
    error += np.sum((np.clip(toe_angle, *desired_ranges['toe_angle']) - toe_angle) ** 2)
    error += np.sum((np.clip(caster_angle, *desired_ranges['caster_angle']) - caster_angle) ** 2)
    error += np.sum((np.clip(ackermann_angle, *desired_ranges['ackermann_angle']) - ackermann_angle) ** 2)
    error += np.sum((np.clip(roll_angle, *desired_ranges['roll_angle']) - roll_angle) ** 2)
    error += np.sum((np.clip(steer_angle, *desired_ranges['steer_angle']) - steer_angle) ** 2)
    
    return error

# Perform the optimization
result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

# Reshape the optimized hardpoints back into a dictionary
optimized_hardpoints = {
    'UCA_inboard': result.x[0:3],
    'UCA_outboard': result.x[3:6],
    'LCA_inboard': result.x[6:9],
    'LCA_outboard': result.x[9:12],
    'steering_rack': result.x[12:15],
    'wheel_center': result.x[15:18]
}

# Function to plot suspension geometry
def plot_suspension(ax, hardpoints, title):
    ax.plot([hardpoints['UCA_inboard'][0], hardpoints['UCA_outboard'][0]],
            [hardpoints['UCA_inboard'][1], hardpoints['UCA_outboard'][1]],
            [hardpoints['UCA_inboard'][2], hardpoints['UCA_outboard'][2]], 'b-o', label='Upper Control Arm')
    
    ax.plot([hardpoints['LCA_inboard'][0], hardpoints['LCA_outboard'][0]],
            [hardpoints['LCA_inboard'][1], hardpoints['LCA_outboard'][1]],
            [hardpoints['LCA_inboard'][2], hardpoints['LCA_outboard'][2]], 'g-o', label='Lower Control Arm')
    
    ax.plot([hardpoints['steering_rack'][0], hardpoints['wheel_center'][0]],
            [hardpoints['steering_rack'][1], hardpoints['wheel_center'][1]],
            [hardpoints['steering_rack'][2], hardpoints['wheel_center'][2]], 'r-o', label='Steering Link')
    
    ax.plot([hardpoints['wheel_center'][0]], [hardpoints['wheel_center'][1]], [hardpoints['wheel_center'][2]], 'ko', label='Wheel Center')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)

print("\nInitial Hardpoints:")
for key, value in initial_hardpoints.items():
    print(f"{key}: {[f'{v:.8f}' for v in value]}")

print("\nOptimized Hardpoints:")
for key, value in optimized_hardpoints.items():
    print(f"{key}: {[f'{v:.8f}' for v in value]}")

# Plot initial and optimized suspension geometries
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
plot_suspension(ax1, initial_hardpoints, 'Initial Suspension Geometry')

ax2 = fig.add_subplot(122, projection='3d')
plot_suspension(ax2, optimized_hardpoints, 'Optimized Suspension Geometry')

plt.tight_layout()
plt.show()
