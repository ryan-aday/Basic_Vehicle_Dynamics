print("Ryan Aday\nCar Vehicle Dynamics Optimizer\n")
print("Version 1.1: Added wheel graphics\n")

print("""Optimizes front and rear suspension geometry 
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
    front_wheel_center (assumes steeing link attaches to center)
    rear_wheel_center

Swaybar Characteristics: 
    front_sway_bar_stiffness
    rear_sway_bar_stiffness

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
    sys
    \n""")

try:
    import numpy as np
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import sys
except ImportError:
    import sys
    sys.exit("""
        You need the numpy, scipy, and matplotlib libraries.
        To install these libraries, please enter:
        pip install numpy scipy matplotlib
        """)

# Define hardpoints (example values)
initial_hardpoints = {
    'front_UCA_inboard': [0.4, 0.5, 0.6],
    'front_UCA_outboard': [0.8, 1.0, 0.5],
    'front_LCA_inboard': [0.2, 0.5, 0.3],
    'front_LCA_outboard': [0.8, 1.0, 0.2],
    'front_steering_rack': [0.2, 0.6, 0.4],
    'front_wheel_center': [0.8, 1.0, 0.4],
    'rear_UCA_inboard': [-0.4, 0.5, 0.6],
    'rear_UCA_outboard': [-0.8, 1.0, 0.5],
    'rear_LCA_inboard': [-0.6, 0.5, 0.3],
    'rear_LCA_outboard': [-0.8, 1.0, 0.2],
    'rear_wheel_center': [-0.8, 1.0, 0.4],
    'front_sway_bar': [0.2, 0.7, 0.2],
    'rear_sway_bar': [-0.2, 0.7, 0.2]
}

# Initial values for the sway bar stiffness (example values in N/m)
initial_sway_bar_stiffness = {
    'front_sway_bar': 30000,
    'rear_sway_bar': 30000
}

# Vehicle and suspension dimensions
wheelbase = 2.7  # meters
track_width = 1.725  # meters
steering_rack_length = 0.5  # meters (example value)
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
    (0.3, 0.5), (0.4, 0.6), (0.5, 0.7),   # front_UCA_inboard
    (0.7, 0.9), (0.9, 1.1), (0.4, 0.6),   # front_UCA_outboard
    (0.1, 0.3), (0.4, 0.6), (0.2, 0.4),   # front_LCA_inboard
    (0.7, 0.9), (0.9, 1.1), (0.1, 0.3),   # front_LCA_outboard
    (0.1, 0.3), (0.5, 0.7), (0.3, 0.5),   # front_steering_rack
    (0.7, 0.9), (0.9, 1.1), (0.3, 0.5),   # front_wheel_center
    (-0.5, -0.3), (0.4, 0.6), (0.5, 0.7), # rear_UCA_inboard
    (-0.9, -0.7), (0.9, 1.1), (0.4, 0.6), # rear_UCA_outboard
    (-0.7, -0.5), (0.4, 0.6), (0.2, 0.4), # rear_LCA_inboard
    (-0.9, -0.7), (0.9, 1.1), (0.1, 0.3), # rear_LCA_outboard
    (-0.9, -0.7), (0.9, 1.1), (0.3, 0.5), # rear_wheel_center
    (0.1, 0.3), (0.6, 0.8), (0.1, 0.3),   # front_sway_bar
    (-0.3, -0.1), (0.6, 0.8), (0.1, 0.3), # rear_sway_bar
    (10000, 50000),  # front_sway_bar_stiffness
    (10000, 50000)   # rear_sway_bar_stiffness
]

# Flatten the initial hardpoints and sway bar stiffness for the optimizer
initial_guess = np.array([value for points in initial_hardpoints.values() for value in points] +
                         list(initial_sway_bar_stiffness.values()))

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
def calculate_roll_angle(sprung_mass, unsprung_mass, spring_constant, track_width, lateral_acceleration, front_sway_bar_stiffness, rear_sway_bar_stiffness):
    roll_stiffness = spring_constant * (track_width / 2)**2 + front_sway_bar_stiffness + rear_sway_bar_stiffness
    roll_angle = (lateral_acceleration * 9.81 * (sprung_mass + unsprung_mass) * track_width) / (2 * roll_stiffness)
    return np.degrees(roll_angle)

# Objective function to minimize
def objective(params):
    # Extract hardpoints and sway bar stiffness from the parameter list
    hardpoints_flat = params[:len(initial_guess)-2]
    sway_bar_stiffness_flat = params[len(initial_guess)-2:]
    
    # Reshape the flat lists back into dictionaries
    hardpoints = {
        'front_UCA_inboard': hardpoints_flat[0:3],
        'front_UCA_outboard': hardpoints_flat[3:6],
        'front_LCA_inboard': hardpoints_flat[6:9],
        'front_LCA_outboard': hardpoints_flat[9:12],
        'front_steering_rack': hardpoints_flat[12:15],
        'front_wheel_center': hardpoints_flat[15:18],
        'rear_UCA_inboard': hardpoints_flat[18:21],
        'rear_UCA_outboard': hardpoints_flat[21:24],
        'rear_LCA_inboard': hardpoints_flat[24:27],
        'rear_LCA_outboard': hardpoints_flat[27:30],
        'rear_wheel_center': hardpoints_flat[30:33],
        'front_sway_bar': hardpoints_flat[33:36],
        'rear_sway_bar': hardpoints_flat[36:39]
    }
    sway_bar_stiffness = {
        'front_sway_bar': sway_bar_stiffness_flat[0],
        'rear_sway_bar': sway_bar_stiffness_flat[1]
    }
    
    # Generate data for demonstration
    displacement = np.linspace(-0.1, 0.1, 100)  # Displacement in meters
    velocity = np.gradient(displacement, edge_order=2)  # Velocity in meters/second
    lateral_acceleration = np.linspace(-1, 1, 100)  # Lateral acceleration in g

    # Calculate wheel travel
    wheel_travel = calculate_wheel_travel(spring_constant, damping_coefficient, displacement, velocity)
    
    front_camber_angle = calculate_camber_angle(hardpoints['front_LCA_outboard'], hardpoints['front_UCA_outboard'], wheel_travel)
    front_toe_angle = calculate_toe_angle(hardpoints['front_steering_rack'], wheel_travel)
    front_caster_angle = calculate_caster_angle(hardpoints['front_UCA_inboard'], hardpoints['front_LCA_inboard'], wheel_travel)
    front_ackermann_angle = calculate_ackermann_angle(wheelbase, track_width, wheel_travel)
    front_steer_angle = calculate_steer_angle(steering_rack_length, wheel_travel)
    front_roll_angle = calculate_roll_angle(sprung_mass, unsprung_mass, spring_constant, track_width, lateral_acceleration, sway_bar_stiffness['front_sway_bar'], sway_bar_stiffness['rear_sway_bar'])

    rear_camber_angle = calculate_camber_angle(hardpoints['rear_LCA_outboard'], hardpoints['rear_UCA_outboard'], wheel_travel)
    rear_toe_angle = calculate_toe_angle(hardpoints['rear_wheel_center'], wheel_travel)
    rear_caster_angle = calculate_caster_angle(hardpoints['rear_UCA_inboard'], hardpoints['rear_LCA_inboard'], wheel_travel)
    rear_ackermann_angle = calculate_ackermann_angle(wheelbase, track_width, wheel_travel)
    rear_steer_angle = calculate_steer_angle(steering_rack_length, wheel_travel)
    rear_roll_angle = calculate_roll_angle(sprung_mass, unsprung_mass, spring_constant, track_width, lateral_acceleration, sway_bar_stiffness['front_sway_bar'], sway_bar_stiffness['rear_sway_bar'])
    
    # Calculate the sum of squared errors for the desired ranges
    error = 0
    error += np.sum((np.clip(front_camber_angle, *desired_ranges['camber_angle']) - front_camber_angle) ** 2)
    error += np.sum((np.clip(front_toe_angle, *desired_ranges['toe_angle']) - front_toe_angle) ** 2)
    error += np.sum((np.clip(front_caster_angle, *desired_ranges['caster_angle']) - front_caster_angle) ** 2)
    error += np.sum((np.clip(front_ackermann_angle, *desired_ranges['ackermann_angle']) - front_ackermann_angle) ** 2)
    error += np.sum((np.clip(front_roll_angle, *desired_ranges['roll_angle']) - front_roll_angle) ** 2)
    error += np.sum((np.clip(front_steer_angle, *desired_ranges['steer_angle']) - front_steer_angle) ** 2)
    error += np.sum((np.clip(rear_camber_angle, *desired_ranges['camber_angle']) - rear_camber_angle) ** 2)
    error += np.sum((np.clip(rear_toe_angle, *desired_ranges['toe_angle']) - rear_toe_angle) ** 2)
    error += np.sum((np.clip(rear_caster_angle, *desired_ranges['caster_angle']) - rear_caster_angle) ** 2)
    error += np.sum((np.clip(rear_ackermann_angle, *desired_ranges['ackermann_angle']) - rear_ackermann_angle) ** 2)
    error += np.sum((np.clip(rear_roll_angle, *desired_ranges['roll_angle']) - rear_roll_angle) ** 2)
    error += np.sum((np.clip(rear_steer_angle, *desired_ranges['steer_angle']) - rear_steer_angle) ** 2)
    
    return error

# Perform the optimization
result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

# Reshape the optimized hardpoints and sway bar stiffness back into dictionaries
optimized_hardpoints = {
    'front_UCA_inboard': result.x[0:3],
    'front_UCA_outboard': result.x[3:6],
    'front_LCA_inboard': result.x[6:9],
    'front_LCA_outboard': result.x[9:12],
    'front_steering_rack': result.x[12:15],
    'front_wheel_center': result.x[15:18],
    'rear_UCA_inboard': result.x[18:21],
    'rear_UCA_outboard': result.x[21:24],
    'rear_LCA_inboard': result.x[24:27],
    'rear_LCA_outboard': result.x[27:30],
    'rear_wheel_center': result.x[30:33],
    'front_sway_bar': result.x[33:36],
    'rear_sway_bar': result.x[36:39]
}
optimized_sway_bar_stiffness = {
    'front_sway_bar': result.x[39],
    'rear_sway_bar': result.x[40]
}

# Function to plot suspension geometry
def plot_suspension(ax, hardpoints, title, wheel_radius, wheel_width):
    # Front suspension
    ax.plot([hardpoints['front_UCA_inboard'][1], hardpoints['front_UCA_outboard'][1]],
            [hardpoints['front_UCA_inboard'][0], hardpoints['front_UCA_outboard'][0]],
            [hardpoints['front_UCA_inboard'][2], hardpoints['front_UCA_outboard'][2]], 'b-o', label='Front Upper Control Arm')
    
    ax.plot([hardpoints['front_LCA_inboard'][1], hardpoints['front_LCA_outboard'][1]],
            [hardpoints['front_LCA_inboard'][0], hardpoints['front_LCA_outboard'][0]],
            [hardpoints['front_LCA_inboard'][2], hardpoints['front_LCA_outboard'][2]], 'g-o', label='Front Lower Control Arm')
    
    ax.plot([hardpoints['front_steering_rack'][1], hardpoints['front_wheel_center'][1]],
            [hardpoints['front_steering_rack'][0], hardpoints['front_wheel_center'][0]],
            [hardpoints['front_steering_rack'][2], hardpoints['front_wheel_center'][2]], 'r-o', label='Front Steering Link')
    
    ax.plot([hardpoints['front_sway_bar'][1], hardpoints['front_UCA_outboard'][1]],
            [hardpoints['front_sway_bar'][0], hardpoints['front_UCA_outboard'][0]],
            [hardpoints['front_sway_bar'][2], hardpoints['front_UCA_outboard'][2]], 'purple', label='Front Sway Bar')
    
    # Rear suspension
    ax.plot([hardpoints['rear_UCA_inboard'][1], hardpoints['rear_UCA_outboard'][1]],
            [hardpoints['rear_UCA_inboard'][0], hardpoints['rear_UCA_outboard'][0]],
            [hardpoints['rear_UCA_inboard'][2], hardpoints['rear_UCA_outboard'][2]], 'b-o', label='Rear Upper Control Arm')
    
    ax.plot([hardpoints['rear_LCA_inboard'][1], hardpoints['rear_LCA_outboard'][1]],
            [hardpoints['rear_LCA_inboard'][0], hardpoints['rear_LCA_outboard'][0]],
            [hardpoints['rear_LCA_inboard'][2], hardpoints['rear_LCA_outboard'][2]], 'g-o', label='Rear Lower Control Arm')
    
    ax.plot([hardpoints['rear_wheel_center'][1]], [hardpoints['rear_wheel_center'][0]], [hardpoints['rear_wheel_center'][2]], 'ko', label='Rear Wheel Center')
    
    ax.plot([hardpoints['rear_sway_bar'][1], hardpoints['rear_UCA_outboard'][1]],
            [hardpoints['rear_sway_bar'][0], hardpoints['rear_UCA_outboard'][0]],
            [hardpoints['rear_sway_bar'][2], hardpoints['rear_UCA_outboard'][2]], 'purple', label='Rear Sway Bar')
    
    # Draw wheels as wireframe cylinders in the xz plane
    for wheel_center in ['front_wheel_center', 'rear_wheel_center']:
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(-wheel_width / 2, wheel_width / 2, 2)
        theta_grid, x_grid = np.meshgrid(theta, z)
        y_grid = wheel_radius * np.cos(theta_grid)
        z_grid = wheel_radius * np.sin(theta_grid)
        x_grid += hardpoints[wheel_center][1]
        y_grid += hardpoints[wheel_center][0]
        z_grid += hardpoints[wheel_center][2]

        ax.plot_wireframe(x_grid, y_grid, z_grid, color='grey')

    ax.set_title(title)
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)

print("\nInitial Hardpoints:")
for key, value in initial_hardpoints.items():
    print(f"{key}: {[f'{v:.8f}' for v in value]}")

print("\nInitial Sway Bar Stiffness:")
for key, value in initial_sway_bar_stiffness.items():
    print(f"{key}: {value:.8f}")

print("\nOptimized Hardpoints:")
for key, value in optimized_hardpoints.items():
    print(f"{key}: {[f'{v:.8f}' for v in value]}")

print("\nOptimized Sway Bar Stiffness:")
for key, value in optimized_sway_bar_stiffness.items():
    print(f"{key}: {value:.8f}")

# Plot initial and optimized suspension geometries
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
plot_suspension(ax1, initial_hardpoints, 'Initial Suspension Geometry', wheel_radius, wheel_width)

ax2 = fig.add_subplot(122, projection='3d')
plot_suspension(ax2, optimized_hardpoints, 'Optimized Suspension Geometry', wheel_radius, wheel_width)

plt.tight_layout()
plt.show()
