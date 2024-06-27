print("Ryan Aday\nCar Vehicle Dynamics Optimizer\n")
print("Version 1.0\n")

print("""Plots front and rear suspension dynamics with
provided hardpoints.

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

Vehicle Dynamics Properties Graphed:
    camber_angle
    toe_angle
    caster_angle
    ackermann_angle
    steer_angle
    roll_angle

Requires libraries:
    numpy
    matplotlib   
    sys
    \n""")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
except ImportError:
    sys.exit("""
        You need the numpy and matplotlib libraries.
        To install these libraries, please enter:
        pip install numpy scipy matplotlib
        """)

# Define hardpoints (example values)
hardpoints = {
    'front_UCA_inboard': np.array([0.4, 0.5, 0.6]),  # Front Upper Control Arm inboard point
    'front_UCA_outboard': np.array([0.8, 1.0, 0.5]), # Front Upper Control Arm outboard point
    'front_LCA_inboard': np.array([0.2, 0.5, 0.3]),  # Front Lower Control Arm inboard point
    'front_LCA_outboard': np.array([0.8, 1.0, 0.2]), # Front Lower Control Arm outboard point
    'front_steering_rack': np.array([0.2, 0.6, 0.4]), # Front Steering rack point
    'front_wheel_center': np.array([0.8, 1.0, 0.4]),  # Front Wheel center point
    'rear_UCA_inboard': np.array([-0.4, 0.5, 0.6]),   # Rear Upper Control Arm inboard point
    'rear_UCA_outboard': np.array([-0.8, 1.0, 0.5]),  # Rear Upper Control Arm outboard point
    'rear_LCA_inboard': np.array([-0.6, 0.5, 0.3]),   # Rear Lower Control Arm inboard point
    'rear_LCA_outboard': np.array([-0.8, 1.0, 0.2]),  # Rear Lower Control Arm outboard point
    'rear_wheel_center': np.array([-0.8, 1.0, 0.4]),  # Rear Wheel center point
    'front_sway_bar': np.array([0.2, 0.7, 0.2]),      # Front Sway bar point
    'rear_sway_bar': np.array([-0.2, 0.7, 0.2])       # Rear Sway bar point
}

# Vehicle and suspension dimensions
wheelbase = 2.7  # meters
track_width = 1.725  # meters
steering_rack_length = 0.5  # meters (example value)
wheel_radius = 0.3  # meters
wheel_width = 0.2  # meters

# Spring and damper properties
spring_constant = 30000  # N/m
damping_coefficient = 1500  # Ns/m

# Mass properties
sprung_mass = 1200  # kg
unsprung_mass = 200  # kg

# Initial values for the sway bar stiffness (example values in N/m)
initial_sway_bar_stiffness = {
    'front_sway_bar': 30000,
    'rear_sway_bar': 30000
}

# Function to calculate wheel travel based on spring and damper properties
def calculate_wheel_travel(spring_constant, damping_coefficient, displacement, velocity):
    spring_force = spring_constant * displacement
    damping_force = damping_coefficient * velocity
    total_force = spring_force + damping_force
    wheel_travel = total_force / spring_constant
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
    roll_angle = (lateral_acceleration * (sprung_mass + unsprung_mass) * track_width) / (2 * roll_stiffness)
    return np.degrees(roll_angle)

# Generate data for demonstration
displacement = np.linspace(-0.1, 0.1, 100)  # Displacement in meters
velocity = np.gradient(displacement, edge_order=2)  # Velocity in meters/second
lateral_acceleration = np.linspace(-1, 1, 100)  # Lateral acceleration in g

# Calculate wheel travel
wheel_travel = calculate_wheel_travel(spring_constant, damping_coefficient, displacement, velocity)

# Calculate angles for both front and rear suspensions
front_camber_angle = calculate_camber_angle(hardpoints['front_LCA_outboard'], hardpoints['front_UCA_outboard'], wheel_travel)
front_toe_angle = calculate_toe_angle(hardpoints['front_steering_rack'], wheel_travel)
front_caster_angle = calculate_caster_angle(hardpoints['front_UCA_inboard'], hardpoints['front_LCA_inboard'], wheel_travel)
front_ackermann_angle = calculate_ackermann_angle(wheelbase, track_width, wheel_travel)
front_steer_angle = calculate_steer_angle(steering_rack_length, wheel_travel)
front_roll_angle = calculate_roll_angle(sprung_mass, unsprung_mass, spring_constant, track_width, lateral_acceleration, initial_sway_bar_stiffness['front_sway_bar'], initial_sway_bar_stiffness['rear_sway_bar'])

rear_camber_angle = calculate_camber_angle(hardpoints['rear_LCA_outboard'], hardpoints['rear_UCA_outboard'], wheel_travel)
rear_toe_angle = calculate_toe_angle(hardpoints['rear_sway_bar'], wheel_travel)
rear_caster_angle = calculate_caster_angle(hardpoints['rear_UCA_inboard'], hardpoints['rear_LCA_inboard'], wheel_travel)
rear_ackermann_angle = calculate_ackermann_angle(wheelbase, track_width, wheel_travel)
rear_steer_angle = calculate_steer_angle(steering_rack_length, wheel_travel)
rear_roll_angle = calculate_roll_angle(sprung_mass, unsprung_mass, spring_constant, track_width, lateral_acceleration, initial_sway_bar_stiffness['front_sway_bar'], initial_sway_bar_stiffness['rear_sway_bar'])

# Create a figure with subplots for front and rear suspension dynamics
fig, axs = plt.subplots(1, 2, figsize=(12, 16))

# Plot front suspension angles
axs[0].plot(wheel_travel, front_camber_angle, label='Front Camber Angle', color='blue', marker='o')
axs[0].plot(wheel_travel, front_toe_angle, label='Front Toe Angle', color='green', marker='x')
axs[0].plot(wheel_travel, front_caster_angle, label='Front Caster Angle', color='purple', marker='s')
axs[0].plot(wheel_travel, front_ackermann_angle, label='Front Ackermann Angle', color='orange', marker='d')
axs[0].plot(wheel_travel, front_steer_angle, label='Front Steer Angle', color='brown', marker='^')
axs[0].plot(wheel_travel, front_roll_angle, label='Front Roll Angle', color='red', marker='v')

axs[0].set_xlabel('Wheel Travel (m)')
axs[0].set_ylabel('Angle (degrees)')
axs[0].set_title('Front Suspension Angles vs Wheel Travel')
axs[0].legend()
axs[0].grid(True)

# Plot rear suspension angles
axs[1].plot(wheel_travel, rear_camber_angle, label='Rear Camber Angle', color='blue', linestyle='--', marker='o')
axs[1].plot(wheel_travel, rear_toe_angle, label='Rear Toe Angle', color='green', linestyle='--', marker='x')
axs[1].plot(wheel_travel, rear_caster_angle, label='Rear Caster Angle', color='purple', linestyle='--', marker='s')
axs[1].plot(wheel_travel, rear_ackermann_angle, label='Rear Ackermann Angle', color='orange', linestyle='--', marker='d')
axs[1].plot(wheel_travel, rear_steer_angle, label='Rear Steer Angle', color='brown', linestyle='--', marker='^')
axs[1].plot(wheel_travel, rear_roll_angle, label='Rear Roll Angle', color='red', linestyle='--', marker='v')

axs[1].set_xlabel('Wheel Travel (m)')
axs[1].set_ylabel('Angle (degrees)')
axs[1].set_title('Rear Suspension Angles vs Wheel Travel')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
