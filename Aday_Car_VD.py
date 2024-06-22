print("Ryan Aday\nCar Vehicle Dynamics Optimizer\n")
print("Version 1.0\n")

print("""Plots front suspension dynamics with
provided hardpoints.

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
    \n""")

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("""
        You need the numpy and matplotlib libraries.
        To install these libraries, please enter:
        pip install numpy scipy matplotlib
        """)

# Define hardpoints (example values)
hardpoints = {
    'UCA_inboard': np.array([0.5, 0.4, 0.6]),  # Upper Control Arm inboard point
    'UCA_outboard': np.array([1.0, 0.8, 0.5]), # Upper Control Arm outboard point
    'LCA_inboard': np.array([0.5, 0.2, 0.3]),  # Lower Control Arm inboard point
    'LCA_outboard': np.array([1.0, 0.8, 0.2]), # Lower Control Arm outboard point
    'steering_rack': np.array([0.6, 0.2, 0.4]), # Steering rack point
    'wheel_center': np.array([1.0, 0.8, 0.4])  # Wheel center point
}

# Vehicle and suspension dimensions
wheelbase = 2.5  # meters
track_width = 1.6  # meters
steering_rack_length = 0.5  # meters (example value)
wheel_radius = 0.3  # meters
wheel_width = 0.2  # meters

# Spring and damper properties
spring_constant = 30000  # N/m
damping_coefficient = 1500  # Ns/m

# Mass properties
sprung_mass = 1200  # kg
unsprung_mass = 200  # kg

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
def calculate_roll_angle(sprung_mass, unsprung_mass, spring_constant, track_width, lateral_acceleration):
    roll_stiffness = spring_constant * (track_width / 2)**2
    roll_angle = (lateral_acceleration * (sprung_mass + unsprung_mass) * track_width) / (2 * roll_stiffness)
    return np.degrees(roll_angle)

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

# Create a single plot with all curves
plt.figure(figsize=(12, 8))

# Plot all angles on the same plot
plt.plot(wheel_travel, camber_angle, label='Camber Angle', color='blue', marker='o')
plt.plot(wheel_travel, toe_angle, label='Toe Angle', color='green', marker='x')
plt.plot(wheel_travel, caster_angle, label='Caster Angle', color='purple', marker='s')
plt.plot(wheel_travel, ackermann_angle, label='Ackermann Angle', color='orange', marker='d')
plt.plot(wheel_travel, steer_angle, label='Steer Angle', color='brown', marker='^')
plt.plot(wheel_travel, roll_angle, label='Roll Angle', color='red', marker='v')

# Add labels, title, legend, and grid
plt.xlabel('Wheel Travel (m)')
plt.ylabel('Angle (degrees)')
plt.title('Suspension Angles vs Wheel Travel')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
