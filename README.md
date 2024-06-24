# Basic_Vehicle_Dynamics
Plotters and optimizers for vehicle dynamics.

# Car Vehicle Dynamics and Optimizer
Optimizes front and rear suspension geometry with the L-BFGS-B minimization algorithm and provides a plot to demonstrate the differences between the initial and optimized hardpoints.

![Car Suspension Optimizer](https://github.com/ryan-aday/Basic_Vehicle_Dynamics/assets/33794562/42594bb2-d264-48d5-bf0c-4a2228561598)
![Car Suspension Dynamics Plots](https://github.com/ryan-aday/Basic_Vehicle_Dynamics/assets/33794562/69d65ea6-3675-4de4-8a9c-9b440401fcb8)

Initial Hardpoints:
 - UCA_inboard (average location of mounting points)
 - UCA_outboard (average location of mounting points)
 - LCA_inboard  (average location of mounting points)
 - LCA_outboard (average location of mounting points)
 - steering_rack (center location of mounting)
 - front_wheel_center (assumes steeing link attaches to center)
 - rear_wheel_center

Swaybar Characteristics: 
 - front_sway_bar_stiffness
 - rear_sway_bar_stiffness

Provided Constants:
 - Vehicle and suspension dimensions
   - wheelbase
   - track_width
   - steering_rack_length
   - wheel_radius
   - wheel_width

 - Spring and damper properties
   - spring_constant
   - damping_coefficient

 - Mass properties
   - sprung_mass
   - unsprung_mass

Vehicle Dynamics Properties to Optimize With:
 - camber_angle
 - toe_angle
 - caster_angle
 - ackermann_angle
 - steer_angle
 - roll_angle

Requires libraries:
 - numpy
 - scipy
 - matplotlib

# Bicycle Properties and Optimizer
Provides optimized tube dimensions and angles for a standard bike frame while maximizing torsional rigidity and fatigue life (Basquin's Law). 
Also prints out tube lengths, torsional rigidity, and fatigue life for individual tubes and for the bike overall.
NOTE: This uses material properties for 4130 steel. Modify to suit your own needs.

![Bicycle Optimizer](https://github.com/ryan-aday/Basic_Vehicle_Dynamics/assets/33794562/aea4d195-83c8-4a1b-b332-747857bbbcc8)

Provide:
 - Parameters (constants):
   - wheel_diameter # in mm
   - wheel_thickness  # in mm
   - rear_center  # in mm (horizontal distance)
   - front_center # in mm (horizontal distance)
   - material_properties (constants):
    - shear_modulus # MPa for G
    - fatigue_strength_coefficient # MPa for sigma_f_prime
    - fatigue_strength_exponent # dimensionless for b

 - Optimized variables:
   - Tube dimensions (outer diameter, thickness) in mm
     - seat_tube
     - top_tube
     - head_tube
     - down_tube
     - seatstay
     - fork
   - seat_tube_length # in mm
   - top_tube_length # in mm
   - head_tube_length # in mm
   - seat_tube_angle # in degrees (referenced from vertical)
   - head_tube_angle  # in degrees (referenced from vertical)
   - fork_angle # in degrees (referenced from vertical)
   - top_tube_angle  # in degrees (referenced from horizontal)
   - bottom_bracket_drop  # in mm
    
Requires libraries:
 - numpy
 - matplotlib
 - mpl_toolkits
 - scipy
