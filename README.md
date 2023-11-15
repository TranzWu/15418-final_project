

<div align="center">
  <img src="media/water.gif" style="width:400px;"><br><br>
</div>

# Molecular Dynamics (MD) simulator for water using CUDA
Chunzhi Wu and Vincent Tombari

## URL
https://github.com/TranzWu/15418-final_project


## Summary

In this project, we are going implement a parallelized version of Molecular Dynamic engine to simulate water using CUDA.

## Background

Molecular Dynamics (MD) is a widely used and powerful computational technique that provides the insight into the dynamic behavior of molecular systems. It has been used wide in many different fields like drug discovery, materials science and biological systems.

Here's the pseudo code of Velocity Verlet algorithm, which is common used in molecular dynamics simulation to integrate the equations of motion.

```python

dt = time_step  # Time step
mass = particle_mass  # Mass of the particle

# Initial conditions
position = initial_position
velocity = initial_velocity
force = compute_force(position)  # Function to compute the force at the initial position

# Integration loop
for step in range(num_steps):
    # Velocity Verlet algorithm
    
    # Update position
    position += velocity * dt + 0.5 * force / mass * dt**2
    
    # Update force at the new position
    new_force = compute_force(position)
    
    # Update velocity
    velocity += 0.5 * (new_force) / mass * dt
    
    # Update force for the next iteration
    force = new_force
    
    # Additional computations or data collection can be included here
    
    # Output or store the results if needed
    print("Step {}: Position = {}, Velocity = {}".format(step, position, velocity))


```
