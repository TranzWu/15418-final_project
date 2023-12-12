# Molecular Dynamics (MD) simulator for water using CUDA (Project Milestone Report )
Chunzhi Wu and Vincent Tombari

## Work completed so far
So far, we've completed the input and output functions to read the initial positions and output the particle positions to the file, as well as the basic Velocity Verlet to solve the equations of motion of Lennard-Jones (LJ) particles. This includes:

1. Implementing the function to update the velocity of the particles.

2. Implementing the force function to calculate the interactions between each pair of particles within the simulation domain.

3. Implementing energy functions to calculate the total potential energy and kinetic energy of the system."

4. Ensuring the total energy of the system is conserved.

5. Adding periodic boundary conditions (PBC) to the system and update the corresponding force to ensure the total energy of the system is conserved. 

6. Adding bonds and angles to the system and updating the correcpoding energy and force functions to include these interactions. 

In addition, we have implemented majority of the code for our CUDA backend. Specifically, we have parallelized the force computations. However, we have been testing the force computation parallelization using the sequential functions for updating the velocity and positions, and thus are not receiving any speedup yet as we are constantly copying the data between the GPU memory and main memory.

## Progress Towards Goals and Deliverables
We believe we will be able to produce all of our goals and deliverables by the deadline. We are nearing the end, but both partners have lots of free time during finals week. In addition, we speculate we may add an ISPC backend to compare to our CUDA backend if time permits. We have realized that our project is not about simulating thousands or millions of particles, but rather looking at the small scale interactions between few particles. Thus, since there are only few particles and we are taking advantage of data parallelism over the particles, it may be effective to implement an ISPC backend.

## Plan for Poster Session
We plan to show graphs of the speedup computation and our analysis of the performance. We could do a live demo of the rendering of the particles, but this would require us to write our own parallel renderer which we do not believe we have time to do. We also plan to show videos of the particle simulations when we pass the frames to an off-the-shelf renderer.

## Updated Timeline

Here's our new goals and more detailed time lines:
| Time | Task| 
| -------- | -------- | 
| 12/11 - 12/13 | Optimize CUDA backend and implement ISPC backend |
| 12/13 | Run experiments analyzing speedup of CUDA and ISPC backend for different sets of hardware 
| 12/14 | Make Poster and Final Report |

## Preliminary results 

<div align="center">
  <img src="media/lj_particles.gif" style="width:4500px;"><br><br>
</div>

We are currently getting a negative speedup with our CUDA backend. We believe this is because every step of the iteration we are copying the computed forces, velocities, and updated posititions back to the CPU. Thus, we are bottlenecked by memory. We we will update to only communicate the updated positions.

## Planned Experiments

We plan to analyze the speedup of our backend on small and large numbers of particles. Time permitting, we plan to compare the speedup of the CUDA backend against the speedup of the ISPC backend on the different particle counts. 
