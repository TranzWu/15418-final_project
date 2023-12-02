# Molecular Dynamics (MD) simulator for water using CUDA (Project Milestone Report )
Chunzhi Wu and Vincent Tombari

## Work completed so far
So far, we've completed the input and output functions to read the initial positions and output the particle positions to the file, as well as the basic Velocity Verlet to solve the equations of motion of Lennard-Jones (LJ) particles. This includes:

1. Implementing the function to update the velocity of the particles.

2. Implementing the force function to calculate the interactions between each pair of particles within the simulation domain.

3. Implementing energy functions to calculate the total potential energy and kinetic energy of the system."

4. Ensuring the total energy of the system is conserved.

5. Adding periodic boundary conditions (PBC) to the system and update the corresponding force to ensure the total energy of the system is conserved. 

We are still working on adding bonds and angles to the system and update the correcpoding energy and force functions to include these interactions. This should be straight-forward and we are confident that we can get it finished within our posted timelines. 

Here's our new goals and more detailed time lines:
| Time | Task| 
| -------- | -------- | 
| 12/4 - 12/6 | Write kernel functions for force interactions (lj interactions)|
| 12/7 - 12/10 | Write kernel functions for force interactions (bond and angles)|
| 12/10 - 12/14 | Optimize the code, and extra functionalities if we still have time (e,g, radial distribution function), finish the final report |

The general issues that are facing is that for MD code, sometimes the code can be hard to debug (e.g. the unconserved total energy of the system can be attributed to multiple reasons, and it can be hard to pin down the exact reason why it doesn't work). So we need to be more patient and debug our code systematically. 

## preliminary results 

<div align="center">
  <img src="media/lj.gif" style="width:400px;"><br><br>
</div>

