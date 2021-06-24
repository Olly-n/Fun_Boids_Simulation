# Fun Boids Simulations

## TL;DR
This project is just a little fun I had over a rainy weekend. I primarily made it to play around with boids simulations, agent-based modelling and make some aesthetically pleasing animations. It is a relatively efficient simulation using a K-D tree and NumPy arrays. I might try to extend it or do some real analysis on phase transitions in future.

![](Gifs/boids_2000_periodic_2.gif) ![](Gifs/boids_close_pack.gif)

## Basic theory
A boids simulation is a simulation of the flocking behaviours of a large group of animals such as bird/bats or fish. A boid is a simulated member of the flock and uses observations of other boids within a radius $S$ to determine its dynamics. In this simulation the dynamics are determined by 3 main factors which combine to create an effective force. These factors are: 
 1. Separation - a repulsive force that is based on the proximity to other boids around it (theoretically this should be inversely proportional to local separation but I found the results looked nicer with a direct proportionality but both are available in the code)
 2. Alignment - an attractive force that aligns the boids velocity with that  of the local group
 3. Cohesion - a force that pulls towards the centre of mass of the local group that in effect helps attract local boids together

Each of these effective forces has a weighting factor that can be changed to give them more or less strength during the calculations of apparent force. By following these simple rules the flock can exhibit very complex behaviours which seem to very accurately mimic real world animals.

## Implementation
This code uses ideas from agent-based modelling, where each boid is an individual and a flock is build from many individuals which are capable of interacting with each other. Using agents has the additional benefit that it is very easy to introduce new properties to the agents at a later date. 
The project has been set up so that it is very easy to play around with (the basic simulations running in two lines) and lots of parameters to fiddle with such as different boundary conditions (periodic, reflective and repulsive), interaction strengths and sight distance. The boids are initialised with positions from a uniform distribution over the space and velocities from a standard normal distribution. At each time step all the boids are displaced by the velocity * dt (the defined time stepsize) and a new velocity is found using the acceleration based on the forces.
The code is relatively well optimised using a K-D tree for nearest neighbour retrieval (this is log(n) retrieval and construction for each boid in the flock) and vectorised NumPy arrays for calculations. It takes ~1 minute to run a 10 second simulation and animation with 1500 boids (on my 8 year old 13 inch MacBook pro) and will happily run 300-400 boids in near real time, although, it is worth noting that run time is very parameter dependent.

## Running
The code is written in python 3.6 and the requirements can be found in the requirements.txt file.
To run the code just import the module, create a flock using the `Flock` class and then use either `animate_flock_quiver` or `animate_flock_scatter` functions to animate. Alternatively, uncomment the last two functions of the script and change the parameter there and then run directly in the terminal/cmd.

## Files
`boids.py`: Code for boids simulation

`Gifs`: Contains animations of boid simulations

`README.md`: This README

`requirements.txt`: Requirements


## Future work

### Ideas
 - Add different types of boids to simulation (i,e predators or flock leaders)
 - There are obvious phase transitions within the flock over time, would be interesting to investigate these and what parameters cause them.
 - Try out different effective force dynamics (ie random cohesion/separation/alignment strengths or different types of forces)
 - Make an interactive element where you can interact with simulation ( using pygame maybe)
 - Place a flock in a maze and see how that affects dynamics and how the flock reacts
 - Currently this is completely deterministic, try adding random noise into dynamics
### Todo
 - Get coherence force to work over periodic boundaries (doesn't make much difference for now).
 - Build a Google colab notebook with some ready made simulations.
 - Swap to a more flexible animation tool (with smaller videos saves, ~25 MB seems large, and COLOUR)

## Boid gif dump
Just a collection of cool looking simulations:

![](Gifs/boids_2000_periodic_1.gif)
![](Gifs/boids_4000.gif)
![](Gifs/boids_3000.gif)
![](Gifs/boids_3.gif)
![](Gifs/boids_4.gif)
![](Gifs/boids1.gif)
![](Gifs/boids2.gif)
![](Gifs/boids_7.gif)
![](Gifs/boids_final.gif)

