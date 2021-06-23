import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
from copy import deepcopy

class Boid():
    """An individual boid.

    A boid is a simulated animal within a flock. Its dynamics are based on the 
    dynamics of its local group (i.e. the boids around).
    """

    def __init__(
        self,
        position_vec,
        velocity_vec,
        max_speed,
        max_acceleration,
        boundary,
        boundary_type,
        sight_radius,
        seperation_factor,
        alignment_factor,
        cohesion_factor,
    ):
        """Boid installation function.

        Args:
            position_vec (2D array/list): initial position of the boid.
            velocity_vec (2D array/list): initial velocity of the boid .
            max_speed (float): max speed of the boid.
            max_acceleration (float): max acceleration of the boid.
            boundary (tuple(float)): boundary box positions of the simulation (bottom, top, left, right).
            boundary_type (str): boundary conditions to use. Choose from "periodic", "reflective" or "repulsive".
            sight_radius (float): distance boid can see over.
            seperation_factor (float): strength of separation force.
            alignment_factor (float): strength of alignment force.
            cohesion_factor (float): strength of the cohesion force.
        """

        self.position = np.array(position_vec)
        self.velocity = np.array(velocity_vec)
        self.acceleration = np.zeros(2)

        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.sight_radius = sight_radius

        self.seperation_factor = seperation_factor
        self.alignment_factor = alignment_factor
        self.cohesion_factor = cohesion_factor

        self.boundary = boundary
        self.boundary_type = boundary_type
        if boundary_type is "periodic":
            self.boundary_condition = self.periodic
        elif boundary_type is "reflective":
            self.boundary_condition = self.reflective
        elif boundary_type is "repulsive":
            self.boundary_condition = self.repulsive
        else:
            raise Exception("""Please choose boundary conditions from "periodic", "reflective" or "repulsive".""")

        self.mag = lambda x: (x[0]**2 + x[1]**2)**0.5

    def seperation_inv(self, local_flock, relative_disp):
        """Applies separation force based on proximity to other boids."""
        if len(local_flock) != 1:
            bottom, top, left, right = self.boundary
            local_push = relative_disp[local_flock]
            # local_push += np.random.uniform(-1e-7,1e-7,local_push.shape)
            local_push = local_push[~np.all(local_push == 0, axis=1)]
            if self.boundary_type == "periodic":
                local_push[local_push[:,0] > self.sight_radius] -= [abs(right - left), 0]
                local_push[local_push[:,1] > self.sight_radius] -= [0, abs(top - bottom)]
                local_push[local_push[:,0] < -self.sight_radius] += [abs(right - left), 0]
                local_push[local_push[:,1] < -self.sight_radius] += [0, abs(top - bottom)]

            norms = np.expand_dims(np.linalg.norm(local_push, axis=1),axis=1)
            final_push = local_push * 1/norms
            self.acceleration -= self.seperation_factor * np.mean(final_push, axis = 0)

    def seperation_dir(self, local_flock, relative_disp):
        """Applies separation force based on proximity to other boids."""
        bottom, top, left, right = self.boundary
        local_push = relative_disp[local_flock]
        # local_push += np.random.uniform(-1e-7,1e-7,local_push.shape)
        if self.boundary_type == "periodic":
            local_push[local_push[:,0] > self.sight_radius] -= [abs(right - left), 0]
            local_push[local_push[:,1] > self.sight_radius] -= [0, abs(top - bottom)]
            local_push[local_push[:,0] < -self.sight_radius] += [abs(right - left), 0]
            local_push[local_push[:,1] < -self.sight_radius] += [0, abs(top - bottom)]

        self.acceleration -= self.seperation_factor * np.mean(local_push, axis = 0)


    def align(self, local_flock, boid_list):
        """Applies alignment force based on velocities of local boids."""
        local_boids = boid_list[local_flock]
        local_velocity = np.array([0., 0.])
        for boid in local_boids:
            local_velocity += boid.velocity
        self.acceleration += self.alignment_factor * local_velocity/self.mag(local_velocity)

    def cohesion(self, local_flock, flock_pos):
        """Applies cohesion force based on centre of mass of local boids."""
        local_disp = flock_pos[local_flock]
        # print(local_disp)

        self.acceleration += self.cohesion_factor * (np.mean(local_disp, axis=0) - self.position)

    def periodic(self):
        """Periodic boundary condition."""
        bottom, top, left, right = self.boundary
        if self.position[0] > right:
            self.position[0] = left
        if self.position[0] < left:
            self.position[0] = right
        if self.position[1] > top:
            self.position[1] = bottom
        if self.position[1] < bottom:
            self.position[1] = top

    def reflective(self):
        """Reflective boundary condition."""
        bottom, top, left, right = self.boundary
        if self.position[0] >= right:
            self.position[0] = right
            self.velocity[0] *= -1.
            self.acceleration[0] *= -1.
        elif self.position[0] <= left:
            self.position[0] = left
            self.velocity[0] *= -1.
            self.acceleration[0] *= -1.
        elif self.position[1] >= top:
            self.position[1] = top
            self.velocity[1] *= -1.
            self.acceleration[1] *= -1.
        elif self.position[1] <= bottom:
            self.position[1] = bottom
            self.velocity[1] *= -1.
            self.acceleration[1] *= -1.

    def repulsive(self):
        """Repulsive boundary condition.

        Note: horizontal_repel, vertical_repel and repulsion_strength
              are arbitrary. Feel free to play around with them.
              Additionally, adding a reflect after the calculation can stop 
              boids drifting off the plot but can look a little unnatural.
        """
        bottom, top, left, right = self.boundary
        horizontal_repel = (right - left) * 0.05
        vertical_repel = (top - bottom) * 0.05
        repulsion_strength = 0.075

        if self.position[0] + horizontal_repel >= right:
            self.velocity[0] -= self.max_speed*repulsion_strength
        if self.position[0] - horizontal_repel <= left:
            self.velocity[0] += self.max_speed*repulsion_strength
        if self.position[1] + vertical_repel >= top:
            self.velocity[1] -= self.max_speed*repulsion_strength
        if self.position[1] - vertical_repel <= bottom:
            self.velocity[1] += self.max_speed*repulsion_strength
        # self.reflective()

    def step(self, dt):
        """Updates boid's position and velocity based on current acceleration."""
        acceleration_mag = self.mag(self.acceleration)
        if acceleration_mag > self.max_acceleration:
            self.acceleration = self.max_acceleration * self.acceleration/acceleration_mag

        self.position += self.velocity * dt
        self.boundary_condition()
        self.velocity += self.acceleration * dt

        speed = self.mag(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.max_speed * self.velocity/speed
        self.acceleration = np.zeros(2)

class Flock():
    """A flock of boids

    A simulated flock of boids. Creates a list of boids and stores their positions in a KD tree for 
    quick neighbourhood evaluation.
    """

    def __init__(
        self,
        N,
        max_speed=2.,
        max_acceleration=1.,
        boundary=(-20., 20., -20., 20.),
        boundary_type="Periodic",
        sight_radius=2.,
        seperation_factor=1.,
        alignment_factor=1.,
        cohesion_factor=1.,
    ):
        """Creates a flock of boids.

        Args:
            N (int): number of boids in the flock.
            **See boid class for other parameters**
        """
        self.N = N
        self.boid_list = np.empty(N, dtype=Boid)
        self.boundary = boundary
        bottom, top, left, right = boundary
        self.height = top-bottom
        self.width = right-left
        self.boundary_type = boundary_type

        self.flock_pos = min(self.height, self.width)*0.5*np.random.uniform(-1, 1, size=(self.N, 2))
        if boundary_type is "periodic":
            self.epsilon = 1e-7
            self.flock_pos_tree = cKDTree(self.flock_pos+[0.5*self.height,0.5*self.width],
                boxsize = [self.height+self.epsilon,self.width+self.epsilon])
        else:
            self.flock_pos_tree = cKDTree(self.flock_pos)

        # self.flock_pos = np.array([[0,19],[0,-19]])
        self.flock_sight_radius = sight_radius

        for i in range(N):
            self.boid_list[i] = Boid(
                self.flock_pos[i], 
                np.random.randn(2),
                max_speed,
                max_acceleration,
                boundary,
                boundary_type,
                sight_radius,
                seperation_factor,
                alignment_factor,
                cohesion_factor,
            )

    def calc_acceleration(self):
        """Calculates the acceleration felt by each boid in the flock.
            Change separation function in here. 
        """
        local_list = self.flock_pos_tree.query_ball_tree(self.flock_pos_tree, self.flock_sight_radius)
        for local_flock, boid in zip(local_list, self.boid_list):
            relative_disp = self.flock_pos - boid.position

            boid.seperation_dir(local_flock, relative_disp)
            boid.align(local_flock, self.boid_list)
            boid.cohesion(local_flock, self.flock_pos)

    def step(self, dt):
        """Calculates the acceleration of each boid in the flock and then updates their positions and velocities accordingly"""
        self.calc_acceleration()

        for i, boid in enumerate(self.boid_list):
            boid.step(dt)
            self.flock_pos[i] = boid.position

        if self.boundary_type is "periodic":
            self.flock_pos_tree = cKDTree(self.flock_pos+[0.5*self.height,0.5*self.width],
                boxsize = [self.height+self.epsilon, self.width+self.epsilon])
        else:
            cKDTree(self.flock_pos)
        return self.flock_pos


def animate_flock_scatter(flock, step_size=0.1, frames=100, pre_compute=False, continuous=True):
    """Animate a flock using a scatter diagram.

    If pre-compute is set to true the simulation will run at 20 fps.

    Args:
        flock (Flock): flock to be animated.
        step_size (float): size of time step taken.
        frames (int): number of frames to animate.
        pre_compute (bool): weather to per-compute frames before runtime
        continuous (bool): set true to run with no frame limit or to loop if frame limit is set
    """
    fig = plt.figure()
    bottom, top, left, right = flock.boundary
    ax = plt.axes(xlim=(1.1*bottom, 1.1*top), ylim=(1.1*left, 1.1*right))

    if pre_compute and frames is None:
        raise Exception("""Please set number of frames to pre-compute.""")

    positions, = plt.plot([], [], marker=",")
    boid_scatter_plot = ax.scatter(flock.flock_pos[:,0], flock.flock_pos[:,1])
    if pre_compute:
        steps = tuple(deepcopy(flock.step(step_size)) for _ in range(frames))

    def animate(i):

        if i == frames - 1 and continuous is False:
            print("animation ended.")
            plt.close(fig)

        if pre_compute:
            flock_pos = steps[i]
        else:
            flock_pos = flock.step(step_size)

        boid_scatter_plot.set_offsets(flock_pos)
        return boid_scatter_plot

    if pre_compute:
        interval = 50
    else:
        interval = 1

    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=interval, blit=False)
    # plt.show()

def animate_flock_quiver(flock, step_size=0.1, frames=None, pre_compute=False, continuous=True, save = False):
    """Animate a flock using a quiver plot showing that shows the boids' velocities as well as positions.

    Args:
        flock (Flock): flock to be animated.
        step_size (float): size of time step taken.
        frames (int): number of frames to animate.
        pre_compute (bool): whether to per-compute frames before runtime
        continuous (bool): set true to run with no frame limit or to loop
        save (bool): wheather to save the animation or not
    """
    fig = plt.figure()
    bottom, top, left, right = flock.boundary
    ax = plt.axes(xlim=(1.1*bottom, 1.1*top), ylim=(1.1*left, 1.1*right))

    if pre_compute and frames is None:
        raise Exception("""Please set number of frames to pre-compute.""")

    velocity = np.array([boid.velocity for boid in flock.boid_list])
    boid_vector_plot = ax.quiver(flock.flock_pos[:,0], flock.flock_pos[:,1], velocity[:,0], velocity[:,1])

    if pre_compute:
        steps = tuple((deepcopy(flock.step(step_size)), np.array([boid.velocity for boid in flock.boid_list])) for _ in range(frames))

    def animate(i):

        if i == frames - 1 and continuous is False:
            print("animation ended.")
            plt.close(fig)

        if pre_compute:
            flock_pos = steps[i][0]
            velocity = steps[i][1]
        else:
            flock_pos = flock.step(step_size)
            velocity = np.array([boid.velocity for boid in flock.boid_list])

        boid_vector_plot.set_offsets(flock_pos)
        boid_vector_plot.set_UVC(velocity[:,0],velocity[:,1])
        return boid_vector_plot

    if pre_compute:
        interval = 50
    else:
        interval = 1

    ani = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=interval, blit=False)

    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    if save is True:
        f = r"Boids Project/Gifs/new_boids_sim.gif"
        writergif = animation.PillowWriter(fps=20)
        ani.save(f, writer=writergif)

    plt.show()


# flock = Flock(N=1500, 
#             max_speed=2.5,
#             max_acceleration=1.,
#             boundary_type="periodic",
#             boundary=(-20., 20., -20., 20.),
#             sight_radius=4.,
#             seperation_factor=30.,
#             alignment_factor=30.,
#             cohesion_factor=30.,
#             )

# animate_flock_quiver(flock, frames=300, pre_compute=True, continuous=True, save=True)