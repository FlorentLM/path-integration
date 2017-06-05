import numpy as np
from scipy.signal import lfilter
from scipy.interpolate import interp1d
from math import atan2

import bee_simulator
import central_complex
import cx_rate

from environment import *
import trials2

# Define some parameters (based on Stone et al.)
default_acc = 0.15
default_drag = 0.15

noise = 0.1

cx = cx_rate.CXRatePontinHolonomic(noise=noise)   # Latest CX model with pontine cells + holonomic movement
CXLogger = trials2.CXLogger

def compute_distances(pos, goal_coord=np.array([[0., 0.]])):

    """Get the absolute distances between the agent's current position and one
    or multiple goals.
    If no goal is passed, the nest is assumed to be the goal.

    This coordinates-based information is used for 'visual catchment' area only,
    and is *not* used by the central complex in any way for Path Integration."""

    # Make sure the coordinates are in the form of arrays
    pos = np.array(pos)
    goal_coord = np.array(goal_coord)

    # If the array is just 2D, add one dimension to make the function universal
    # (this is just in case)
    if goal_coord.shape == (2,):
        goal_coord = goal_coord[np.newaxis,:]

    computed_distances = np.sqrt((pos[0] - goal_coord[:,0])**2 + (pos[1] - goal_coord[:,1])**2)
    return computed_distances


def get_index(computed_distances, catchment_area=1e-9):

    """ Returns the index of the feeder on which the agent is. """

    if len(np.where(computed_distances <= catchment_area)[0]) == 0:
        return None
    else:
        return int(np.where(computed_distances <= catchment_area)[0])


def feeders_detection(pos, goal_coord, catchment_area=1e-9):

    """ Feeder detection: relative distances to goal(s) are computed and if the
    agent is inside the catchment area of the goal (or one of the goals),
    return which one.

    Otherwise, no feeder detected, return None. """

    computed_distances = compute_distances(pos, goal_coord)

    if any(computed_distances <= catchment_area):               # If agent is within any catchment area...
        ind = get_index(computed_distances, catchment_area)     # ...determine which one
        return index
    else:
        return None


def initialize_route(T_max, current_heading=0.0, current_velocity=np.array([0.0, 0.0])):
    """ Set the initial values for a move (either outbound or inbound). Can be
    chained to a previous move through current heading and current velocity. """

    headings = np.zeros(T_max)
    velocity = np.zeros([T_max, 2])
    headings[0] = current_heading
    velocity[0, :] = current_velocity

    return headings, velocity


def generate_random_rotations(T_max, mu=0.0, kappa=100.0):
    """ Generate a series of smoothed random rotations, for use in random walk
    movements """

    vm = np.random.vonmises(mu, kappa, T_max)
    rotations = lfilter([1.0], [1, -0.4], vm)
    rotations[0] = 0.0

    return rotations


def generate_speed(T_max, vary=True, mean_acc=default_acc, max_acc=default_acc, min_acc=0.0):
    """ Generate a varying acceleration value """

    # Randomly sample some points within acceptable acceleration and
    # interpolate to create smoothly varying speed.

    if vary:
        if T_max > 200:
            num_key_speeds = T_max/50
        else:
            num_key_speeds = 4
        x = np.linspace(0, 1, num_key_speeds)
        y = np.random.random(num_key_speeds) * (max_acc - min_acc) + min_acc
        f = interp1d(x, y, kind='cubic')
        xnew = np.linspace(0, 1, T_max, endpoint=True)
        varying_acceleration = f(xnew)
    else:
        varying_acceleration = mean_acc * np.ones(T_max)

    return varying_acceleration


def move(goal_coord=None, start_coord=np.array([0.0, 0.0]), T_max=50000, goal_radius=0,
         tb1=None, memory=None, ltm=None, random_exploring=False, multiple_stops=False, food_quantity=None,
         current_heading=0.0, current_velocity=np.array([0.0, 0.0]), keep_searching=False,
         stop_when_food=False, arena=False, obstacle=False, return_success=False,
         default_acc=default_acc, drag=default_drag, turn_sharpness=1.0,
         logging=False, bump_shift=0.0, inhib_gain=1.0, filtered_steps=0.0):

    """ Main function to perform a move from a point A to a point B.
    Can be memory-guided, integrator-guided (consumption of a non-zero integrator),
    or a random exploratory walk.
    In all cases, no distinction is needed between outbound or inbound."""

    if goal_coord is None:
        goal_coord = np.array([float('inf'), float('inf')])
    elif goal_coord is 'Nest':
        goal_coord = np.array([[0.0, 0.0]])
    else:
        goal_coord=np.array(goal_coord)

    # Initialize headings and velocity for the T max
    headings, velocity = initialize_route(T_max, current_heading=current_heading, current_velocity=current_velocity)

    # Set random rotations and acceleration if a random walk is needed
    rand_rotation = generate_random_rotations(T_max)
    varying_acceleration = generate_speed(T_max, vary=True)

    # Set the zero-state to the integrator and to the memory if they are not to
    # be used
    if memory is None:
        memory = 0.5 * np.ones(central_complex.N_CPU4)
    if tb1 is None:
        tb1 = np.zeros(central_complex.N_TB1)
    if ltm is None:
        ltm = 0.5 * np.ones(central_complex.N_CPU4)

    # If logging, initialize the log
    if logging:
        cx_log = CXLogger(0, T_max+1, cx)
    else:
        cx_log = None

    # If the goal array is just 2D, add one dimension to make the function universal
    if goal_coord.shape == (2,):
        goal_coord = goal_coord[np.newaxis,:]
    nb_feeders = int(goal_coord.shape[0])

    if multiple_stops == True and food_quantity is None:
        food_quantity = np.ones(nb_feeders)/nb_feeders

    # Initialize start position
    pos = np.array(start_coord)
    at_start = True
    ind = None
    crop_filling = 0.0
    success = 0
    detection = []

    # Initialize catchment area time marker
    z = 0

    for t in range (1, T_max):

        # Compute the distance between the agent and each feeder at this timestep
        computed_dist = compute_distances(pos, goal_coord)

        # Set the according state
        if any(computed_dist <= goal_radius):
            ind = get_index(computed_dist, goal_radius)

            # If still at start, ignore this catchment area
            if at_start:
                state = 'ignore'
            elif computed_dist[ind] <= 1e-9:
                if multiple_stops == False:
                    state = 'feeder'
                else:
                    if food_quantity[ind] > 0.0:
                        state = 'feeder'
                    elif food_quantity[ind] <= 0.0:
                        state = 'ignore'
            else:
                state = 'approaching'
        else:
            state = 'searching'

        # Now, based on state, execute the appropriate move

        if state == 'searching': # 'searching' means that agent left start area
            at_start = False

            # Reinitialize the markers
            ind = None
            z = 0

        if state == 'feeder': # 'feeder' means that agent in ON the feeeder
            z += 1 # Increment the z marker to allow correct value retrieval in next time step
            if multiple_stops == False:
                success = 1
                break
            else:
                crop_filling += food_quantity[ind]
                food_quantity[ind] = 0.0
                at_start = True

        if state == 'approaching': # 'approaching' means that agent in the area
            # Guidance is NOT operated by the path integrating system, the agent
            # heads toward the feeder directly (can be thought of visual navigation)

            if keep_searching == True:
                detection.append(t)
                state = 'ignore'
            else:
                z += 1 # Increment the area marker to allow correct value retrieval in next time step

                # Compute remaining distance between the point of entry in the zone and the center
                # (point of entry is z timesteps ago, so retrieve the t-z value)
                dist_to_center = (goal_coord[ind,:] - np.array(np.cumsum(velocity, axis=0)[t-z])-start_coord)

                # Execute the move for this time step
                headings[t] = atan2(dist_to_center[0], dist_to_center[1])  # Head directly toward center
                velocity[t] = dist_to_center/goal_radius                     # Direct distance divided by nb of steps (area radius)


        if state == 'searching' or state == 'ignore':

            r = headings[t-1] - headings[t-2]
            r = (r + np.pi) % (2 * np.pi) - np.pi

            # Generate the cells activity based on previous movement
            tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor, cpu4_inh = trials2.update_cells(
                heading=headings[t-1] + np.sign(r) * bump_shift,  # Remove sign to use proportionate shift
                velocity=velocity[t-1],
                tb1=tb1,
                memory=memory,
                cx=cx,
                ltm=ltm,
                filtered_steps=filtered_steps,
                inhib_gain=inhib_gain)



            # If random exploration, output is not driven by the CX motor command
            if random_exploring == True:
                rotation = rand_rotation[t]
                acceleration = varying_acceleration[t]

            # If not, motor command drives the output
            else:
                rotation = turn_sharpness * motor
                acceleration = default_acc

            # Include reaction to the environment (arena walls and/or object avoidance)
            if arena is not False:
                rotation += wall_detection(arena, pos, headings, t)

            if obstacle is not False:
                test = obstacle_detection(obstacle, pos)
            else:
                test = 0

            # Execute the move for this time step (either guided by CX or random rotation)

            if test > 0:
                headings[t], velocity[t, :] = np.arctan2(-pos[0], -pos[1]), -pos/(T_max-t)
            else:
                headings[t], velocity[t, :] = bee_simulator.get_next_state(headings[t-1],
                                                                       velocity[t-1, :],
                                                                       rotation,
                                                                       acceleration,
                                                                       drag=drag)
            if logging:
                    cx_log.update_log(t, tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor, cpu4_inh)

        # Finally, compute position at this time step (to use in t+1)
        pos = start_coord + np.cumsum(velocity, axis=0)[t]

    if return_success == True:
        return headings[:t], velocity[:t], t, cx_log.trim(start=1, end=t-z), success
    else:
        return headings[:t], velocity[:t], t, cx_log.trim(start=1, end=t-z), detection


def add_memories(memory_1, memory_2):
    """Basic addition of two memory states (vector addition).
    Returns the shortcut vector.
    Not used in the 'inhibition' approach. """

    base_activity = 0.5 * np.ones(central_complex.N_CPU4)
    mem_1_normalized = memory_1 - base_activity
    mem_2_normalized = memory_2 - base_activity

    shortcut_mem = np.clip(base_activity + mem_1_normalized + mem_2_normalized, 0, 1)

    return shortcut_mem


def get_amplitude(memory):
    """Get rough amplitude of the population activity """

    idx_min = np.argpartition(memory,2)[:2]
    idx_max = np.argpartition(memory,-2)[-2:]

    sigmin = np.cumsum(memory[idx_min[:2]])[-1]/2
    sigmax = np.cumsum(memory[idx_max[:2]])[-1]/2

    amplitude = sigmax - sigmin

    return amplitude


def direct_path(point_a, point_b=np.array([0.0, 0.0])):
    """ Simple metric distance between two points in cartesian coordinates.
    If second point is not passed, origin (nest) is assumed."""

    x1 = point_a[0]
    x2 = point_b[0]
    y1 = point_a[1]
    y2 = point_b[1]

    direct_path = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return direct_path


def adjust_memory(memory, current_memory):
    """Adjust stored memory based on current integrator level
    OLD VERSION, doesn't use the inhibitory connections
    (but works on the same principle)"""

    base_state = 0.5 * np.ones(central_complex.N_CPU4)

    ampl_of_stored_mem = memory - base_state
    ampl_of_current_mem = current_memory - base_state

    k = 0.25

    updated_mem = base_state + (ampl_of_stored_mem - k*ampl_of_current_mem)

    return np.clip(updated_mem, 0, 1)


def get_estimated_distance(cpu4=None, ltm=None, cx=cx, gain=1.0):
    """Get an estimated distance from current position to recalled memory's associated location.
    Uses the inhibitory connections to generate the compound new vector"""

    # Set to zero-state if either one is not given
    if ltm is None:
        ltm = 0.5 * np.ones(central_complex.N_CPU4)
    if cpu4 is None:
        cpu4 = 0.5 * np.ones(central_complex.N_CPU4)

    new_vector = cx.cpu4_inhibition(cpu4, ltm, gain=gain)

    angle, distance = cx.decode_cpu4(new_vector)

    return distance


def get_probabilities(memories, current_mem=None, min_distance=0.0, max_distance=float('inf'), to_ignore=None, use_food=False, return_distances=False):
    estim_dist = []
    probs = []
    ignored = np.array(to_ignore)

    # If one or more memories are available
    if memories is not None and len(memories) > 0:
        for entry in memories:
            if entry not in ignored:
            #if entry[0] not in ignored: # Use this if the memories list includes food quantity
                #dist = get_estimated_distance(current_mem, entry[1]) # Use this if the memories list includes food quantity
                dist = get_estimated_distance(current_mem, entry)
            else:
                dist = float('nan')
            estim_dist.append(dist)

        s=sum([1/float(dist) for dist in estim_dist if max_distance > dist > min_distance])

        for dist in estim_dist:
            if max_distance > dist > min_distance:
                prob = 1/float(dist) / s
            else:
                prob = 0.0
            probs.append(prob)

        if use_food:
            weights = [entry[2] for entry in memories]
            tmp = [p*w for p,w in zip(probs, weights)]
            output_probs = [float(t)/sum(tmp) for t in tmp]
        else:
            output_probs = probs

    # If no memory available, return nothing
    else:
        estim_dist = None
        output_probs = None

    if return_distances:
        return output_probs, estim_dist
    else:
        return output_probs


def get_next_goal(memories, current_mem, min_distance=0.0, max_distance=float('inf'), to_ignore=None, use_food=False, mode='max'):

    """Choose next goal based on available memories
    If current feeder index is given, it will be ignored from the choice."""

    probs, dist = get_probabilities(memories=memories, to_ignore=to_ignore, current_mem=current_mem, min_distance=min_distance, max_distance=max_distance, use_food=use_food, return_distances=True)

    if probs is not None and sum(probs) == 1:

        # Use these probabilities to choose next goal
        if mode == 'rand':
            choice = np.random.choice(range(len(memories)))

        elif mode == 'weight':
            choice = np.random.choice(range(len(memories)), p=probs)

        elif mode == 'max':
            choice = np.argmax(probs)

        #selection = memories[choice][1] # Use this if the memories list includes food quantity
        #index =  memories[choice][0] # Use this if the memories list includes food quantity
        selection = memories[choice]
        index =  [choice]

        return selection, index
    else:
        return None, None
