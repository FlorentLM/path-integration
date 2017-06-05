import numpy as np
from math import atan2

def generate_arena(arena_dimensions, walls_detection_dist=0, arena_offset=[0,0]):

    """ Generate an arena for the walled exploration experiment """

    arena = {'walls': np.array([[- arena_dimensions[0]/2 + arena_offset[0], arena_dimensions[0]/2 + arena_offset[0]], [- arena_dimensions[1]/2 + arena_offset[1], arena_dimensions[1]/2 + arena_offset[1]]]),
             'detection': walls_detection_dist
            }

    return arena


class Obstacle:
    """ Class representing an obstacle """
    def __init__(self, type):
        self.type = 'obstacle'

    def __str__(self):
        return "obstacle"


class Wall(Obstacle):
    """ Class representing a Wall obstacle. """

    def __init__(self, origin, end, detection=9.):
        """ Wall defined by origin and end points, and detection box """
        self.type = 'wall'
        self.origin = origin
        self.end = end
        self.detection = detection

        x1 = self.origin[0]
        y1 = self.origin[1]
        x2 = self.end[0]
        y2 = self.end[1]

        l1 = detection*2

        # Wall vector
        (vx, vy) = (x2 - x1, y2 - y1)

        # Normalized
        vlen = np.sqrt(vx**2 + vy**2)
        (v1x, v1y) = (vx / vlen, vy / vlen)

        # Perpendicular vector
        (u1x, u1y) = (-v1y, v1x)

        # Compute all 4 points coords
        (p1x, p1y) = (x1 - u1x * l1 / 2, y1 - u1y * l1 / 2)
        (p2x, p2y) = (p1x + u1x * l1, p1y + u1y * l1)
        (p4x, p4y) = (p1x + v1x * vlen, p1y + v1y * vlen)
        (p3x, p3y) = (p4x + u1x * l1, p4y + u1y * l1)

        self.detection_box = (p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)

        self.tilt = np.rad2deg(-atan2(-vx, -vy))-90

    def __str__(self):
        return "Wall obstacle"

class Cylinder(Obstacle):
    """ Class representing a cylindrical obstacle. """

    def __init__(self, center, radius, detection=7.):
        """ Cylinder obstacle defined by center, radius and detection box """
        self.type = 'cylinder'
        self.center = center
        self.radius = radius
        self.detection = detection
        self.detection_box = self.radius+self.detection

    def __str__(self):
        return "Cylinder obstacle"


def wall_detection(arena, pos, headings, t):

    """ For each t, check if in agent is in proximity of one of the four walls.
    If so, depending on the approach angle, add some clockwise or
    counter-clockwise rotation (to keep a smooth movement). """

    # Generate the walls
    walls = arena['walls'].copy()

    # Add their detection area
    walls[:,0] += arena['detection']
    walls[:,1] -= arena['detection']

    # Default turn increment sign (zero if no wall is detected)
    sign = 0

    # Actual wall-detection part

    if (walls[0,0] < pos[0]) == False:   # Left wall detected

        if np.deg2rad(-180) < ((headings[t-1] + np.pi) % (2.0 * np.pi) - np.pi) <= np.deg2rad(-90):
            sign = -1

        elif np.deg2rad(-90) < ((headings[t-1] + np.pi) % (2.0 * np.pi) - np.pi) < np.deg2rad(0):
            sign = 1


    if (pos[0] < walls[0,1]) == False:   # Right wall detected

        if np.deg2rad(0) < ((headings[t-1] + np.pi) % (2.0 * np.pi) - np.pi) <= np.deg2rad(90):
            sign = -1

        elif np.deg2rad(90) < ((headings[t-1] + np.pi) % (2.0 * np.pi) - np.pi) < np.deg2rad(180):
            sign = 1


    if (walls[1,0] < pos[1]) == False:   # Bottom wall detected

        if np.deg2rad(90) < ((headings[t-1] + np.pi) % (2.0 * np.pi) - np.pi) < np.deg2rad(180):
            sign = -1

        elif np.deg2rad(-180) <= ((headings[t-1] + np.pi) % (2.0 * np.pi) - np.pi) < np.deg2rad(-90):
            sign = 1


    if (pos[1] < walls[1,1]) == False:   # Top wall detected

        if np.deg2rad(0) < ((headings[t-1] + np.pi) % (2.0 * np.pi) - np.pi) < np.deg2rad(90):
            sign = 1

        elif np.deg2rad(-90) < ((headings[t-1] + np.pi) % (2.0 * np.pi) - np.pi) <= np.deg2rad(0):
            sign = -1

    # Obstacle detected, add rotation to the agent. 15 degrees is a good value
    # for smooth turns
    turn_increment = sign * np.deg2rad(15)

    return turn_increment

def obstacle_detection(obstacle, pos):

    """ Obstacle detection: If the agent is within the detection box, return the
    added rotation increment. If not, return 0. """

    # Make sure the position is an array (just in case)
    M = np.array(pos)

    # Default turn increment. Zero if no obstacle detected.
    turn_increment = 0

    if obstacle.type == 'cylinder':
        if np.sqrt((M[0] - obstacle.center[0])**2 + (M[1] - obstacle.center[1])**2) <= obstacle.detection_box:
            # Obstacle detected, add rotation to the agent. 15 degrees is a good
            # value for smooth turns
            turn_increment = np.deg2rad(15)

    elif obstacle.type == 'wall':
        A = obstacle.detection_box[0]
        B = obstacle.detection_box[1]
        C = obstacle.detection_box[2]
        D = obstacle.detection_box[3]

        AM = (M[0] - A[0]), (M[1] - A[1])
        AB = (B[0] - A[0]), (B[1] - A[1])
        AD = (D[0] - A[0]), (D[1] - A[1])

        if (0 < np.dot(AM, AB) < np.dot(AB, AB)) and (0 < np.dot(AM, AD) < np.dot(AD, AD)):
            # Obstacle detected, add rotation to the agent. 15 degrees is a good
            # value for smooth turns
            turn_increment = np.deg2rad(15)

    return turn_increment
