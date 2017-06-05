from analysis import *

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return(theta, r)

def get_goal_xy_velocity(V, T_outbound):
    XYgoal = np.cumsum(V[:,:T_outbound,:], axis=1)
    Xgoal = XYgoal[:, -1, 0]
    Ygoal = XYgoal[:, -1, 1]
    return Xgoal, Ygoal


def compute_closest_to_feeder(V, T_outbound):
    X, Y = get_xy_from_velocity(V, T_outbound)
    Xgoal, Ygoal = get_goal_xy_velocity(V, T_outbound)

    for i in range(X.shape[0]):
        Y[i,:] -= Ygoal[i]
        X[i,:] -= Xgoal[i]

    D = np.sqrt(X**2 + Y**2)

    min_dists = np.nanmin(D, axis=1)

    d_mu = np.mean(min_dists)
    d_sigma = np.std(min_dists)
    return d_mu, d_sigma


def simple_angular_distance(v, goal, start=None, nb_steps=20, nest_radius=-1):

    if start is None:
        start = np.zeros(2)

    Xstart, Ystart = start[0], start[1]

    XY = np.cumsum(v, axis=0) + start
    X = XY[:,0]
    Y = XY[:,1]

    Xgoal, Ygoal = goal[0], goal[1]

    goal_angle = np.arctan2(-Xgoal, -Ygoal)

    D = np.sqrt(X**2 + Y**2)
    nest_leaving = np.argmax(D > nest_radius, axis=0)

    n = nest_leaving + nb_steps

    if v.shape[0] >= n:
        route_angles = np.arctan2(-X[n], -Y[n])

        return angular_distance(goal_angle, route_angles)
    else:
        return np.nan


def compute_simple_path_straightness(v, goal, return_dist=False):
    XY = np.cumsum(v, axis=0)
    X = XY[:,0]
    Y = XY[:,1]

    Xgoal, Ygoal = goal[0], goal[1]

    # Distances to the goal at each foodward point
    D = np.sqrt((X - Xgoal)**2 + (Y - Ygoal)**2)
    real_D = D[0]


    # Get shortest distance so far to nest at each time step
    # We make the y axis equal, by measuring in terms of proportion of
    # route distance.
    cum_min_dist = np.minimum.accumulate(D.T / real_D)


    # Get cumulative speed
    cum_speed = np.cumsum(np.sqrt((v[:,0]**2 + v[:,1]**2)), axis=0)

    # Now we also make the x axis equal in terms of proportion of distance
    # Time is stretched to compensate for longer/shorter routes
    cum_min_dist_norm = []

    xs = np.linspace(0, real_D*2, 500, endpoint=False)
    cum_min_dist_norm.append(np.interp(xs, cum_speed, cum_min_dist))

    if return_dist is True:
        return np.array(cum_min_dist_norm), real_D
    else:
        return np.array(cum_min_dist_norm)

def compute_tortuosity(cum_min_dist):
    """Computed with tau = L / C."""
    mu = np.nanmean(cum_min_dist, axis=1)
    tortuosity = 1.0 / (1.0 - mu[len(mu)/2])
    return tortuosity
