from trials import *


W_LTM_CPU4 = np.array([
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])

class CXLogger(object):
    """Class to store logs in of central complex cell activations.
    With added support for CPU4 inhibition"""

    def __init__(self, T_outbound, T_inbound, cx=None):
        """Initialise log as many zerod out numpy arrays."""
        self.T_outbound = T_outbound
        self.T_inbound = T_inbound
        T = T_outbound + T_inbound
        self.T = T
        self.cx = cx
        if issubclass(cx.__class__, cx_basic.CXBasic):
            self.tl2 = np.empty([1, T])
            self.cl1 = np.empty([1, T])
        else:
            self.tl2 = np.empty([central_complex.N_TL2, T])
            self.cl1 = np.empty([central_complex.N_CL1, T])
        self.tb1 = np.empty([central_complex.N_TB1, T])
        self.tn1 = np.empty([central_complex.N_TN1, T])
        self.tn2 = np.empty([central_complex.N_TN2, T])
        self.memory = np.empty([central_complex.N_CPU4, T])
        self.cpu4 = np.empty([central_complex.N_CPU4, T])
        self.cpu1 = np.empty([
            central_complex.N_CPU1A + central_complex.N_CPU1B, T
        ])
        self.motor = np.empty(T)
        self.cpu4_inh = np.empty([central_complex.N_CPU4, T])

    def update_log(self, t, tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1=np.nan,
                   motor=np.nan, cpu4_inh=np.nan):
        """Add the latest value to each cell type."""
        self.tl2[:, t] = tl2
        self.cl1[:, t] = cl1
        self.tb1[:, t] = tb1
        self.tn1[:, t] = tn1
        self.tn2[:, t] = tn2
        self.memory[:, t] = memory
        self.cpu4[:, t] = cpu4
        self.cpu1[:, t] = cpu1
        self.motor[t] = motor
        self.cpu4_inh[:, t] = cpu4_inh

    def trim(self, start=None, end=None):
        if start and end:
            end += 1
            newlength = end - start
        elif start is None:
            end += 1
            newlength = self.T - end
        else:
            newlength = self.T - start
        trimmed = CXLogger(T_outbound=newlength, T_inbound=0, cx=self.cx)
        trimmed.tl2 = self.tl2[:, start:end]
        trimmed.cl1 = self.cl1[:, start:end]
        trimmed.tb1 = self.tb1[:, start:end]
        trimmed.tn1 = self.tn1[:, start:end]
        trimmed.tn2 = self.tn2[:, start:end]
        trimmed.memory = self.memory[:, start:end]
        trimmed.cpu4 = self.cpu4[:, start:end]
        trimmed.cpu1 = self.cpu1[:, start:end]
        trimmed.motor = self.motor[start:end]
        trimmed.cpu4_inh = self.cpu4_inh[:, start:end]
        trimmed.T = trimmed.T_outbound + trimmed.T_inbound
        return trimmed

    def __add__(self, other):
        """Combine two logs into one big one (normally outbound and
        inbound)."""
        combined = CXLogger(T_outbound=self.T, T_inbound=other.T-1, cx=self.cx)
        combined.tl2[:, :self.T] = self.tl2
        combined.cl1[:, :self.T] = self.cl1
        combined.tb1[:, :self.T] = self.tb1
        combined.tn1[:, :self.T] = self.tn1
        combined.tn2[:, :self.T] = self.tn2
        combined.memory[:, :self.T] = self.memory
        combined.cpu4[:, :self.T] = self.cpu4
        combined.cpu1[:, :self.T] = self.cpu1
        combined.motor[:self.T] = self.motor
        combined.cpu4_inh[:, :self.T] = self.cpu4_inh

        # Here we skip the first element of inbound as duplicate (clumsy
        # coding, think of fix)
        combined.tl2[:, self.T:] = other.tl2[:, 1:]
        combined.cl1[:, self.T:] = other.cl1[:, 1:]
        combined.tb1[:, self.T:] = other.tb1[:, 1:]
        combined.tn1[:, self.T:] = other.tn1[:, 1:]
        combined.tn2[:, self.T:] = other.tn2[:, 1:]
        combined.memory[:, self.T:] = other.memory[:, 1:]
        combined.cpu4[:, self.T:] = other.cpu4[:, 1:]
        combined.cpu1[:, self.T:] = other.cpu1[:, 1:]
        combined.motor[self.T:] = other.motor[1:]
        combined.cpu4_inh[:, self.T:] = other.cpu4_inh[:, 1:]
        return combined


def update_cells(heading, velocity, tb1, memory, cx, ltm, filtered_steps=0.0, inhib_gain=1.0):
    """Generate activity for all cells, based on previous activity and current
    motion."""

    # Compass
    tl2 = cx.tl2_output(heading)
    cl1 = cx.cl1_output(tl2)
    tb1 = cx.tb1_output(cl1, tb1)

    # Speed
    flow = cx.get_flow(heading, velocity, filtered_steps)
    tn1 = cx.tn1_output(flow)
    tn2 = cx.tn2_output(flow)

    # Update memory for distance just travelled
    memory = cx.cpu4_update(memory, tb1, tn1, tn2)
    cpu4 = cx.cpu4_output(memory)

    cpu4_inh = cx.cpu4_inhibition(cpu4, ltm, gain=inhib_gain)

    # Steer based on memory and direction
    cpu1 = cx.cpu1_output(tb1, cpu4_inh)
    motor = cx.motor_output(cpu1)

    return tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor, cpu4_inh


def generate_memory(headings, velocity, cx, bump_shift=0.0, filtered_steps=0.0,
                    logging=False):
    """For an outbound route, generate all the cell activity."""
    T = len(headings)

    if logging:
        cx_log = CXLogger(T, 0, cx)

    # Initialise TB and memory
    tb1 = np.zeros(central_complex.N_TB1)
    memory = 0.5 * np.ones(central_complex.N_CPU4)

    for t in range(T):
        tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor, cpu4_inh = update_cells(
            heading=headings[t], velocity=velocity[t], tb1=tb1, memory=memory,
            cx=cx, filtered_steps=filtered_steps, ltm=None)
        if logging:
            cx_log.update_log(t, tl2, cl1, tb1, tn1, tn2,
                              memory, cpu4, cpu1, motor, cpu4_inh)

    if logging:
        return cx_log
    else:
        return tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu4_inh
