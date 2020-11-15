'''
This code tries to determine the parameters characterizing the aerodynamical properties
of the frisbee in the model on which the calculation of the frisbee's trajectory is based.
The code attempts to find a set of parameters which reproduces as closely as possible
a given set of input trajectories (corresponding to different initial conditions).

Instead of looking at the whole trajectory, each trajectory is described by a few 'features',
such as the x- and y-coordinates of the landing point, the time of flight, etc. These features
should ideally be things that can be measured to a reasonable accuracy from an actual throw
of the frisbee, using e.g. a video camera and a rangefinder.

The code uses scipy.optimize.minimize to look for the set of parameters that minimizes the
value of a certain 'cost function'. The cost function looks at the features of the trajectories,
and measures how badly the trajectories resulting from a particular choice of the aerodynamical
parameters differ from the corresponding input trajectories.
'''

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from numba import jit   # Speeds up numpy-based functions by a surprisingly large amount

''' A hack to make numba understand np.arrays (source: github.com/numba/numba/issues/4470) '''
from numba import types
from numba.extending import overload

@overload(np.array)
def np_array_ol(x):
    if isinstance(x, types.Array):
        def impl(x):
            return np.copy(x)
        return impl
''' End of hack '''

# Applying jit to the function 'eom' reduces the time required to calculate 
# a typical trajectory from about 0.85 s to about 0.12 s.
# jit also has the option 'fastmath = True', but using it here doesn't seem 
# to make any math happen any faster.

@jit(nopython = True)
def eom(t, f, w_0, wi, coeff):
    
    x, y, z, v1, v2, v3, a, b, da, db = f
    # w_0 = initial angular velocity around the disc's axis of symmetry
    # wi = wind (in the coordinate system fixed to Earth)

    # Parameters and constants
    m = 0.175               # Mass of the disc
    I1 = 0.0007             # Moment of inertia about an axis in the disc's plane
    I3 = 0.0014             # Moment of inertia about the symmetry axis
    Ar = np.pi*0.1055**2    # Area of the disc
    rho = 1.2               # Density of air
    g = 9.81                # Gravitational acceleration
   
    # Transform the components of wind into the rotating coordinate system
    w = np.array([0, 0, 0])
    w[0] = np.cos(b)*wi[0] + np.sin(a)*np.sin(b)*wi[1] - np.cos(a)*np.sin(b)*wi[2]
    w[1] = np.cos(a)*wi[1] + np.sin(a)*wi[2]
    w[2] = np.sin(b)*wi[0] - np.sin(a)*np.cos(b)*wi[1] + np.cos(a)*np.cos(b)*wi[2]    
    
    # Angle of attack
    v = np.array([v1, v2, v3])
    delta = -np.arcsin((v3 - wi[2])/np.linalg.norm(np.array(v) - np.array(w)))
    
    '''
    The aerodynamical properties of the disc are defined by eight parameters.
    
    CD_0, CL_0, delta_0 and kappa (k) determine the drag and lift coefficients as a function 
    of the angle of attack. delta_0 is the angle of attack at which drag is minimized.
    The slope of the curve C_L(delta) is not an independent parameter, but is chosen
    so that C_L = 0 when delta = delta_0.

    h_1, h_2, delta_1 and delta_2 determine the center of pressure as a function of delta.
    The location of the center of pressure depends linearly on delta up to the angle delta_1.
    For delta > delta_1 there is also a term quadratic in delta.
    
    The minimization algorithm will try to choose the values of these parameters so that
    the trajectories resulting from them match the given input trajectories as closely
    as possible.

    Note the scaling between k and h_2, and the corresponding elements of coeff.
    If some element of coeff is very large compared to the others, 'minimize' seems to have
    problems with understanding that it is allowed to change its value.
    '''

    # If optimizing only h_1, h_2, delta_1 and delta_2, set the values of
    # the drag and lift parameters here:

    # CD_0 = 0.05
    # CL_0 = 0.15
    # delta_0 = -3.5*np.pi/180
    # k = 1.5
    
    CD_0 = coeff[4]
    CL_0 = coeff[5]
    delta_0 = coeff[6]
    k = 20*coeff[7]
    
    l = -CL_0/delta_0

    h_1 = coeff[0]
    h_2 = 50*coeff[1]
    delta_1 = coeff[2]
    delta_2 = coeff[3]

    # Step function 
    def theta(x):
        if x > 0:
            return 1
        else:
            return 0

    # Drag and lift coefficients 
    CD = CD_0 + k*(delta - delta_0)**2
    CL = CL_0 + l*delta 

    # Center of pressure
    R = h_1*(delta - delta_1) + theta(delta - delta_2)*h_2*(delta - delta_2)**2

    # Components of force in the frisbee's coordinate system
    F1 = -0.5*rho*Ar*np.linalg.norm(np.array(v) - np.array(wi))*(v[0]-wi[0])*(CD + CL*(v[2]-wi[2])/np.sqrt((v[0]-wi[0])**2 + (v[1]-wi[1])**2)) + m*g*np.cos(a)*np.sin(b)
    F2 = -0.5*rho*Ar*np.linalg.norm(np.array(v) - np.array(wi))*(v[1]-wi[1])*(CD + CL*(v[2]-wi[2])/np.sqrt((v[0]-wi[0])**2 + (v[1]-wi[1])**2)) - m*g*np.sin(a)
    F3 = -0.5*rho*Ar*np.linalg.norm(np.array(v) - np.array(wi))*(CD*(v[2]-wi[2]) - CL*np.sqrt((v[0]-wi[0])**2 + (v[1]-wi[1])**2)) - m*g*np.cos(a)*np.cos(b)

    # Components of torque in the frisbee's coordinate system
    # Includes a damping term added by hand and controlled by the parameter q
    q = 0.0002
    T1 = 0.5*rho*Ar*R*np.linalg.norm(np.array(v) - np.array(wi))*(v[1]-wi[1])*(CL - CD*(v[2]-wi[2])/np.sqrt((v[0]-wi[0])**2 + (v[1]-wi[1])**2)) - 0.5*rho*Ar*np.linalg.norm(np.array(v) - np.array(wi))**2*q*da
    T2 = -0.5*rho*Ar*R*np.linalg.norm(np.array(v) - np.array(wi))*(v[0]-wi[0])*(CL - CD*(v[2]-wi[2])/np.sqrt((v[0]-wi[0])**2 + (v[1]-wi[1])**2)) - 0.5*rho*Ar*np.linalg.norm(np.array(v) - np.array(wi))**2*q*db

    # Equations of motion, in the form [x. y. z. v1. v2. v3. a. b. a.. b..]
    return [np.cos(b)*v1 + np.sin(b)*v3,
            np.sin(a)*np.sin(b)*v1 + np.cos(a)*v2 - np.sin(a)*np.cos(b)*v3,
            -np.cos(a)*np.sin(b)*v1 + np.sin(a)*v2 + np.cos(a)*np.cos(b)*v3,
            F1/m + da*np.sin(b)*v2 - db*v3,
            F2/m + da*np.cos(b)*v3 - da*np.sin(b)*v1,
            F3/m + db*v1 - da*np.cos(b)*v2,
            da,
            db,
            (T1 - I3*w_0*db)/I1/np.cos(b) + 2*da*db*np.tan(b),
            (T2 + I3*w_0*da*np.cos(b))/I1 - da**2*np.sin(b)*np.cos(b)]

# Condition to stop integrating when the frisbee has reached the ground
def landed(t, f, w_0, wi, coeff):
    return f[2]
landed.terminal = True

def frisbee_solve(init, coeff): 

    # Initial conditions
    x_0 = init[0]
    y_0 = init[1]
    z_0 = init[2]
    v_0 = init[3]
    phi = init[4]
    w_0 = init[5]
    a_0 = init[6]
    b_0 = init[7]
    da_0 = init[8]
    db_0 = init[9]

    wind = np.array([0, 0, 0])    # Wind (in the coordinate system fixed to Earth)

    # Components of initial velocity in the Earth coordinate system
    v0_x = 0
    v0_y = v_0*np.cos(phi)
    v0_z = v_0*np.sin(phi)

    # Transform the initial velocity to the frisbee's coordinate system
    v0_1 = np.cos(b_0)*v0_x + np.sin(a_0)*np.sin(b_0)*v0_y - np.cos(a_0)*np.sin(b_0)*v0_z
    v0_2 = np.cos(a_0)*v0_y + np.sin(a_0)*v0_z
    v0_3 = np.sin(b_0)*v0_x - np.sin(a_0)*np.cos(b_0)*v0_y + np.cos(a_0)*np.cos(b_0)*v0_z

    # 'method' can be 'RK23', 'RK45' or 'DOP853'. When numba is used for eom, it seems that
    # 'DOP853' is the fastest, while 'RK45' is the most accurate.
    sol = solve_ivp(eom, [0, 100], [x_0, y_0, z_0, v0_1, v0_2, v0_3, a_0, b_0, da_0, db_0], 
            args = (w_0, wind, coeff), method = 'DOP853', dense_output = True, events = landed) 
    
    # The function returns the 'features' that are used to characterize the trajectory:
    # x- and y-coordinates of the landing point, maximal height, maximal deviation
    # on either side of the x=0 plane, and the time of flight.

    t_F = sol.t_events[0][0]
    x_F = sol.y_events[0][0][0]
    y_F = sol.y_events[0][0][1] 
    
    N = 1000 
    t = np.linspace(0, t_F, N)
    s = sol.sol(t)

    z_max = max(s[2,:])
    x_min = min(s[0,:])
    x_max = max(s[0,:])

    return [x_F, y_F, z_max, x_min, x_max, t_F]

def score(coeff, init, goal):
   
    # score = sum_i w_i(x_i - a_i)^2
    # x_i = achieved value of the i-th feature
    # a_i = desired value of the i-th feature
    # w_i = weights (because the values of some features are much larger than others)

    w = np.array([1, 1, 100, 1, 1, 100])
    
    s = 0
    for i in range(len(goal)):
        x = np.array(frisbee_solve(init[i], coeff))
        a = np.array(goal[i])
        s = s + sum(w*abs(x - a)**2)
    
    return s

# Optimize all the parameters (all_parameters = 1)
# or only the center of pressure parameters (all_parameters = 0).
# In the latter case, set the drag and lift parameters in 'eom'
# (on line 89 at the time of writing this comment).
all_parameters = 1

# Initial conditions:
# x_0, y_0, z_0     Coordinates of the starting point
# v_0               Magnitude of initial velocity. (The vector v_0 lies in the yz-plane.)
# phi_0             Angle of initial velocity relative to the horizontal plane
# omega_0           Initial angular velocity around the disc's symmetry axis
# alpha_0, beta_0   Angles defining the orientation of the disc
# da_0, db_0        Angular velocities of alpha and beta

# 'init' contains the initial conditions for each input trajectory, in the form
# [x_0, y_0, z_0, v_0, phi_0, omega_0, alpha_0, beta_0, da_0, db_0]

init = [[0, 0, 1, 28, 10*np.pi/180, -34*np.pi, 6*np.pi/180, -10*np.pi/180, 0, 0],
        [0, 0, 1, 28, 10*np.pi/180, -34*np.pi, 6*np.pi/180, -5*np.pi/180, 0, 0],
        [0, 0, 1, 28, 10*np.pi/180, -34*np.pi, 6*np.pi/180, 0*np.pi/180, 0, 0],
        [0, 0, 1, 28, 10*np.pi/180, -34*np.pi, 6*np.pi/180, 5*np.pi/180, 0, 0],
        [0, 0, 1, 28, 10*np.pi/180, -34*np.pi, 6*np.pi/180, 10*np.pi/180, 0, 0],
        [0, 0, 1, 28, 15*np.pi/180, -34*np.pi, 10*np.pi/180, -10*np.pi/180, 0, 0],
        [0, 0, 1, 28, 15*np.pi/180, -34*np.pi, 10*np.pi/180, -5*np.pi/180, 0, 0],
        [0, 0, 1, 28, 15*np.pi/180, -34*np.pi, 10*np.pi/180, 0*np.pi/180, 0, 0],
        [0, 0, 1, 28, 15*np.pi/180, -34*np.pi, 10*np.pi/180, 5*np.pi/180, 0, 0],
        [0, 0, 1, 28, 15*np.pi/180, -34*np.pi, 10*np.pi/180, 10*np.pi/180, 0, 0]]

# 'goal' contains the 'features' of each input trajectory: [x_F, y_F, z_max, x_min, x_max, t_F].
# For now we use input trajectories calculated by frisbee_solve (using the parameters coeff_g).
# This ensures that the problem has a well-defined solution which 'minimize' can find, 
# i.e. there exists a set of parameters which will reproduce all the input trajectories.

coeff_g = np.array([0.1, 3/50, 3*np.pi/180, 6*np.pi/180, 0.05, 0.15, -3.5*np.pi/180, 1.5/20])

# If optimizing only the center of pressure parameters, give a coeff_g containing four elements:
# coeff_g = np.array([0.1, 3/50, 2*np.pi/180, 5*np.pi/180])

goal = [frisbee_solve(init[0], coeff_g),
        frisbee_solve(init[1], coeff_g),
        frisbee_solve(init[2], coeff_g),
        frisbee_solve(init[3], coeff_g),
        frisbee_solve(init[4], coeff_g),
        frisbee_solve(init[5], coeff_g),
        frisbee_solve(init[6], coeff_g),
        frisbee_solve(init[7], coeff_g),
        frisbee_solve(init[8], coeff_g),
        frisbee_solve(init[9], coeff_g)]

if all_parameters == 1:

    # Initial guess
    guess = [0.15, 4/50, 4*np.pi/180, 5*np.pi/180, 0.055, 0.143, -4*np.pi/180, 2/20]

    # The parameters are determined by minimizing the value of 'score'.
    # Sometimes 'minimize' tends to get stuck and doesn't know in which direction it should move.
    # Increasing the value of 'eps' from the default 1e-8 helps to avoid this problem.
    # We tell 'minimize' to stop minimizing after 100 iterations, if it hasn't otherwise 
    # stopped by then.

    coeff = minimize(score, guess, (init, goal), method = 'L-BFGS-B', 
            bounds = ((0.01, 1), (0.01, 1), (0, 1), (0, 1),
                (0.01, 1), (0.01, 1), (-1, -np.pi/180), (0, 1)),
            options = {'eps': 1e-6, 'maxiter': 100, 'iprint': 1000})

    print('')
    
    # The best parameters found by 'minimize' 
    print('CL_0    =', coeff.x[4])
    print('CD_0    =', coeff.x[5])
    print('delta_0 =', 180/np.pi*coeff.x[6])
    print('kappa   =', 20*coeff.x[7])
    print('h_1     =', coeff.x[0])
    print('h_2     =', 50*coeff.x[1])
    print('delta_1 =', 180/np.pi*coeff.x[2])
    print('delta_2 =', 180/np.pi*coeff.x[3])

if all_parameters == 0:

    guess = [0.2, 5/50, 3*np.pi/180, 6*np.pi/180]

    coeff = minimize(score, guess, (init, goal), method = 'L-BFGS-B', 
            bounds = ((0.01, 1), (0.01, 1), (0, 1), (0, 1)),
            options = {'eps': 1e-6, 'maxiter': 100, 'iprint': 1000})

    print('')

    print('h_1     =', coeff.x[0])
    print('h_2     =', 50*coeff.x[1])
    print('delta_1 =', 180/np.pi*coeff.x[2])
    print('delta_2 =', 180/np.pi*coeff.x[3])

# Score of the best solution 
s = score(coeff.x, init, goal)
print('')
print('Score   =', s) 

np.set_printoptions(precision = 4, suppress = True)

# Features of the achieved trajectories and the corresponding input trajectories
for i in range(len(goal)):
    print('')
    print(np.array(frisbee_solve(init[i], coeff.x)))
    print(np.array(goal[i]))
