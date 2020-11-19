import numpy as np
from scipy.integrate import solve_ivp
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

'''
Applying jit to the function 'eom' reduces the time required to calculate a typical trajectory from 
about 0.85 s to about 0.12 s.

jit also has the option 'fastmath = True', but using it here doesn't seem to make any math happen
any faster.
'''

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
    w = np.zeros(3)
    w[0] = np.cos(b)*wi[0] + np.sin(a)*np.sin(b)*wi[1] - np.cos(a)*np.sin(b)*wi[2]
    w[1] = np.cos(a)*wi[1] + np.sin(a)*wi[2]
    w[2] = np.sin(b)*wi[0] - np.sin(a)*np.cos(b)*wi[1] + np.cos(a)*np.cos(b)*wi[2]    
    
    # Angle of attack
    v = np.array([v1, v2, v3])
    delta = -np.arcsin((v[2] - w[2])/np.linalg.norm(v - w))
    
    # Aerodynamical parameters

    CD_0 = coeff[0]         # Drag coefficient at delta = delta_0
    CL_0 = coeff[1]         # Lift coefficient at zero angle of attack
    delta_0 = coeff[2]      # Angle of attack at which drag is minimized
    k = coeff[3]            # Parametrizes the dependence of C_D on delta
    l = -CL_0/delta_0       # Slope of the curve C_L(delta)
                            # Chosen so that C_L = 0 when delta = delta_0
    
    # These parameters determine how the center of pressure depends on the angle of attack
    h_1 = coeff[4]
    h_2 = coeff[5]
    delta_1 = coeff[6]
    delta_2 = coeff[7]

    # Step function 
    def theta(x):
        if x > 0:
            return 1
        else:
            return 0

    # Drag and lift coefficients 
    CD = CD_0 + k*(delta - delta_0)**2
    CL = CL_0 + l*delta 

    # Center of pressure. Depends linearly on delta up to delta = delta_2, after which picks up
    # a quadratic term as well.
    R = h_1*(delta - delta_1) + theta(delta - delta_2)*h_2*(delta - delta_2)**2

    # Components of force in the frisbee's coordinate system
    F1 = -0.5*rho*Ar*np.linalg.norm(v - w)*(v[0] - w[0])*(CD + CL*(v[2] - w[2])
            /np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) + m*g*np.cos(a)*np.sin(b)
    F2 = -0.5*rho*Ar*np.linalg.norm(v - w)*(v[1] - w[1])*(CD + CL*(v[2] - w[2])
            /np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) - m*g*np.sin(a)
    F3 = -0.5*rho*Ar*np.linalg.norm(v - w)*(CD*(v[2] - w[2]) 
            - CL*np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) - m*g*np.cos(a)*np.cos(b)

    # Components of torque in the frisbee's coordinate system
    # Includes a damping term added by hand and controlled by the parameter q
    q = 0.0002

    T1 = 0.5*rho*Ar*R*np.linalg.norm(v - w)*(v[1] - w[1])*(CL - CD*(v[2] - w[2])
            /np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) - 0.5*rho*Ar*np.linalg.norm(v - w)**2*q*da
    T2 = -0.5*rho*Ar*R*np.linalg.norm(v - w)*(v[0] - w[0])*(CL - CD*(v[2] - w[2])
            /np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) - 0.5*rho*Ar*np.linalg.norm(v - w)**2*q*db

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

    # Initial conditions:
    # x_0, y_0, z_0     Coordinates of the starting point
    # v_0               Magnitude of initial velocity
    # theta_0, phi_0    Spherical angles of the initial velocity vector
    # omega_0           Initial angular velocity around the disc's symmetry axis
    # alpha_0, beta_0   Angles defining the orientation of the disc
    # da_0, db_0        Angular velocities of alpha and beta
    # w_x, w_y, w_z     Components of wind velocity (in the Earth coordinate system)
    
    x_0 = init[0]
    y_0 = init[1]
    z_0 = init[2]
    v_0 = init[3]
    th_0 = init[4]
    ph_0 = init[5]
    w_0 = init[6]
    a_0 = init[7]
    b_0 = init[8]
    da_0 = init[9]
    db_0 = init[10]
    
    wind = np.array([init[11], init[12], init[13]])

    # Components of initial velocity in the Earth coordinate system
    v0_x = v_0*np.cos(th_0)*np.sin(ph_0)
    v0_y = v_0*np.cos(th_0)*np.cos(ph_0)
    v0_z = v_0*np.sin(th_0)

    # Transform the initial velocity to the frisbee's coordinate system
    v0_1 = np.cos(b_0)*v0_x + np.sin(a_0)*np.sin(b_0)*v0_y - np.cos(a_0)*np.sin(b_0)*v0_z
    v0_2 = np.cos(a_0)*v0_y + np.sin(a_0)*v0_z
    v0_3 = np.sin(b_0)*v0_x - np.sin(a_0)*np.cos(b_0)*v0_y + np.cos(a_0)*np.cos(b_0)*v0_z

    # 'method' can be 'RK23', 'RK45' or 'DOP853'. When numba is used for eom, it seems that
    # 'DOP853' is the fastest, while 'RK45' is the most accurate.
    sol = solve_ivp(eom, [0, 100], [x_0, y_0, z_0, v0_1, v0_2, v0_3, a_0, b_0, da_0, db_0], 
            args = (w_0, wind, coeff), method = 'DOP853', dense_output = True, events = landed) 
    
    return sol
