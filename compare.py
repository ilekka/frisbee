'''
This code demonstrates how badly the outcome of a given throw is affected by small perturbations
in the initial conditions of the throw. The code compares the results of two sets of throws (made
with different discs and different intended initial conditions) when small, randomly generated
perturbations are applied to the initial conditions. The same perturbations are used for both sets
of throws to try to avoid the possibility that one throw might outperform the other simply by having
a better luck of the draw. The calculation is meant to illustrate that certain types of throws and
certain kinds of discs are less sensitive to small errors made by the thrower.
'''

import numpy as np
import matplotlib.pyplot as plt
from solve_eom import frisbee_solve

# Initial conditions:
# x_0, y_0, z_0     Coordinates of the starting point
# v_0               Magnitude of initial velocity
# theta_0, phi_0    Spherical angles of the initial velocity vector
# omega_0           Initial angular velocity around the disc's symmetry axis
# alpha_0, beta_0   Angles defining the orientation of the disc
# da_0, db_0        Angular velocities of alpha and beta
# w_x, w_y, w_z     Components of wind velocity

# Initial conditions of each target throw

init_1 = np.array([0, 0, 1, 28, 12*np.pi/180, 0*np.pi/180, -34*np.pi, 
    8*np.pi/180, -5*np.pi/180, 0, 0, 0, 0, 0])

init_2 = np.array([0, 0, 1, 25, 12*np.pi/180, 0*np.pi/180, -34*np.pi, 
    8*np.pi/180, 6.5*np.pi/180, 0, 0, 0, 0, 0])

# Aerodynamical parameters of the two discs, in the form
# [CD_0, CL_0, delta_0, kappa, h_1, h_2, delta_1, delta_2]

coeff_1 = np.array([0.05, 0.15, -3.5*np.pi/180, 1.5, 0.1, 8, 1*np.pi/180, 4*np.pi/180])
coeff_2 = np.array([0.05, 0.15, -3.5*np.pi/180, 1.5, 0.1, 0, 4*np.pi/180, 6*np.pi/180])

# Solution, coordinates of the landing point and time of flight for each target throw

sol_1 = frisbee_solve(init_1, coeff_1)
x_1 = sol_1.y_events[0][0][0]
y_1 = sol_1.y_events[0][0][1]
t_1 = sol_1.t_events[0][0]

sol_2 = frisbee_solve(init_2, coeff_2)
x_2 = sol_2.y_events[0][0][0]
y_2 = sol_2.y_events[0][0][1]
t_2 = sol_2.t_events[0][0]

N = 500     # Number of trials

# We generate a set of random numbers between -1 and 1 in the matrix R. These are converted into
# perturbations for the initial conditions by scaling then with the weights stored in the vector w.
# The same perturbations are applied to the initial conditions of both target throws, to avoid the
# possibility that pure luck might be the reason why one throw performed better than the other.

L = len(init_1)
R = 1 - 2*np.random.random_sample((N, L))

w = np.array([0, 0, 0, 1, 3*np.pi/180, 3*np.pi/180, 0, 3.5*np.pi/180, 3.5*np.pi/180, 0, 0, 0, 0, 0])
# w = np.array([0, 0, 0, 1, 2.5*np.pi/180, 2.5*np.pi/180, 0, 4*np.pi/180, 4*np.pi/180, 0, 0, 0, 0, 0])

X_1 = np.zeros(N)
Y_1 = np.zeros(N)
X_2 = np.zeros(N)
Y_2 = np.zeros(N)

# Calculate the outcome of each perturbed throw.
# The landing point coordinates are stored in the vectors X_1, Y_1 and X_2, Y_2.

for i in range(N):

    init = init_1 + w*R[i]
    sol = frisbee_solve(init, coeff_1)
    X_1[i] = sol.y_events[0][0][0]
    Y_1[i] = sol.y_events[0][0][1]

    init = init_2 + w*R[i]
    sol = frisbee_solve(init, coeff_2)
    X_2[i] = sol.y_events[0][0][0]
    Y_2[i] = sol.y_events[0][0][1]

N_t = 1000              # Number of time steps for plotting
T_1 = np.linspace(0, t_1, N_t)
T_2 = np.linspace(0, t_2, N_t)
S_1 = sol_1.sol(T_1)    # Contains the values of each dynamical variable at the times stored in T_1
S_2 = sol_2.sol(T_2)

# Fig 1. Trajectories of the target throws and landing points of the perturbed throws seen from above

plt.figure(1)

plt.plot(S_1[1,:], -S_1[0,:], 'b-', linewidth = 3)
plt.plot(S_2[1,:], -S_2[0,:], 'r-', linewidth = 3)

plt.plot(Y_1, -X_1, 'bo', markersize = 4, alpha = 0.5)
plt.plot(Y_2, -X_2, 'ro', markersize = 4, alpha = 0.5)

plt.plot(y_1, -x_1, 'go', markersize = 8)
plt.plot(y_2, -x_2, 'go', markersize = 8)

plt.xlabel('y')
plt.ylabel('x')

# Fig 2. Trajectories of the target throws seen from the side

plt.figure(2)

plt.plot(S_1[1,:], S_1[2,:], 'b-', linewidth = 3)
plt.plot(S_2[1,:], S_2[2,:], 'r-', linewidth = 3)

plt.xlabel('y')
plt.ylabel('z')

# https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
# This makes 3D plots look nicer
""" patch start """
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
""" patch end """

# Fig 3. 3d view of the target trajectories and the results of the perturbed throws

fig = plt.figure(3)

ax = fig.add_subplot(projection = '3d')

# Set the limits so that the drawing fits within the plot and the trajectories don't look 
# too disproportionate

x_max = max(max(abs(X_1)), max(abs(X_2)))
y_max = max(max(abs(Y_1)), max(abs(Y_2)))
z_max = max(max(S_1[2,:]), max(S_2[2,:]))

y_L = 20*np.ceil(y_max/20)
x_L = max(20*np.ceil(x_max/20), np.ceil(20*np.ceil(y_max/60)))
z_L = max(20, 10*np.ceil(z_max/10))

ax.set_xlim3d(-x_L, x_L)
ax.set_ylim3d(0, y_L)
ax.set_zlim3d(0, z_L)
ax.view_init(elev = 25, azim = -100)
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1) # Remove empty space around the plot

# The hack for removing margins spoils the x-axis, so we draw it again 
ax.plot([-x_L, x_L], [0, 0], [0, 0], color = 'k', linewidth = 0.5)

O = np.zeros(N)

ax.plot(X_1, Y_1, O, 'bo', markersize = 4, alpha = 0.5)
ax.plot(X_2, Y_2, O, 'ro', markersize = 4, alpha = 0.5)

ax.plot(S_1[0,:], S_1[1,:], S_1[2,:], 'b-', linewidth = 4)
ax.plot(S_2[0,:], S_2[1,:], S_2[2,:], 'r-', linewidth = 4)

ax.plot([x_1], [y_1], [0], 'go', markersize = 8)
ax.plot([x_2], [y_2], [0], 'go', markersize = 8)

ax.set_zticks([0, 5, 10, 15, 20])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Fig 4. Distribution of each set of results in terms of distance to the target point

# Distance to target for each throw
D_1 = np.sqrt((X_1 - x_1)**2 + (Y_1 - y_1)**2)
D_2 = np.sqrt((X_2 - x_2)**2 + (Y_2 - y_2)**2)

# Distance to target of the worst throw
d_max = max(max(D_1), max(D_2))

# Histogram bins range from 0 to d_max in steps of 1 meter
bins = np.arange(0, np.ceil(d_max), 1)

f, (ax1, ax2) = plt.subplots(1, 2, sharex = True, sharey = True)

ax1.hist(D_1, bins, color = 'blue')
ax2.hist(D_2, bins, color = 'red')

ax1.set_xlabel('Distance to target')
ax1.set_ylabel('No. of throws')

ax2.set_xlabel('Distance to target')
ax2.set_ylabel('No. of throws')

# Fig 5. Both distributions overlaid on top of each other

plt.figure(5)

plt.hist(D_1, bins, color = 'blue', alpha = 0.5)
plt.hist(D_2, bins, color = 'red', alpha = 0.5)

plt.xlabel('Distance to target')
plt.ylabel('No. of throws')

# Descriptive statistics of the results

b_1 = sum(D_1 < 3)                  # No. of throws within 3 m of the target
c_1 = sum(D_1 - 3 < 7) - b_1        # No. of throws within 10 m but outside 3 m
o_1 = sum(D_1 > 10)                 # No. of throws outside 10 m
a_1 = round(np.average(D_1), 2)     # Average distance to target
p_1 = round(min(D_1), 2)            # Distance to target of the best throw
w_1 = round(max(D_1), 2)            # Distance to target of the worst throw

b_2 = sum(D_2 < 3)
c_2 = sum(D_2 - 3 < 7) - b_2
o_2 = sum(D_2 > 10)
a_2 = round(np.average(D_2), 2)
p_2 = round(min(D_2), 2)
w_2 = round(max(D_2), 2)

print('')

print('Out of', N, 'repetitions of the first throw:')
print(b_1, 'throws were within 3 m of the target.')
print(c_1, 'throws were outside 3 m but within 10 m of the target.')
print(o_1, 'throws were farther than 10 m from the target.')
print('The average distance to target was', a_1, 'm.')
print('The best throw was', p_1, 'm from the target.')
print('The worst throw was', w_1, 'm from the target.')

print('')

print('Out of', N, 'repetitions of the second throw:')
print(b_2, 'throws were within 3 m of the target.')
print(c_2, 'throws were outside 3 m but within 10 m of the target.')
print(o_2, 'throws were farther than 10 m from the target.')
print('The average distance to target was', a_2, 'm.')
print('The best throw was', p_2, 'm from the target.')
print('The worst throw was', w_2, 'm from the target.')

print('')

plt.show()
