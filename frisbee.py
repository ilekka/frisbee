''' 
The code calculates and plots the trajectory of a disc golf frisbee for given initial conditions.

The aerodynamical properties of the disc are determined by the drag and lift coefficients, and by
the dependence of the center of pressure on the angle of attack.

For the drag and lift coefficients, we imitate the values found by wind tunnel measurements in
https://www.escholar.manchester.ac.uk/api/datastream?publicationPid=uk-ac-man-scw:132975&datastreamId=FULL-TEXT.PDF

The function which determines how the center of pressure varies with the angle of attack is simply
guessed in such a way that the results of the simulation have a reasonably close resemblance to the
flight of an actual golf disc.
'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from subprocess import call
import time

# Give the initial conditions:
x_0 = 0         # Coordinates of the starting point
y_0 = 0
z_0 = 1 
v_0 = 28        # Magnitude of initial velocity. (The initial velocity vector lies in the yz-plane.)
phi = 10*np.pi/180  # Angle of initial velocity relative to the horizontal plane
w_0 = -34*np.pi     # Initial angular velocity around the disc's symmetry axis
a_0 = 6*np.pi/180   # The disc is rotated by the angle alpha around the x-axis...
b_0 = 6*np.pi/180   # ...and then by the angle beta around the y-axis attached to the disc.
da_0 = 0*np.pi      # Off-axis components of angular velocity
db_0 = 0*np.pi
wind = [0, 0, 0]    # Wind in the coordinate system fixed to Earth
disc = 3            # 1-5 (1 = most overstable, 5 = most understable)

# What to plot? (1 = yes, any other number = no)
plot2d = 1      # 2D plots of the trajectory and the angles of the disc
plot3d = 1      # 3D animation of the trajectory (created using FuncAnimation)
plot3d_disc = 1 # 3D animation of the trajectory and the disc (created from png images). 
                # Quite slow, which seems like a fair punishment for not understanding how 
                # to use FuncAnimation properly.

filename = 'anim1.mp4'      # Filename to save the first animation
filename_2 = 'anim2.mp4'    # Filename to save the second animation

# Directory in which frisbee.py lives. Animations will be saved in this directory.
dir = '/home/ilkka/Python/frisbee/github/' 

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

# Using this instead of np.linalg.norm speeds up the code by at least several hundredths of a second
def norm3(v):
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def eom(t, f , w_0, wi, disc):
    x, y, z, v1, v2, v3, a, b, da, db = f
    # w_0 = initial angular velocity around the disc's axis of symmetry
    # wi = wind (in the coordinate system fixed to Earth)
    # disc = 1, 2, 3, 4 or 5, with 3 being maybe the most realistic approximation of a reasonable distance driver

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
    delta = -np.arcsin((v[2] - w[2])/norm3(v - w))
    
    # Aerodynamical coefficients
    CD_0 = 0.05
    CL_0 = 0.15
    delta_0 = -3.5*np.pi/180    # Angle of attack at which drag is minimized
    k = 1.5
    l = -CL_0/delta_0           # Chosen so that C_L = 0 when delta = delta_0

    CD = CD_0 + k*(delta - delta_0)**2
    CL = CL_0 + l*delta 

    # Here we calculate the center of pressure as a function of the angle of attack.
    # Depends linearly on delta up to some angle delta_2, after which it also picks up a quadratic term.
    # This function can be very different for different kinds of discs.
    # In principle different discs should probably also have different C_D and C_L.
    
    if disc == 1: # Very overstable
        h_1 = 0.1
        h_2 = 10
        delta_1 = 1*np.pi/180
        delta_2 = 5*np.pi/180
    
    if disc == 2: # Still quite overstable
        h_1 = 0.1
        h_2 = 10
        delta_1 = 2*np.pi/180
        delta_2 = 5*np.pi/180
    
    if disc == 4: # Straight/slightly understable (?)
        h_1 = 0.1
        h_2 = 3
        delta_1 = 4*np.pi/180
        delta_2 = 6*np.pi/180
    
    if disc == 5: # Severely flippy
        h_1 = 0.1
        h_2 = 0
        delta_1 = 4*np.pi/180
        delta_2 = 6*np.pi/180
    
    if disc == 3: # I think this behaves the most like a reasonable distance driver
        h_1 = 0.1
        h_2 = 3
        delta_1 = 2*np.pi/180
        delta_2 = 5*np.pi/180
    
    # Step function:
    def theta(x):
        if x > 0:
            return 1
        else:
            return 0

    R = h_1*(delta - delta_1) + theta(delta - delta_2)*h_2*(delta - delta_2)**2

    # Components of force in the frisbee's coordinate system
    F1 = -0.5*rho*Ar*norm3(v - w)*(v[0] - w[0])*(CD + CL*(v[2] - w[2])
            /np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) + m*g*np.cos(a)*np.sin(b)
    F2 = -0.5*rho*Ar*norm3(v - w)*(v[1] - w[1])*(CD + CL*(v[2] - w[2])
            /np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) - m*g*np.sin(a)
    F3 = -0.5*rho*Ar*norm3(v - w)*(CD*(v[2] - w[2]) 
            - CL*np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) - m*g*np.cos(a)*np.cos(b)

    # Components of torque in the frisbee's coordinate system
    # Includes a damping term added by hand and controlled by the parameter q
    q = 0.0002
    T1 = 0.5*rho*Ar*R*norm3(v - w)*(v[1] - w[1])*(CL - CD*(v[2] - w[2])
            /np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) - 0.5*rho*Ar*norm3(v - w)**2*q*da
    T2 = -0.5*rho*Ar*R*norm3(v - w)*(v[0] - w[0])*(CL - CD*(v[2] - w[2])
            /np.sqrt((v[0] - w[0])**2 + (v[1] - w[1])**2)) - 0.5*rho*Ar*norm3(v - w)**2*q*db

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
def landed(t, f, w_0, wi, disc):
    return f[2]
landed.terminal = True

def frisbee_solve(x_0, y_0, z_0, v0_x, v0_y, v0_z, w_0, a_0, b_0, da_0, db_0, wind, disc):

    # Transform the initial velocity to the frisbee's coordinate system
    v0_1 = np.cos(b_0)*v0_x + np.sin(a_0)*np.sin(b_0)*v0_y - np.cos(a_0)*np.sin(b_0)*v0_z
    v0_2 = np.cos(a_0)*v0_y + np.sin(a_0)*v0_z
    v0_3 = np.sin(b_0)*v0_x - np.sin(a_0)*np.cos(b_0)*v0_y + np.cos(a_0)*np.cos(b_0)*v0_z

    # "method" can also be 'RK45' or 'DOP853'. As far as accuracy goes, it apparently makes 
    # no difference which one is chosen, but 'RK23' seems the fastest by a nose.
    sol = solve_ivp(eom, [0, 100], [x_0, y_0, z_0, v0_1, v0_2, v0_3, a_0, b_0, da_0, db_0], args = (w_0, wind, disc), method = 'RK23', dense_output = True, events = landed) # , max_step = 10**-3)

    # Time of flight, x- and y-coordinates of the landing point, and length of the throw
    t_F = sol.t_events[0][0]
    x_F = sol.y_events[0][0][0]
    y_F = sol.y_events[0][0][1] 
    d_F = np.sqrt(x_F**2 + y_F**2)

    return sol, t_F, x_F, y_F, d_F

# Components of initial velocity in the Earth coordinate system
v0_x = 0
v0_y = v_0*np.cos(phi)
v0_z = v_0*np.sin(phi)

# Solve the equations of motion
sol, t_F, x_F, y_F, d_F = frisbee_solve(x_0, y_0, z_0, v0_x, v0_y, v0_z, w_0, a_0, b_0, da_0, db_0, wind, disc)

# Print the result of the throw
print('t_F = ', t_F)
print('x_F = ', x_F)
print('y_F = ', y_F)
print('dis = ', d_F)

n_fig = 1 # This is used to set the figure number of the animation in the second step

if plot2d == 1:

    # If 5 figures are created in this step, the next one will be number 6, otherwise number 1
    n_fig = 6 

    N = 1000
    t = np.linspace(0, t_F, N)
    s = sol.sol(t)

    plt.figure(1)
    plt.plot(s[0,:], s[1,:])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Top view')

    plt.figure(2)
    plt.plot(s[1,:], s[2,:])
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Side view')

    plt.figure(3)
    plt.plot(s[0,:], s[2,:])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('View from behind')

    plt.figure(4)
    plt.plot(t, 180/np.pi*s[6,:], label='alpha')
    plt.plot(t, 180/np.pi*s[7,:], label='beta')
    plt.xlabel('t')
    plt.legend(loc='best')
    plt.title('Angles')

    # Extract the angle of attack from the solution
    d = np.zeros(N)
    for k in range(N):
        v = np.array([s[3,k], s[4,k], s[5,k]])
        a = s[6,k]
        b = s[7,k]
        w = np.zeros(3)
        w[0] = np.cos(b)*wind[0] + np.sin(a)*np.sin(b)*wind[1] - np.cos(a)*np.sin(b)*wind[2]
        w[1] = np.cos(a)*wind[1] + np.sin(a)*wind[2]
        w[2] = np.sin(b)*wind[0] - np.sin(a)*np.cos(b)*wind[1] + np.cos(a)*np.cos(b)*wind[2]    
        d[k] = -np.arcsin((v[2] - w[2])/norm3(v - w))

    plt.figure(5)
    plt.plot(t, 180/np.pi*d)
    plt.xlabel('t')
    plt.ylabel('delta')
    plt.title('Angle of attack')
    
    # Show the plots at this point only if the following steps are not executed
    if plot3d != 1 and plot3d_disc != 1: 
        plt.show()

if plot3d == 1:

    N = int(np.floor(30*t_F)) # For a 30 fps animation, there must be 30 time steps per second
    t = np.linspace(0, t_F, N)
    s = sol.sol(t)

    fig = plt.figure(n_fig, figsize=(12,8))
    ax = fig.add_subplot(projection='3d')

    # Set the limits so that the trajectory stays within the plot and doesn't look too disproportionate
    y_L = 20*np.ceil(y_F/20)
    x_L = max(20*np.ceil(x_F/20), np.ceil(20*np.ceil(y_L/60)))
    z_L = max(20, 10*np.ceil(max(s[2,:])/10))

    # Create a "data object" which contains the curves to be plotted
    data = [np.zeros((3,N)) for i in range(3)]
    for i in range(N):
        # The complete trajectory:
        data[0][0,i] = s[0,i]
        data[0][1,i] = s[1,i]
        data[0][2,i] = s[2,i]
        # Projection onto the xy-plane:
        data[1][0,i] = s[0,i]
        data[1][1,i] = s[1,i]
        data[1][2,i] = 0
        # Projection onto the yz-plane:
        data[2][0,i] = x_L
        data[2][1,i] = s[1,i]
        data[2][2,i] = s[2,i]

    def update(n, data, draw): 
        for dr, da in zip(draw, data):
            dr.set_data(da[0:2, :n])
            dr.set_3d_properties(da[2, :n])
        return draw

    ltype = ['b-', 'b:', 'b:']
    lwidth = [3, 1, 1]

    draw = [ax.plot(data[i][0, 0:1], data[i][1, 0:1], data[i][2, 0:1], ltype[i], linewidth=lwidth[i])[0] for i in range(3)]

    ax.set_xlim3d(-x_L,x_L)
    ax.set_ylim3d(0,y_L)
    ax.set_zlim3d(0,z_L)
    ax.view_init(elev=30, azim=-125)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Remove empty space around the plot

    # The hack for removing margins spoils the x-axis, so we draw it again 
    ax.plot([-x_L,x_L], [0, 0], [0, 0], color='k', linewidth=0.5)
    
    ani = FuncAnimation(fig, update, N, fargs=(data, draw), interval=33.33, repeat_delay=1000)

    ani.save(filename)

    if plot3d_disc != 1: # Show the plots now only if the next step doesn't run
        plt.show()

if plot3d_disc == 1:

    N = int(np.floor(30*t_F))
    t = np.linspace(0, t_F, N)
    s = sol.sol(t)

    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    # Sets the axes of the 3D plot so that the frisbee looks like a circle and not like an ellipse
    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d ([z_middle - plot_radius, z_middle + plot_radius])

    # This plots the trajectory up to the i-th step and draws a picture of the frisbee and its velocity at that moment
    def draw_disc(i, sol, limits, alpha, color):

        fig = plt.figure(7, figsize=(19.2,7.2)) # figsize is in units of 100px
        plt.clf
        ax1 = fig.add_subplot(1, 2, 1, projection='3d') # Trajectory plotted here
        ax2 = fig.add_subplot(1, 2, 2, projection='3d') # Frisbee drawn here

        x_min = limits[0]
        x_max = limits[1]
        y_min = limits[2]
        y_max = limits[3]
        z_min = limits[4]
        z_max = limits[5]

        # Restore the x-axis, which has been spoiled by removing margins
        ax1.plot([x_min,x_max], [y_min, y_min], [z_min, z_min], color='k', linewidth=0.5)
        ax2.plot([-2, 2], [-1, -1], [-2, -2], color='k', linewidth=0.5)

        ax1.set_xlim3d(x_min,x_max)
        ax1.set_ylim3d(y_min, y_max)
        ax1.set_zlim3d(z_min, z_max)
        ax1.view_init(elev=30, azim=-125)

        O = np.zeros(N)
        X = x_max*np.ones(N)
        ax1.plot(sol[0,:][:i], sol[1,:][:i], O[:i], 'b:', linewidth=1) # Projection onto xy-plane
        ax1.plot(X[:i], sol[1,:][:i], sol[2,:][:i], 'b:', linewidth=1) # Projection onto yz-plane
        ax1.plot(sol[0,:][:i], sol[1,:][:i], sol[2,:][:i], 'b-', linewidth=3) # Trajectory

        ax2.set_xlim3d(-2,2)
        ax2.set_ylim3d(-1,3)
        ax2.set_zlim3d(-1.5,1.5)
        ax2.view_init(elev=30, azim=-125)
       
        # We don't want to have tick values in the plot which shows the frisbee
        ax2.xaxis.set_major_formatter(plt.NullFormatter()) 
        ax2.yaxis.set_major_formatter(plt.NullFormatter()) 
        ax2.zaxis.set_major_formatter(plt.NullFormatter())

        set_axes_equal(ax2)
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Remove empty space around the plot  

        # Calculate the components of velocity in the Earth coordinate system, given the components in the rotating coordinate system and the angles of the frisbee
        v1 = sol[3,i]
        v2 = sol[4,i]
        v3 = sol[5,i]
        a = sol[6,i]
        b = sol[7,i]
        
        vx = np.cos(b)*v1 + np.sin(b)*v3
        vy = np.sin(a)*np.sin(b)*v1 + np.cos(a)*v2 - np.sin(a)*np.cos(b)*v3
        vz = -np.cos(a)*np.sin(b)*v1 + np.sin(a)*v2 + np.cos(a)*np.cos(b)*v3
        
        # The frisbee is drawn as a 360-sided polygon rotated by the angles alpha and beta
        th = np.linspace(0, 359/180*np.pi, 360)
        X = np.cos(th)
        Y = np.sin(th)
        Z = np.zeros(360)
        
        rX = np.zeros(360)
        rY = np.zeros(360)
        rZ = np.zeros(360)
        for j in range(360):
            rX[j] = np.cos(b)*X[j] + np.sin(b)*Z[j]
            rY[j] = np.sin(a)*np.sin(b)*X[j] + np.cos(a)*Y[j] - np.sin(a)*np.cos(b)*Z[j]
            rZ[j] = -np.cos(a)*np.sin(b)*X[j] + np.sin(a)*Y[j] + np.cos(a)*np.cos(b)*Z[j]
        
        vtx = [list(zip(rX, rY, rZ))]
        ax2.add_collection3d(Poly3DCollection(vtx, alpha=alpha, facecolors=color))
        
        # "Projection" to show the angle alpha
        y1 = np.cos(a)
        y2 = -np.cos(a)
        z1 = np.sin(a)
        z2 = -np.sin(a)
        ax2.plot([2,2], [y1, y2], [z1, z2], color=color, linewidth=2)
        
        # "Projection" to show the angle beta
        x1 = np.cos(b)
        x2 = -np.cos(b)
        z1 = -np.sin(b)
        z2 = np.sin(b)
        ax2.plot([x1, x2], [3,3], [z1, z2], color=color, linewidth=2)

        k = 0.1 # Scale the velocity vector so it fits within the plot
        
        # Draw the velocity vector and its projections
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        ax2.quiver(0, 0, 0, vx/v, vy/v, vz/v, length=k*v, arrow_length_ratio=0.1, color='r')
        
        v = np.sqrt(vy**2 + vz**2)
        ax2.quiver(2, 0, 0, 0, vy/v, vz/v, length=k*v, arrow_length_ratio=0.1, color='r')

        v = np.sqrt(vx**2 + vz**2)
        ax2.quiver(0, 3, 0, vx/v, 0, vz/v, length=k*v, arrow_length_ratio=0.1, color='r')

        v = np.sqrt(vx**2 + vy**2)
        ax2.quiver(0, 0, -2, vx/v, vy/v, 0, length=k*v, arrow_length_ratio=0.1, color='r')

    # Create a directory labeled by current time, in which the png files are saved.
    # This way we can make many animations without having to clear the directory by hand.
    # Remember to delete these directories every now and then.
    t = time.localtime()
    current_time = time.strftime('%H%M%S', t)
    dir_1 = dir + 'png_' + current_time
    call(['mkdir', dir_1]) 

    # Set the limits of the trajectory plot in a "clever" way
    y_L = 20*np.ceil(y_F/20)
    x_L = max(20*np.ceil(x_F/20), np.ceil(20*np.ceil(y_L/60)))
    z_L = max(25, 10*np.ceil(max(s[2,:])/10))
    limits = [-x_L, x_L, 0, y_L, 0, z_L]
    
    # Draw each step of the animation and save it as a png file
    for i in range(N):
        draw_disc(i, s, limits, 0.7, 'b')
        filename = dir_1 + '/disc_{:03}.png'.format(i)
        plt.savefig(filename)
        plt.close()

    # Compile the pngs into an animation using ffmpeg
    png_names = dir_1 + '/disc_%03d.png'
    call(['ffmpeg', '-framerate', '30', '-i', png_names, filename_2])

    plt.show() # Show the plots created in the first two stages
