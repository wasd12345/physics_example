# Find steady state solution of Laplace's equation for given Dirichlet boundary conditions on a 2D rectangular patch
# Uses iterative method that converges to arbitrary precision
# Python 2.7.8 |Anaconda 2.1.0 (64-bit)
# numpy 1.9.2
# matplotlib 1.4.3




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import time






#==============================================================================
# DEFINING PARAMETERS
#==============================================================================

# Computational Parameters
N_iterations = 10000                      # integer number of iterations
max_tolerance = 1e-8                    # maximum allowable error between iterations (if fully converged, should be 0.)

# Rectangle Boundaries
xmin = 2.3                               # min x coordinate of rectangle boundary
xmax = 9.2                               # max x coordinate of rectangle boundary
ymin = 1.5                               # min y coordinate of rectangle boundary
ymax = 12.1                              # max y coordinate of rectangle boundary

# Dirichlet Boundary Conditions
f1 = lambda y: 4.*np.cos(2.*y)          # Boundary condition for line x=xmin
f2 = lambda y: 1.*y                     # Boundary condition for line x=xmax
f3 = lambda x: 3.*np.sin(2.*x)          # Boundary condition for line y=ymin
f4 = lambda x: .6*x                     # Boundary condition for line y=ymax

# Grid Cell Size
dx = .05                                # x-dimension size of each grid cell
dy = .05                                # y-dimension size of each grid cell

# Random seed:
seed_num = None                         # Or set = some integer 


#Spatial Layout:
#                    f4
#    ymax---XXXXXXXXXXXXXXXXXXXXXXXX
#           XXXXXXXXXXXXXXXXXXXXXXXX
#           XXXXXXXXXXXXXXXXXXXXXXXX
#    f1     XXXXXXXXXXXXXXXXXXXXXXXX  f2
#           XXXXXXXXXXXXXXXXXXXXXXXX
#           XXXXXXXXXXXXXXXXXXXXXXXX
#           XXXXXXXXXXXXXXXXXXXXXXXX
#    ymin---XXXXXXXXXXXXXXXXXXXXXXXX
#           |         f3           |
#           |xmin                  |xmax






#==============================================================================
# MAKE FOLDER FOR SAVING OUTPUT
#==============================================================================

t_start = time.time()
timestamp = time.strftime('%Y_%m_%d__%H_%M_%S',time.localtime(t_start))
os.mkdir(timestamp)
os.chdir(timestamp)











#==============================================================================
# INITIALIZING GRID
#==============================================================================

nx = int((xmax - xmin)/dx) #number of points in x direction
ny = int((ymax - ymin)/dy) #number of points in y direction
print 'nx = {0}, ny = {1}'.format(nx,ny)
print '\n'*2

#Randomly intialize grid values using standard normal distribution
np.random.seed(seed_num)
u = np.random.normal(loc=0.0,scale=1.0,size=(ny,nx))







#==============================================================================
# SPECIFYING BOUNDARY CONDITIONS
#==============================================================================

x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)

# u(x,y) on boundary line x=xmin, for ymin < y < ymax
u[:,0] = f1(y)

# u(x,y) on boundary line x=xmax, for ymin < y < ymax
u[:,-1] = f2(y)

# u(x,y) on boundary line y=ymax, for xmin < x < xmax
u[-1,:] = f4(x)

# u(x,y) on boundary line y=ymin, for xmin < x < xmax
u[0,:] = f3(x)

# Vertically flip array so ymax on top and ymin on bottom:
u = np.flipud(u)








#==============================================================================
# SPECIFYING CORNER VALUES
#==============================================================================

# Since with all 4 boundaries fixed, the 4 corner points are ignored in the iterations
# The corners are defined as the mean value of the 2 pixels they share sides with
u[0,0] = .5*(u[0,1] + u[1,0])
u[-1,0] = .5*(u[-2,0] + u[-1,1])
u[0,-1] = .5*(u[0,-2] + u[1,-1])
u[-1,-1] = .5*(u[-2,-1] + u[-1,-2])









#==============================================================================
# ITERATIVELY SOLVING LAPLACE'S EQUATION
#==============================================================================

#Format of precision_stats_array is [min,mean,max]
precision_stats_array = np.nan*np.ones((N_iterations,3))
diff = np.zeros((u.shape[0]-2,u.shape[1]-2))
for N in xrange(N_iterations):

    #Print status every 100 iterations:
    if N % 100 == 0:
        print 'Iteration {0} of {1}'.format(N,N_iterations)

    #Boundarie values are fixed, so leave them constant, and only adjust interior points
#    xxxxxxxxxxxxxxxxxxxx    
#    x------------------x
#    x------------------x
#    x------------------x
#    x------------------x
#    x------------------x
#    xxxxxxxxxxxxxxxxxxxx   

    #Fast way of calculating interior point updated values:
    #Gives nonsense values for boundaries (since edges rolled over),
    #but that is irrelavnt since edges remain fixed and are not updated.
    temp = .25*(np.roll(u,1,axis=0) + np.roll(u,-1,axis=0) + np.roll(u,1,axis=1) + np.roll(u,-1,axis=1))[1:-1,1:-1]
    diff = u[1:-1,1:-1] - temp
    abs_diff = np.abs(diff)
    u[1:-1,1:-1] = temp
    
    #Finding error statistics
    precision_stats_array[N] = np.array([abs_diff.min(),abs_diff.mean(),abs_diff.max()])
    max_diff = precision_stats_array[N,2]
    
    #Stop iterating anymore if specified precision is reached:
    if max_diff < max_tolerance: 
        N_iterations = N+1 #Update N_iterations just for naming purposes
        break


# Saving error array and grid values as Numpy arrays (and text files for quick manual data validation)
np.save('precision_stats_array_for_{}_iterations.npy'.format(N_iterations), precision_stats_array)
np.save('grid_values_for_{}_iterations.npy'.format(N_iterations), u)
np.savetxt('precision_stats_array_for_{}_iterations.txt'.format(N_iterations), precision_stats_array, delimiter='\t')
np.savetxt('grid_values_for_{}_iterations.txt'.format(N_iterations), u, delimiter='\t')







#==============================================================================
# VISUALIZING SOLUTION HEATMAP / SURFACE
#==============================================================================

allfigsize = (16,9)

# Laplace solution heatmap with linear color scale
plt.figure(figsize=allfigsize)
plt.title("Steady State Solution to Laplace's Equation\nAfter {} Iterations".format(N_iterations),fontsize=30)
plt.imshow(u,interpolation='none',extent=[xmin,xmax,ymin,ymax],aspect='auto')
cb = plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('X',fontsize=30)
plt.ylabel('Y',fontsize=30)
plt.show()
plt.savefig('Laplace_Solution_Heatmap_Iteration{0}_dx{1}_dy{2}.png'.format(N_iterations,dx,dy))

# Laplace solution surface
fig = plt.figure(figsize=allfigsize)
ax = fig.add_subplot(111, projection='3d')
plt.title("Steady State Solution to Laplace's Equation\nAfter {} Iterations".format(N_iterations),fontsize=30)
x,y = np.meshgrid(x,y)
ax.plot_surface(x, y, u,  rstride=4, cstride=4, cmap=cm.coolwarm)
ax.tick_params(labelsize=20)
ax.set_xlabel('X',fontsize=20)
ax.set_ylabel('Y',fontsize=20)
ax.set_zlabel('u(x,y)',fontsize=20)
plt.show()
plt.savefig('Laplace_Solution_Surface_Iteration{0}_dx{1}_dy{2}.png'.format(N_iterations,dx,dy))






#==============================================================================
# VISUALIZING ERROR (DIFFERENCE BETWEEN Nth AND N-1th ITERATION)
#==============================================================================

# Difference heatmap with linear color scale
plt.figure(figsize=allfigsize)
plt.title("Penultimate - Final Iteration",fontsize=30)
plt.imshow(diff,interpolation='none',extent=[xmin,xmax,ymin,ymax],aspect='auto')
cb = plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('X',fontsize=30)
plt.ylabel('Y',fontsize=30)
plt.show()
plt.savefig('Diff_Array_Iteration{0}_dx{1}_dy{2}.png'.format(N_iterations,dx,dy))

# Time series of convergence precision statistics
plt.figure(figsize=allfigsize)
plt.title("Convergence Precision Statistics as Function of Iteration Number",fontsize=30)
markersize=7
plt.semilogy(precision_stats_array[:,2],color='r',label='Max diff',marker='^',markersize=markersize,markeredgecolor='r')
plt.semilogy(precision_stats_array[:,1],color='k',label='Mean diff',marker='o',markersize=markersize,markeredgecolor='k')
plt.semilogy(precision_stats_array[:,0],color='b',label='Min diff',marker='v',markersize=markersize,markeredgecolor='b')
plt.legend(numpoints=1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Iteration',fontsize=30)
plt.ylabel('|difference|',fontsize=30)
plt.show()
plt.savefig('Error_Statistics_Iteration{0}_dx{1}_dy{2}.png'.format(N_iterations,dx,dy))








t_end = time.time()
print 'Approximate run time {}'.format(t_end-t_start)