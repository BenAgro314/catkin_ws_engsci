#%%
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from mpl_toolkits.mplot3d import Axes3D



def create_sphere(x_center, y_center, z_center, radius):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + x_center
    y = radius * np.outer(np.sin(u), np.sin(v)) + y_center
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center
    return x, y, z

#%%
#log_folder = "/home/agrobenj/catkin_ws/log_files/ROB498_Drone-2023-03-21.12_06_53/"  # 85%
log_folder = "/home/agrobenj/catkin_ws/log_files/c3_01_novicon/"
#log_folder = "/home/agrobenj/catkin_ws/log_files/ROB498_Drone-2023-03-24.12_25_25/"
#log_folder = "/home/agrobenj/catkin_ws/log_files/ROB498_Drone-2023-03-24.12_39_06/"

log_folder = os.path.normpath(log_folder)
log_folder_parts = log_folder.split(os.sep)
waypoint_path = os.path.join(log_folder, "waypoints.npy")
csv_path = os.path.join(log_folder, log_folder_parts[-1] + ".csv")



#%%

waypoints = np.load(waypoint_path)
print(waypoints)
csv_data = np.genfromtxt(csv_path, delimiter=',')[1:]

#%%

R = 0.35

fig = plt.figure()

pts = csv_data[:, [1, 2, 3], None] # (N, 3, 1)

# Create a 3D axis
ax = fig.add_subplot(111, projection='3d')

for pt in waypoints:
    x, y, z = pt
    color = 'r'
    if min(np.linalg.norm(pts[:, :, 0] - pt[None, :], axis = 1)) < R:
        color = 'g'
    #x_center, y_center, z_center = waypoints[i], y[i], z[i]
    xs, ys, zs = create_sphere(x, y, z, R)
    ax.plot_surface(xs, ys, zs, color=color, alpha=0.5)



# Create a 3D scatter plot
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
plt.show()
