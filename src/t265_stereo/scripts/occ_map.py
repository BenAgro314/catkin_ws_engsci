#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import sensor_msgs.point_cloud2 as pc2
from bresenham import bresenham

# Grid parameters
grid_size = 4.0  # meters
grid_resolution = 0.1  # meters per cell
grid_shape = (int(grid_size / grid_resolution), int(grid_size / grid_resolution))
grid_center = (grid_size / 2.0, grid_size / 2.0)

# Initialize the occupancy grid with -1 (unknown)
occupancy_grid = -np.ones(grid_shape, dtype=np.int8)

def update_occupancy_grid(sensor_origin, points_2d):
    global occupancy_grid
    # Convert points to grid coordinates
    grid_origin = ((sensor_origin + grid_center) / grid_resolution).astype(int)
    grid_coords = ((points_2d + grid_center)/ grid_resolution).astype(int)

    # Update the occupancy grid using raycasting
    for coord in grid_coords:
        x, y = coord
        if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1]:
            # Get cells along the ray from sensor origin to point
            ray_cells = list(bresenham(grid_origin[0], grid_origin[1], x, y))
            # Mark cells along the ray as free (0)
            for cell in ray_cells[:-1]:
                occupancy_grid[cell] = 0
            # Mark the endpoint cell as occupied (100)
            occupancy_grid[x, y] = 100

def pointcloud_callback(msg):
    # Extract 3D points from the point cloud message
    points_3d = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

    # Project 3D points to 2D points (XY plane)
    points_2d = points_3d[:, :2]

    # Sensor origin (assumed to be at the map frame origin)
    sensor_origin = np.array([0.0, 0.0])

    # Update the occupancy grid using raycasting
    update_occupancy_grid(sensor_origin, points_2d)

    # Convert the occupancy grid to an OccupancyGrid message
    occupancy_msg = OccupancyGrid()
    occupancy_msg.header.stamp = rospy.Time.now()
    occupancy_msg.header.frame_id = "map"
    occupancy_msg.info.resolution = grid_resolution
    occupancy_msg.info.width = grid_shape[1]
    occupancy_msg.info.height = grid_shape[0]
    occupancy_msg.info.origin.position.x = -grid_center[0]
    occupancy_msg.info.origin.position.y = -grid_center[1]
    occupancy_msg.info.origin.orientation.w = 1.0
    occupancy_msg.data = occupancy_grid.flatten()

    # Publish the OccupancyGrid message
    pub.publish(occupancy_msg)

if __name__ == "__main__":
    # Create a publisher for the occupancy grid
    rospy.init_node("pointcloud_to_occupancy_grid")
    pub = rospy.Publisher("/occupancy_grid", OccupancyGrid, queue_size=10)
    
    # # Create a tf buffer and listener
    # tf_buffer = tf2_ros.Buffer()
    # tf_listener = tf2_ros.TransformListener(tf_buffer)

    # tf_filter = message_filters.Subscriber('/tf', tf2_msgs.TFMessage)
    # tf_filter.registerCallback(lambda msg: tf_buffer.set_transform(msg.transforms[0], "default_authority"))

    # # Synchronize the tf transformation and point cloud topic
    # sync = message_filters.ApproximateTimeSynchronizer([tf_filter, pointcloud_sub], queue_size=10, slop=0.1)
    # sync.registerCallback(callback)

    # Subscribe to the point cloud topic
    rospy.Subscriber("/camera/points", PointCloud2, pointcloud_callback)

    # Keep the node running
    rospy.spin()