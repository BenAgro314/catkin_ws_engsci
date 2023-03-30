#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import sensor_msgs.point_cloud2 as pc2
from bresenham import bresenham
import tf2_ros
import tf.transformations
from sensor_msgs.msg import PointCloud2, PointField
from sklearn.linear_model import RANSACRegressor
import numpy as np
from scipy.optimize import minimize
from visualization_msgs.msg import Marker

R = 0.15

marker_pub = rospy.Publisher('/cylinder_marker', Marker, queue_size=10)

pub_xyz = rospy.Publisher('/map_points', PointCloud2, queue_size=10)

def numpy_to_PointCloud2(array, frame_id):
    msg = PointCloud2()

    msg.header.stamp = rospy.Time().now()

    msg.header.frame_id = frame_id

    if len(array.shape) == 3:
        msg.height = array.shape[1]
        msg.width = array.shape[0]
    else:
        msg.height = 1
        msg.width = len(array)

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * array.shape[0]
    # msg.is_dense = int(np.isfinite(array).all())
    msg.is_dense = False
    msg.data = np.asarray(array, np.float32).tostring()
    return msg

def rpy_to_quaternion(roll, pitch, yaw):
    # Convert RPY angles to quaternion
    quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    return quaternion

def publish_cylinder_marker(w_fit, C_fit, r_fit):
    marker = Marker()
    marker.header.frame_id = 'map'  # Frame in which the marker is defined
    marker.header.stamp = rospy.Time.now()  # Timestamp
    marker.ns = 'cylinder'  # Namespace for the marker
    marker.id = 0  # Unique ID for the marker
    marker.type = Marker.CYLINDER  # Marker type: cylinder
    marker.action = Marker.ADD  # Action: add/modify the marker

    orientation = w_fit / np.linalg.norm(w_fit)

    # Compute the quaternion from the orientation vector
    quaternion = tf.transformations.quaternion_about_axis(0, np.array([0, 0, 1]))

    # Set the pose of the cylinder (position and orientation)
    marker.pose.position.x = C_fit[0]
    marker.pose.position.y = C_fit[1]
    marker.pose.position.z = C_fit[2]
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]

    # Set the scale of the cylinder (radius and height)
    marker.scale.x = 2*r_fit  # Diameter in the x direction (radius)
    marker.scale.y = 2*r_fit  # Diameter in the y direction (radius)
    marker.scale.z = 2.2  # Height in the z direction

    # Set the color and transparency (alpha) of the cylinder
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.8

    # Set the lifetime of the marker (0 means infinite)
    marker.lifetime = rospy.Duration(0)

    # Publish the marker
    marker_pub.publish(marker)


# Define the objective function to minimize
def circle_objective_function(center, X, Y, known_radius):
    x0, y0 = center
    return np.sum(((X - x0) ** 2 + (Y - y0) ** 2 - known_radius**2) ** 2)

def residuals(center, points, known_radius):
    return np.abs(np.linalg.norm(points - center, axis = -1) - known_radius)

def fit_circle(X, Y, known_radius):
    """
    Fit a circle with known radius to X and Y data by optimizing for the (x, y) parameters of the circle's center.

    Parameters:
        X (numpy.ndarray): X data.
        Y (numpy.ndarray): Y data.
        known_radius (float): Known radius of the circle.

    Returns:
        numpy.ndarray: Estimated (x, y) coordinates of the circle's center.
    """


    # Initial guess for the circle's center
    initial_guess = (np.mean(X), np.mean(Y))

    # Perform the optimization
    result = minimize(circle_objective_function, initial_guess, args=(X, Y, known_radius), method='L-BFGS-B')

    # Extract the estimated (x, y) coordinates of the circle's center
    estimated_center = result.x

    return estimated_center


def fit_cylinder_ransac(points, known_radius, known_orientation, max_trials=100, residual_threshold=0.1):
    """
    Fit a cylinder with known orientation and radius to point cloud data using RANSAC.

    Parameters:
        points (numpy.ndarray): Point cloud data (N, 3) where N is the number of points.
        known_radius (float): Known radius of the cylinder.
        known_orientation (numpy.ndarray): Known orientation of the cylinder (3,).
        max_trials (int): Maximum number of RANSAC iterations.
        residual_threshold (float): Maximum distance for a data point to be considered as an inlier.

    Returns:
        center (numpy.ndarray): Estimated center of the cylinder (3,).
        inliers (numpy.ndarray): Boolean array indicating which points are inliers.
    """
    # Normalize the orientation vector
    known_orientation = known_orientation / np.linalg.norm(known_orientation)

    # Project points onto the plane perpendicular to the known orientation
    projected_points = (points - np.outer(np.dot(points, known_orientation), known_orientation))[:, :2]

    best_inlier_count = 0
    best_center=None
    best_inliers=None

    for _ in range(max_trials):
        # select a random subset of 3 points
        idx = np.random.randint(0, len(projected_points), size = (10,))
        pts = projected_points[idx]
        center = fit_circle(pts[:, 0], pts[:, 1], known_radius)

        res = residuals(np.array(center), projected_points, known_radius)
        inliers = res < residual_threshold

        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count:
            best_center = center
            best_inliers = inliers
            best_inlier_count = inlier_count
    print(best_inlier_count)
    return best_center, best_inliers

def transform_to_matrix(transform):
    # Extract translation and rotation from the Transform message
    translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
    rotation_quaternion = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])

    # Convert quaternion to rotation matrix
    rotation_matrix = tf.transformations.quaternion_matrix(rotation_quaternion)

    # Set the translation part of the transformation matrix
    rotation_matrix[:3, 3] = translation

    return rotation_matrix

rospy.init_node("pointcloud_to_occupancy_grid")

tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)

def pointcloud_callback(msg):
    # Extract 3D points from the point cloud message
    try:
        t_map_optical = tf_buffer.lookup_transform(
            'map', 'camera_optical_frame', rospy.Time(0)).transform
    except Exception as e:
        return
    #t_map_optical = TransformStamped
    t_map_optical = transform_to_matrix(t_map_optical)
    pts_optical = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
    pts_optical = np.concatenate((pts_optical, np.ones_like(pts_optical[:, -1:])), axis = -1)[:, :, None] # (N, 4, 1)
    pts_map = t_map_optical @ pts_optical
    pts_map = pts_map[:, :-1, 0] # (N, 3)

    #center, inliers = fit_cylinder_ransac(pts_map, R, np.array([0, 0, 1]), max_trials=100, residual_threshold=0.05)

    #inlier_pts = pts_map[inliers]
    #if center is None:
    #    return

    #obj = circle_objective_function(center, inlier_pts[:, 0], inlier_pts[:, 1], known_radius=R)
    #print(obj)

    pub_xyz.publish(numpy_to_PointCloud2(pts_map, frame_id="map"))

    #if sum(inliers) < 2000 or obj > 0.2:
    #    return 

    ##marker_pub.publish()
    #publish_cylinder_marker(np.array([0, 0, 1]), [center[0], center[1], 1.1], R)
    #print(center)


if __name__ == "__main__":
    # Create a publisher for the occupancy grid
    
    rospy.Subscriber("/camera/points", PointCloud2, pointcloud_callback)

    # Keep the node running
    rospy.spin()