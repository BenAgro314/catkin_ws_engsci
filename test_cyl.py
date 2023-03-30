import numpy as np
from sklearn.linear_model import RANSACRegressor

class CylinderEstimator:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def score(self, X, y):
        pass

    def predict(X):
        pass

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
    projected_points = points - np.outer(np.dot(points, known_orientation), known_orientation)


    # Use RANSAC to estimate the center of the cylinder in the projected plane
    model = RANSACRegressor(residual_threshold=residual_threshold, max_trials=max_trials)
    model.fit(projected_points[:, :2], projected_points[:, 2])

    # Get the inliers from the RANSAC model
    inliers = model.inlier_mask_

    # Estimate the center of the cylinder using the inliers
    center_2d = np.mean(projected_points[inliers, :2], axis=0)
    center = np.hstack([center_2d, model.predict([center_2d])[0]])

    return center, inliers

# Example usage
#points = np.random.rand(100, 3)  # Example point cloud data

theta = np.linspace(0, 2*np.pi, 100)
known_radius = 0.5
known_orientation = np.array([0, 0, 1])

points = np.stack((2 + np.cos(theta)*known_radius, 3 + np.sin(theta) * known_radius,np.zeros_like(theta)) + np.random.randn(theta.shape[0]), axis = -1)

center, inliers = fit_cylinder_ransac(points, known_radius, known_orientation)
print(center)
