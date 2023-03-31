import numpy as np

def reproject_2D_to_3D(bbox, actual_width, intrinsic_matrix):
    # Extract the focal length (fx) and the optical center (cx, cy) from the intrinsic matrix
    fx = intrinsic_matrix[0, 0]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Calculate the width of the bounding box in pixels
    bbox_width_pixels = bbox[2] - bbox[0]

    # Calculate the depth (Z) of the object based on the known width and the width in pixels
    depth = (fx * actual_width) / bbox_width_pixels

    # Calculate the 2D coordinates of the center of the bounding box
    center_x_2D = (bbox[0] + bbox[2]) / 2
    center_y_2D = (bbox[1] + bbox[3]) / 2

    # Reproject the 2D center to 3D
    center_x_3D = (center_x_2D - cx) * depth / fx
    center_y_3D = (center_y_2D - cy) * depth / fx

    # Return the 3D coordinates of the center of the bounding box
    return center_x_3D, center_y_3D, depth

# Define the bounding box coordinates (x_min, y_min, x_max, y_max)
bbox = [100, 150, 300, 400]

# Define the actual width of the object in the real world (in meters or any other unit)
actual_width = 1.0

# Define the camera calibration parameters (intrinsic matrix)
intrinsic_matrix = np.array([[800, 0, 320],
                             [0, 800, 240],
                             [0, 0, 1]])

# Reproject the 2D bounding box to 3D
center_x_3D, center_y_3D, depth = reproject_2D_to_3D(bbox, actual_width, intrinsic_matrix)

# Print the 3D coordinates of the center of the bounding box
print(f"3D Coordinates: ({center_x_3D}, {center_y_3D}, {depth})")