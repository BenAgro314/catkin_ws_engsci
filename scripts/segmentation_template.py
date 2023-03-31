import cv2
import numpy as np
import os
import time

files = ['frame_0137.png', 'frame_0173.png']

K = np.array([[1581.5, 0, 1034.7], # needs to be tuned
                            [0, 1588.7, 557.16],
                            [0, 0, 1]])
D = np.array([[-0.37906155, 0.2780121, -0.00092033, 0.00087556, -0.21837157]])

def undistort_image(img, K, D):
    # Undistort the image
    img = cv2.undistort(img, K, D)
    return img

def reproject_2D_to_3D(bbox, actual_height, K):
    # Extract the focal length (fx) and the optical center (cx, cy) from the intrinsic matrix
    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    # Calculate the width of the bounding box in pixels
    bbox_height_pixels = bbox[3] - bbox[1]

    # Calculate the depth (Z) of the object based on the known width and the width in pixels
    depth = (fx * actual_height) / bbox_height_pixels

    # Calculate the 2D coordinates of the center of the bounding box
    center_x_2D = (bbox[0] + bbox[2]) / 2
    center_y_2D = (bbox[1] + bbox[3]) / 2

    # Reproject the 2D center to 3D
    center_x_3D = (center_x_2D - cx) * depth / fx
    center_y_3D = (center_y_2D - cy) * depth / fx

    # Return the 3D coordinates of the center of the bounding box
    return center_x_3D, center_y_3D, depth


for f in files:
    image = cv2.imread(f)
    image = undistort_image(image, K, D)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imshow('Original Image', image)
    cv2.waitKey(0)