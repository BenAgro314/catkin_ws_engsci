import cv2
import numpy as np
import os
import time
import glob

#files = ['frame_0137.png', 'frame_0173.png', 'frame_0070.png', 'frame_0304.png']
files = glob.glob('/home/agrobenj/catkin_ws/images/flying_v2/*.png')

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


def get_mask_from_range(hsv_img, low, high):
    mask = cv2.inRange(hsv_image, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

for f in files:
    image = cv2.imread(f)
    image = undistort_image(image, K, D)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[:, image.shape[1]//2:]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color yellow in the HSV color space
    lower_yellow = (10, 0, 0)
    upper_yellow = (40, 255, 255)
    
    lower_red = (0, 0, 0)
    upper_red = (10, 255, 255)

    lower_green = (40, 0, 0)
    upper_green = (80, 255, 255)

    # Create a binary mask using the defined yellow range
    yellow_mask = get_mask_from_range(hsv_image, lower_yellow, upper_yellow)
    red_mask = get_mask_from_range(hsv_image, lower_red, upper_red)
    green_mask = get_mask_from_range(hsv_image, lower_green, upper_green)
    

    yellow_segment = cv2.bitwise_and(image, image, mask=yellow_mask)
    red_segment = cv2.bitwise_and(image, image, mask=red_mask)
    green_segment = cv2.bitwise_and(image, image, mask=green_mask)

    # Display the original image, edges, and mask
    cv2.imshow('Original Image', image)
    cv2.imshow('Yellow Segment', yellow_segment)
    cv2.imshow('Red Segment', red_segment)
    cv2.imshow('Green Segment', green_segment)

    # Find contours in the binary mask
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 10000
    # # Fit a bounding rectangle to each contour and draw it on the original image
    for contour in yellow_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #print(w, h)
            aspect_ratio = float(w) / h
            if 1.0 < aspect_ratio:  # Aspect ratio range for the object of interest
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # # Display the original image with rectangles
    cv2.imshow('Image with Rectangles', image)
    #equalized_image = np.stack([cv2.equalizeHist(hsv_image[:, :, j]) if j in [0] else hsv_image[:, :, j] for j in range(3)], axis = -1)
    #cv2.imshow('Equalize', cv2.cvtColor(equalized_image, cv2.COLOR_HSV2BGR))
    #equalized_image = np.stack([cv2.equalizeHist(image[:, :, j]) for j in range(3)], axis = -1)
    #cv2.imshow('Equalize', equalized_image)


    cv2.waitKey(0)