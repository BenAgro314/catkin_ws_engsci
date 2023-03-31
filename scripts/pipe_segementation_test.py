import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
directory_path = '/home/agrobenj/catkin_ws/images/imx219_images/'
# List all files and directories in the specified directory
all_files_and_directories = os.listdir(directory_path)
# Filter the list to include only files (exclude directories)
files_only = [f for f in all_files_and_directories if os.path.isfile(os.path.join(directory_path, f))]
files = sorted(files_only)

for f in files:
    img_path = os.path.join(directory_path, f)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(image)
    #plt.show()


    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color yellow in the HSV color space
    lower_yellow = (0, 0, 0)
    upper_yellow = (40, 255, 255)

    # Create a binary mask using the defined yellow range
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Perform morphological operations (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find connected components and their statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Set a threshold for the minimum area of the connected components to keep
    min_area_threshold = 100000

    # Create a new binary mask to store the filtered connected components
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)

    # Iterate through the connected components and keep only the large ones
    for i in range(1, num_labels):  # Start from 1 to skip the background component (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_threshold:
            filtered_mask[labels == i] = 255

    # Use the filtered binary mask to segment the large groups of pixels from the original image
    segmented_large_groups = cv2.bitwise_and(image, image, mask=filtered_mask)

    # Use the binary mask to segment the yellow regions from the original image
    segmented_yellow = cv2.bitwise_and(image, image, mask=mask)

    # Display the original image, binary mask, and segmented yellow regions
    #cv2.imshow('Original Image', image)
    #cv2.imshow('Binary Mask', mask)
    #cv2.imshow('Segmented Yellow', segmented_yellow)
    cv2.imshow('Segmented Groups', np.concatenate((segmented_large_groups, image), axis=1))
    cv2.waitKey(1)
    time.sleep(0.1)
    #cv2.destroyAllWindows()