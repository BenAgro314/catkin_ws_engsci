import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import imageio
directory_path = '/home/agrobenj/catkin_ws/images/imx219_images_handheld/'
# List all files and directories in the specified directory
all_files_and_directories = os.listdir(directory_path)
# Filter the list to include only files (exclude directories)
files_only = [f for f in all_files_and_directories if os.path.isfile(os.path.join(directory_path, f))]
files = sorted(files_only)

#files = ['frame_0173.png']#['frame_0137.png', 'frame_0173.png']

intrinsic_matrix = np.array([[800, 0, 320], # needs to be tuned
                             [0, 800, 240],
                             [0, 0, 1]])

images = []



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

for f in files:
    img_path = os.path.join(directory_path, f)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(image)
    #plt.show()


    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color yellow in the HSV color space
    lower_yellow = (10, 0, 0)
    upper_yellow = (40, 255, 255)
    

    # Create a binary mask using the defined yellow range
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Perform morphological operations (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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
    #segmented_yellow = cv2.bitwise_and(image, image, mask=mask)

    # Create a binary mask using the defined yellow range
    lower_green = (40, 0, 0)
    upper_green = (80, 255, 255)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_only = cv2.bitwise_and(image, image, mask=green_mask)

    #lower_red = (120, 0, 0)
    #upper_red = (170, 255, 255)
    ##red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    #red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    #red_only = cv2.bitwise_and(image, image, mask=red_mask)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Calculate the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        percent_green = np.sum(green_mask[y:y+h, x:x+w])/(255 * w * h)
        #percent_red = np.sum(red_mask[y:y+h, x:x+w])/(255 * w * h)


        bbox = [y, x, y+h, x+w] # x_min, y_min, x_max, y_max (flipped because the images are sidways)
        p_box_cam =  reproject_2D_to_3D(bbox, 0.3, intrinsic_matrix)
        #print(p_box_cam)
        # Draw the bounding rectangle on the original image or a blank image
        if percent_green > 0.03:
            cv2.rectangle(segmented_large_groups, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(segmented_large_groups, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # Display the original image, binary mask, and segmented yellow regions
    cv2.imshow('Original Image', image)
    #cv2.imshow('Green Image', green_only)
    #cv2.imshow('Red Image', red_only)
    #cv2.imshow('Binary Mask', mask)
    #cv2.imshow('Segmented Yellow', segmented_large_groups)
    #res_img = np.concatenate((segmented_large_groups, image), axis=1)
    res_img = segmented_large_groups
    #res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    images.append(res_img)
    cv2.imshow('Segmented Groups', images[-1])
    cv2.waitKey(1)
    time.sleep(0.05)
    #cv2.destroyAllWindows()


#imageio.mimsave("test_seg.mp4", images, fps=5)