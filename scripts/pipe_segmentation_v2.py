import cv2
import numpy as np
import os
import time
import imageio
import glob

#files = ['frame_0137.png', 'frame_0173.png', 'frame_0070.png', 'frame_0304.png']
files = glob.glob('/home/agrobenj/catkin_ws/images/flying_v2/*.png')
#print(files)
files = sorted(files, key=lambda x: x[-8:])

K = np.array([[1581.5, 0, 1034.7], # needs to be tuned
                            [0, 1588.7, 557.16],
                            [0, 0, 1]])
D = np.array([[-0.37906155, 0.2780121, -0.00092033, 0.00087556, -0.21837157]])


images = []

def undistort_image(img, K, D):
    # Undistort the image
    img = cv2.undistort(img, K, D)
    return img

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


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
    mask = cv2.inRange(hsv_img, low, high)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# template = cv2.imread('template.png')    
# template_height, template_width, _ = template.shape
# method = cv2.TM_CCOEFF_NORMED

prev_rects = []
for f in files:


    image = cv2.imread(f)
    image = undistort_image(image, K, D)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = image[:, image.shape[1]//2:]


    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #hue = hsv_image[:, :, 0]#.astype(np.int16)
    #cv2.imshow('Hue', hue)
    sat = hsv_image[:, :, 1]#.astype(np.int16)
    sat = cv2.equalizeHist(sat)
    hsv_image[:, :, 1] = sat
    #print(sat.min(), sat.max())
    #cv2.imshow('Sat', sat)
    #target_hue = 30
    #d1 = np.abs(hue - target_hue)
    #d  = np.minimum(d1, 180 - d1)
    #cv2.imshow('Hue diff', d.astype(np.uint8))


    # Define the lower and upper bounds for the color yellow in the HSV color space
    lower_yellow = (3, 240, 0)
    upper_yellow = (80, 255, 255)
    
    lower_red = (0, 0, 0)
    upper_red = (5, 255, 255)

    lower_green = (40, 0, 0)
    upper_green = (80, 255, 255)

    # Create a binary mask using the defined yellow range
    yellow_mask = get_mask_from_range(hsv_image, lower_yellow, upper_yellow)
    yellow_mask = cv2.blur(yellow_mask, (21, 5))
    #yellow_mask[:yellow_mask.shape[0]//5, :] = 0
    #yellow_mask[4*yellow_mask.shape[0]//5:, :] = 0
    #yellow_mask[:, :yellow_mask.shape[1]//4] = 0
    #yellow_mask[:, 4*yellow_mask.shape[1]//5:] = 0
    red_mask = get_mask_from_range(hsv_image, lower_red, upper_red)
    green_mask = get_mask_from_range(hsv_image, lower_green, upper_green)
    

    yellow_segment = cv2.bitwise_and(image, image, mask=yellow_mask)
    red_segment = cv2.bitwise_and(image, image, mask=red_mask)
    green_segment = cv2.bitwise_and(image, image, mask=green_mask)

    # Display the original image, edges, and mask
    #cv2.imshow('Original Image', image)
    cv2.imshow('Yellow Segment', yellow_segment)
    #cv2.imshow('Red Segment', red_segment)
    #cv2.imshow('Green Segment', green_segment)

    # Find contours in the binary mask
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_copy = image.copy()

    min_area = 30000
    # # Fit a bounding rectangle to each contour and draw it on the original image
    for contour in yellow_contours:
        area = cv2.contourArea(contour)
    
        if area > min_area:
            #rect = cv2.minAreaRect(contour)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            #cv2.drawContours(image,[box],0,(0,0,255),2)

            #rows,cols = image.shape[:2]
            #[vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
            #lefty = int((-x*vy/vx) + y)
            #righty = int(((cols-x)*vy/vx)+y)
            #cv2.line(image,(cols-1,righty),(0,lefty),(0,255,0),2)
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            aspect_ratio = float(w) / h
            if 3 < aspect_ratio:  # Aspect ratio range for the object of interest

                # filter out boxes on the top
                if x == 0 and not x+w > image.shape[1] //2:
                    continue


                max_iou = 0
                for x_other, y_other, w_other, h_other in prev_rects:
                    bbox_curr = {
                        "x1": 0,
                        "x2": 0+w,
                        "y1": 0,
                        "y2": 0+h,
                    }
                    bbox_other = {
                        "x1": 0,
                        "x2": 0+w_other,
                        "y1": 0,
                        "y2": 0+h_other,
                    }
                    iou = get_iou(bbox_curr, bbox_other)
                    if iou > max_iou:
                        max_iou = iou
                #print(max_iou)
                if max_iou < 0.80: # must find match in previous frame
                    continue

                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bbox = [x, y, x+w, y+h] # x_min, y_min, x_max, y_max
                p_box_cam =  reproject_2D_to_3D(bbox, 0.3, K)
                percent_green = np.sum(green_mask[y:y+h, x:x+w])/(255 * w * h)
                percent_red = np.sum(red_mask[y:y+h, x:x+w])/(255 * w * h)
                if percent_green > 0.03:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                elif percent_red > 0.03:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    prev_rects = []

    for contour in yellow_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 3 < aspect_ratio:
                prev_rects.append((x, y, w, h))

    # # Display the original image with rectangles
    cv2.imshow('Image with Rectangles', image)
    images.append(image)
    #equalized_image = np.stack([cv2.equalizeHist(hsv_image[:, :, j]) if j in [0] else hsv_image[:, :, j] for j in range(3)], axis = -1)
    #cv2.imshow('Equalize', cv2.cvtColor(equalized_image, cv2.COLOR_HSV2BGR))
    #equalized_image = np.stack([cv2.equalizeHist(image[:, :, j]) for j in range(3)], axis = -1)
    #cv2.imshow('Equalize', equalized_image)

    # result = cv2.matchTemplate(img_copy, template, method)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #     top_left = min_loc
    # else:
    #     top_left = max_loc
    # bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
    # cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 2)
    # cv2.imshow('Template Matching', img_copy)

    #cv2.imshow('Template Matching', result)


    cv2.waitKey(1)

cv2.destroyAllWindows()
imageio.mimsave("out.mp4", [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images], fps=10)