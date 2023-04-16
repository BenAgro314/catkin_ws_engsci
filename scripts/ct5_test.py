import cv2
import numpy as np
import os
import time


def multi_scale_template_matching(image, template, scale_range=(0.5, 2.0), scale_step=0.1, ths=0.6):
    # Store the width and height of the template
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    image_gray = cv2.equalizeHist(image_gray)
    image_gray = 255 - cv2.inRange(image_gray, 0, 30)
    #print(mask.max(), mask.min())
    #cv2.imshow('g', mask)

    template_width, template_height = template_gray.shape[::-1]
    
    # Initialize variables to store the best match
    best_scale = None
    best_match = None
    max_corr = -1
    
    # Loop over the scales
    for scale in np.arange(scale_range[0], scale_range[1], scale_step):
        # Resize the template according to the current scale
        resized_template = cv2.resize(template_gray, (int(template_width * scale), int(template_height * scale)))
        
        # Perform template matching
        result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        
        # Find the maximum correlation value
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Update the best match if the current match is better
        if max_val > max_corr:
            max_corr = max_val
            best_scale = scale
            best_match = max_loc
            
    # Compute the coordinates of the bounding box around the best match
    top_left = best_match
    bottom_right = (int(top_left[0] + template_width * best_scale), int(top_left[1] + template_height * best_scale))
    
    # Draw the bounding box on the image
    #cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)


    
    return (top_left, bottom_right), best_scale, max_corr

ys = []
for i in range(6):
    yf = f"/home/agrobenj/catkin_ws/images/y-{i+1}.png"
    ys.append(cv2.imread(yf))


def calculate_iou(rect1, rect2):
    # Unpack the coordinates of the two rectangles
    x1_tl, y1_tl, x1_br, y1_br = rect1[0] + rect1[1]
    x2_tl, y2_tl, x2_br, y2_br = rect2[0] + rect2[1]
    
    # Calculate the coordinates of the intersection rectangle
    x_int_tl = max(x1_tl, x2_tl)
    y_int_tl = max(y1_tl, y2_tl)
    x_int_br = min(x1_br, x2_br)
    y_int_br = min(y1_br, y2_br)
    
    # Calculate the area of the intersection rectangle
    int_area = max(0, x_int_br - x_int_tl) * max(0, y_int_br - y_int_tl)
    
    # Calculate the area of the two input rectangles
    rect1_area = (x1_br - x1_tl) * (y1_br - y1_tl)
    rect2_area = (x2_br - x2_tl) * (y2_br - y2_tl)
    
    # Calculate the area of the union of the two rectangles
    union_area = rect1_area + rect2_area - int_area
    
    # Calculate the IoU
    iou = int_area / union_area
    
    return iou

def match_nums(image, ths = 0.6):

    rects_and_corrs = [None, None, None, None]

    for y_ind, y in enumerate(ys):
        rect, best_scale, max_corr = multi_scale_template_matching(np.copy(image), y, (0.05, 0.5), 0.05)
        #rects.append(rect)
        add = False
        for i, r_and_c in enumerate(rects_and_corrs):
            if r_and_c is None:
                add = True
                break
            r, corr, _ = r_and_c
            iou = calculate_iou(r, rect)
            if iou > ths:
                if max_corr > corr:
                    add = True
                    break
                else:
                    add = False
                    break
        if add:
            rects_and_corrs[i] = (rect, max_corr, y_ind + 1)
                    
                    
    rects_and_corrs = sorted(rects_and_corrs, key = lambda x: x[0][0][0] if x is not None else np.inf)
    #print(rects_and_corrs)
    for el in rects_and_corrs:
        if el is None:
            break
        r, c, n = el
        cv2.rectangle(image, r[0], r[1], (0, 255, 0), 2)
        cv2.putText(image, f"y-{n}", r[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image, [r[-1] if r is not None else r for r in rects_and_corrs]




#f = "/home/agrobenj/catkin_ws/images/ct5_test1.png" 
#f = "/home/agrobenj/catkin_ws/images/ct5_test1_perspective.png" 
f = "/home/agrobenj/rob498_bag_files/ct5_images/frame_0521.png"


image = cv2.imread(f)
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#image = undistort_image(image, K, D)
h, w = image.shape[:2]
sf = 1.0
nw, nh = int(sf * w), int(sf * h)
image = cv2.resize(image, (nw, nh))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#image = detect_and_warp_red_rectangle(image)
#image= homo_match(image, y1)

#for y in ys:
#    image_drawn, best_scale, max_corr = multi_scale_template_matching(np.copy(image), y, (0.05, 0.3), 0.05)
#    print("best_scale:",best_scale)
#    print("max_corr:",max_corr)


image_drawn, nums = match_nums(np.copy(image))
print(f"Numbers: {nums}")
cv2.imshow('Original Image', image_drawn)
cv2.waitKey(0)