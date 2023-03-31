#!/usr/bin/env python3

import rospy
import cv2
import matplotlib.pyplot as plt
import tf.transformations
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker

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

class Detector:

    def __init__(self):
        self.intrinsic_matrix = np.array([[800, 0, 320], # needs to be tuned
                                    [0, 800, 240],
                                    [0, 0, 1]])
        self.image_sub= rospy.Subscriber("imx219_image", Image, callback = self.image_callback)
        self.seg_image_pub= rospy.Publisher("imx219_seg", Image, queue_size=10)
        self.bridge = CvBridge()
        self.marker_pub = rospy.Publisher('/cylinder_marker', Marker, queue_size=10)

    def publish_cylinder_marker(self, w_fit, C_fit, r_fit, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id  # Frame in which the marker is defined
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
        self.marker_pub.publish(marker)

    def image_callback(self, msg: Image):

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
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

        # Find contours in the binary mask
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Calculate the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)

            bbox = [y, x, y+h, x+w] # x_min, y_min, x_max, y_max (flipped because the images are sidways)
            p_box_cam =  reproject_2D_to_3D(bbox, 0.3, self.intrinsic_matrix)
            self.publish_cylinder_marker(np.array([1, 0, 0]), p_box_cam, 0.15, frame_id="imx219")
            #print(p_box_cam)
            # Draw the bounding rectangle on the original image or a blank image
            cv2.rectangle(segmented_large_groups, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #res_img = np.concatenate((segmented_large_groups, image), axis=1)
        #res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        #cv2.imshow('Segmented Groups', res_img)
        #cv2.waitKey(1)

        seg_image_msg = self.bridge.cv2_to_imgmsg(segmented_large_groups, encoding='rgb8')
        seg_image_msg.header.stamp = rospy.Time.now()
        self.seg_image_pub.publish(seg_image_msg)


if __name__ == "__main__":
    rospy.init_node("obstacle_detector")
    detector = Detector()
    rospy.spin()