#!/usr/bin/env python3

from offboard_py.scripts.utils import numpy_to_pointcloud2, quaternion_to_euler, scale_image
import rospy
import cv2
import tf2_ros
import matplotlib.pyplot as plt
import tf.transformations
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import time

PUB_IMAGE = True
UNDISTORT = False

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

def rotate_image(image, angle):
    # Get the dimensions of the image
    (height, width) = image.shape[:2]

    # Calculate the center of the image
    center = (width // 2, height // 2)

    # Create the rotation matrix using the center and the specified angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation matrix to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def get_mask_from_range(hsv_img, low, high):
    mask = cv2.inRange(hsv_img, low, high)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def reproject_2D_to_3D(bbox, actual_height, K):
    # Extract the focal length (fx) and the optical center (cx, cy) from the intrinsic matrix

    # Calculate the 2D coordinates of the center of the bounding box
    center_x_2D = (bbox[0] + bbox[2]) / 2
    center_y_2D = (bbox[1] + bbox[3]) / 2

    Kinv = np.linalg.inv(K)

    upper_y_2D = np.array([center_x_2D, bbox[1], 1])[:, None]
    lower_y_2D = np.array([center_x_2D, bbox[3], 1])[:, None]

    upper_y_plane = Kinv @ upper_y_2D
    lower_y_plane = Kinv @ lower_y_2D

    depth = actual_height / (lower_y_plane[1] - upper_y_plane[1])

    center_2D = np.array([center_x_2D, center_y_2D, 1])[:, None]
    center_plane = Kinv @ center_2D

    # Reproject the 2D center to 3D
    center_x_3D = center_plane[0] * depth
    center_y_3D = center_plane[1] * depth

    # Return the 3D coordinates of the center of the bounding box
    return center_x_3D, center_y_3D, depth

def reproject_2D_to_3D_v2(height, center, actual_height, K):
    # height
    # center (x, y)
    # Extract the focal length (fx) and the optical center (cx, cy) from the intrinsic matrix

    # Calculate the 2D coordinates of the center of the bounding box
    center_x_2D = center[0]
    center_y_2D = center[1]

    Kinv = np.linalg.inv(K)

    depth = actual_height / (Kinv[1, 1] * height)

    center_2D = np.array([center_x_2D, center_y_2D, 1])[:, None]
    center_plane = Kinv @ center_2D

    # Reproject the 2D center to 3D
    center_x_3D = center_plane[0] * depth
    center_y_3D = center_plane[1] * depth

    # Return the 3D coordinates of the center of the bounding box
    return [center_x_3D.item(), center_y_3D.item(), depth]

class Detector:

    def __init__(self):
        #self.K = np.array(
        #    [
        #        [342.6836426,    0.,         313.55097801],
        #        [  0., 607.63147143, 297.50936008],
        #        [  0.,           0.,           1.        ]
        #    ]
        #)
        #self.D = np.array([[-3.97718724e-01, 3.27660950e-02, -5.45843945e-04, -8.40769238e-03, 9.20723812e-01]])
        self.K = np.array(
            [
                [334.94030171,    0.,         280.0627713],
                [  0., 595.99313333, 245.316628],
                [  0.,           0.,           1.        ]
            ]
        )
        self.D = np.array([[-0.36600591, 0.20973317, -0.00181088, 0.00102208, -0.07504754]])
        self.seg_image_pub= rospy.Publisher("imx219_seg", Image, queue_size=10)
        self.bridge = CvBridge()
        self.marker_pub = rospy.Publisher('/cylinder_marker', Marker, queue_size=10)

        self.det_point_pub = rospy.Publisher("det_points", PointCloud2, queue_size=10)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Define the source and target frames
        self.source_frame = 'map'
        self.target_frame = 'base_link'

        self.prev_rects = []
        self.image_sub= rospy.Subscriber("imx219_image", Image, callback = self.image_callback)

        self.shape = (540, 540)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, self.shape, cv2.CV_32FC1)

    def publish_cylinder_marker(self, w_fit, C_fit, r_fit, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id  # Frame in which the marker is defined
        marker.header.stamp = rospy.Time.now()  # Timestamp
        marker.ns = 'cylinder'  # Namespace for the marker
        marker.id = 0  # Unique ID for the marker
        marker.type = Marker.CYLINDER  # Marker type: cylinder
        marker.action = Marker.ADD  # Action: add/modify the marker

        # Compute the quaternion from the orientation vector
        #quaternion = tf.transformations.quaternion_about_axis(0, np.array([1, 0, 0]))
        angle = np.arccos(w_fit[2])
        axis = np.cross([0, 0, 1], w_fit)
        axis = axis / np.linalg.norm(axis)
        quaternion = tf.transformations.quaternion_about_axis(angle, axis)

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

        image_time = msg.header.stamp
        #image_time = rospy.Time(0)
        if not  self.tf_buffer.can_transform('map', 'base_link', image_time, timeout=rospy.Duration(0.2)):
            print(f"Obstacle detector not up yet. Image time - current time: {image_time.to_sec() - rospy.Time.now().to_sec()} s")
            return
        t_map_base = self.tf_buffer.lookup_transform(
        "map", "base_link", image_time).transform
        q = t_map_base.rotation
        roll, pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        assert self.shape == image.shape[:2]
        #image = undistort_image(image, self.K, self.D)
        if UNDISTORT:
            image = cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        image = rotate_image(image, np.rad2deg(pitch))
        scale = 1.0
        image = scale_image(image, scale)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        sat = hsv_image[:, :, 1]#.astype(np.int16)
        sat = cv2.equalizeHist(sat)
        hsv_image[:, :, 1] = sat

        # Define the lower and upper bounds for the color yellow in the HSV color space
        lower_yellow = (10, 200, 0)
        upper_yellow = (80, 255, 255)
        
        #lower_red = (0, 0, 0)
        #upper_red = (10, 255, 255)

        lower_green = (40, 0, 0)
        upper_green = (80, 255, 255)

        # Create a binary mask using the defined yellow range
        yellow_mask = get_mask_from_range(hsv_image, lower_yellow, upper_yellow)
        yellow_mask = cv2.blur(yellow_mask, (21, 5))
        #red_mask = get_mask_from_range(hsv_image, lower_red, upper_red)
        green_mask = get_mask_from_range(hsv_image, lower_green, upper_green)
        

        yellow_segment = cv2.bitwise_and(image, image, mask=yellow_mask)
        #red_segment = cv2.bitwise_and(image, image, mask=red_mask)
        #green_segment = cv2.bitwise_and(image, image, mask=green_mask)

        # Find contours in the binary mask
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 1000 * scale
        det_points = []

        for contour in yellow_contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                rect = cv2.minAreaRect(contour)

                angle = rect[-1]

                if abs(angle) < 10:
                    w = rect[1][0]
                    h = rect[1][1]
                elif abs(angle) > 80:
                    h = rect[1][0]
                    w = rect[1][1]
                else:
                    continue
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                y_max = np.max(box[:, 1])
                y_min = np.max(box[:, 1])
                aspect_ratio = float(w) / h

                if 2.0 < aspect_ratio:  # Aspect ratio range for the object of interest

                    if (y_min < image.shape[0]//8) == (y_max > 7 * image.shape[0]//8):
                        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        #bbox = [x/scale, y/scale, (x+w)/scale, (y+h)/scale] # x_min, y_min, x_max, y_max
                        #p_box_cam =  reproject_2D_to_3D(bbox, 0.3, self.K)
                        p_box_cam =  reproject_2D_to_3D_v2(h, rect[0], 0.3, self.K)


                        mask = np.zeros_like(hsv_image[:,:,0])
                        cv2.fillPoly(mask, [box], 255)
                        #print(np.sum(hsv_image[..., 2][mask]) / np.sum(mask))
                        green_val = np.sum(np.logical_and(mask ,green_mask)) / np.sum(mask)

                        if green_val > 1e-4:
                            if PUB_IMAGE:
                                cv2.drawContours(yellow_segment,[box],0,(0,255,0),2)
                            p_box_cam.append(ord('g')) # green
                        else: 
                            if PUB_IMAGE:
                                cv2.drawContours(yellow_segment,[box],0,(0,0,255),2)
                            p_box_cam.append(ord('r')) # red

                        det_points.append(np.array(p_box_cam))

        det_points = np.stack(det_points, axis = 0) if len(det_points) > 0 else np.array([])
        pc = numpy_to_pointcloud2(det_points, frame_id='map', extra_features=['c'])
        pc.header.stamp = image_time
        self.det_point_pub.publish(pc)

        #msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if PUB_IMAGE:
            #msg = self.bridge.cv2_to_imgmsg(image)
            msg = self.bridge.cv2_to_imgmsg(yellow_segment)
            #msg.header.stamp = rospy.Time.now()
            self.seg_image_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("obstacle_detector")
    detector = Detector()
    rospy.spin()
