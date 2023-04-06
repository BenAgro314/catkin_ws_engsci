#!/usr/bin/env python3
from threading import Lock
from copy import deepcopy
import time
from functools import partial
from offboard_py.scripts.utils import config_to_pose_stamped, get_config_from_pose_stamped, numpy_to_pose, shortest_signed_angle, slerp_pose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
import networkx as nx
from nav_msgs.msg import Path
from std_msgs.msg import Header


class LocalPlanner:

    def __init__(self):
        self.map_sub= rospy.Subscriber("occ_map", OccupancyGrid, callback = self.map_callback)
        self.map = None
        self.map_width = None
        self.map_height = None
        self.map_res = None
        self.map_origin = None

        self.map_lock = Lock()
        self.path_pub = rospy.Publisher('local_plan', Path, queue_size=10)

        self.vehicle_radius = 0.3

    def point_to_ind(self, pt):
        # pt.shape == (3, 1) or (2, 1)
        assert self.map_res is not None
        x = pt[0, 0]
        y = pt[1, 0]

        row_ind = (y - self.map_origin.y) // (self.map_res) 
        col_ind = (x - self.map_origin.x) // (self.map_res) 

        return (row_ind, col_ind)

    def path_to_path_message(self, path, z, yaw, frame_id='map'):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = frame_id

        for point1, point2 in zip(path[:-1], path[1:]): 
            x1 = point1[1] * self.map_res + self.map_origin.x
            y1 = point1[0] * self.map_res + self.map_origin.y
            p1 = np.array([x1, y1])

            x2 = point2[1] * self.map_res + self.map_origin.x
            y2 = point2[0] * self.map_res + self.map_origin.y
            p2 = np.array([x2, y2])
            num_pts = max(2, np.int(np.round(np.linalg.norm(p1 - p1) / (self.map_res/2))))
            pts = np.linspace(p1, p2, num = num_pts)

            for pt in pts:
                x,y = pt
                pose = config_to_pose_stamped(x, y, z, yaw, frame_id=frame_id)
                pose.header = path_msg.header
                path_msg.poses.append(pose)

        return path_msg

    def map_callback(self, map_msg):
        with self.map_lock:
            self.map_width = map_msg.info.width
            self.map_height = map_msg.info.height
            self.map_res = map_msg.info.resolution
            self.map_origin = map_msg.info.origin.position
            self.map = np.array(map_msg.data, dtype=np.uint8).reshape((self.map_height, self.map_width, 1))
        pass


    def run(self, t_map_d: PoseStamped, t_map_d_goal: PoseStamped) -> Path:

        x1, y1, z1, _, _, yaw1 = get_config_from_pose_stamped(t_map_d)
        x2, y2, z2, _, _, yaw2 = get_config_from_pose_stamped(t_map_d_goal)

        dyaw = np.abs(shortest_signed_angle(yaw1, yaw2))
        dd = np.sqrt((x1 - x2)**2  + (y1 - y2)**2 + (z1 - z2)**2)

        n = max(1, int((dd / 0.2)) + int((dyaw / np.deg2rad(10))))

        poses = []
        for i in range(n):
            p = slerp_pose(t_map_d.pose, t_map_d_goal.pose, rospy.Time(0), rospy.Time(1), rospy.Time((i+1)/n), 'map')
            p.header.stamp = rospy.Time.now()
            poses.append(p)

        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'map'

        path_msg.poses = poses
        self.path_pub.publish(path_msg)

        return path_msg
