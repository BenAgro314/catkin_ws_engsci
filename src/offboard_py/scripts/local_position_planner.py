#!/usr/bin/env python3
from threading import Lock
from copy import deepcopy
import time
from functools import partial
from offboard_py.scripts.utils import are_angles_close, config_to_pose_stamped, get_config_from_pose_stamped, numpy_to_pose, shortest_signed_angle, slerp_pose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
import networkx as nx
from nav_msgs.msg import Path
from std_msgs.msg import Header
import pyastar2d

def coll_free(p1, p2, coll_fn, steps=10):
    pts = np.linspace(p1, p2, steps)
    return not np.any(coll_fn(pts))

def inds_to_robot_circle(self, inds, vehicle_radius = None):
    if vehicle_radius is None:
        vehicle_radius = self.vehicle_radius
    radius = vehicle_radius / self.map_res # radius in px

    N = inds.shape[0]
    indexings = np.arange(np.floor(-radius), np.ceil(radius),dtype=np.int64)

    # Creates a addition map that adds onto center point to produce points in circle
    x_indexings, y_indexings= np.meshgrid(indexings,indexings)
    x_indexings = np.tile(x_indexings.ravel(order='F'),(N,1)) # (N,len_indexings^2)
    y_indexings = np.tile(y_indexings.ravel(order='F'),(N,1)) # (N,len_indexings^2)
    distances = np.sqrt(x_indexings**2 + y_indexings**2) # (N,len_indexings^2)

    x_circlePts = np.tile(inds[:,1],(x_indexings.shape[1],1)).T + x_indexings # (N,len_indexings^2)
    y_circlePts = np.tile(inds[:,0],(y_indexings.shape[1],1)).T  + y_indexings # (N,len_indexings^2)

    x_circlePts = x_circlePts[distances < radius].reshape((N,-1))
    x_circlePts[x_circlePts >= self.map_width] = self.map_width-1
    x_circlePts[x_circlePts<0]=0

    y_circlePts = y_circlePts[distances < radius].reshape((N,-1))
    y_circlePts[y_circlePts >= self.map_height] = self.map_height-1
    y_circlePts[y_circlePts<0]=0
    
    circlePts = np.stack((y_circlePts,x_circlePts),axis=-1) # (num_pts, num_pts_per_circle, 2)

    return circlePts # (num_pts, num_pts_per_circle, 2)

class LocalPlanner:

    def __init__(self):
        self.map_sub= rospy.Subscriber("occ_map", OccupancyGrid, callback = self.map_callback)
        self.map = None
        self.map_width = None
        self.map_height = None
        self.map_res = None
        self.map_origin = None

        self.waypoint_trans_ths = 0.08 # 0.08 # used in pose_is_close
        self.waypoint_yaw_ths = np.deg2rad(10.0) # used in pose_is_close

        self.map_lock = Lock()
        self.path_pub = rospy.Publisher('local_plan', Path, queue_size=10)

        self.vehicle_radius = 0.3
        self.current_path = None

    def point_to_ind(self, pt):
        # pt.shape == (3, 1) or (2, 1)
        assert self.map_res is not None
        x = pt[0, 0]
        y = pt[1, 0]

        row_ind = (y - self.map_origin.y) // (self.map_res) 
        col_ind = (x - self.map_origin.x) // (self.map_res) 

        return (row_ind, col_ind)

    def path_to_path_message(self, path, t_map_d, t_map_d_goal, frame_id='map'):

        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = frame_id

        n = len(path)
        for i, point1 in enumerate(path):
            p = slerp_pose(t_map_d.pose, t_map_d_goal.pose, rospy.Time(0), rospy.Time(1), rospy.Time((i+1)/n), 'map')
            p.header.stamp = rospy.Time.now()
            _, _, z, _, _, yaw = get_config_from_pose_stamped(p)

            x1 = point1[1] * self.map_res + self.map_origin.x
            y1 = point1[0] * self.map_res + self.map_origin.y
            p1 = np.array([x1, y1])
            pose = config_to_pose_stamped(x1, y1, z, yaw, frame_id=frame_id)
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


    def plan(self, occ_map, t_map_d, t_map_d_goal, collision_fn):
        x1, y1, z1, _, _, yaw1 = get_config_from_pose_stamped(t_map_d)
        x2, y2, z2, _, _, yaw2 = get_config_from_pose_stamped(t_map_d_goal)


        weights = np.ones_like(occ_map).astype(np.float32)
        weights[occ_map > 50] = np.inf
        r1, c1 = self.point_to_ind(np.array([x1, y1, 0])[:, None]) # turn those into indices into occ map
        r2, c2 = self.point_to_ind(np.array([x2, y2, 0])[:, None])

        path = pyastar2d.astar_path(weights, (int(r1), int(c1)), (int(r2), int(c2)), allow_diagonal=True)
        if path is None:
            print(f"Failed to find path! Staying still instead")
            path = [(r1, c1)]

        #if len(path) == 1:
            #path = path + path

        # shorten path
        shorter_path = []
        i = 0
        while i < len(path):
            shorter_path.append(path[i])
            j = i+1
            while j < len(path)-1 and coll_free(path[i], path[j], collision_fn):
                j+=1
            i = j
        #print(f"Astar time: {dt}")
        #path_msg = self.path_to_path_message(shorter_path, z2, yaw, frame_id='map')

        path_msg = self.path_to_path_message(shorter_path, t_map_d, t_map_d_goal, 'map')
        if len(path_msg.poses) == 1:
            path_msg.poses = [path_msg.poses[0]] * 2
        path_msg.poses[-1] = t_map_d_goal # to handle discretization

        return path_msg


    def pose_is_close(self, pose1: PoseStamped, pose2: PoseStamped):
        # TOOD: include some meaasure of rotation diff
        cfg1 = get_config_from_pose_stamped(pose1)
        cfg2 = get_config_from_pose_stamped(pose2)
        trans_is_close =  np.linalg.norm(cfg1[:3] - cfg2[:3]) < self.waypoint_trans_ths

        yaw_is_close = are_angles_close(cfg1[-1], cfg2[-1], self.waypoint_yaw_ths)
        return trans_is_close and yaw_is_close

    def inds_to_robot_circle(self, inds, vehicle_radius = None):
        if vehicle_radius is None:
            vehicle_radius = self.vehicle_radius
        radius = vehicle_radius / self.map_res # radius in px

        N = inds.shape[0]
        indexings = np.arange(np.floor(-radius), np.ceil(radius),dtype=np.int64)

        # Creates a addition map that adds onto center point to produce points in circle
        x_indexings, y_indexings= np.meshgrid(indexings,indexings)
        x_indexings = np.tile(x_indexings.ravel(order='F'),(N,1)) # (N,len_indexings^2)
        y_indexings = np.tile(y_indexings.ravel(order='F'),(N,1)) # (N,len_indexings^2)
        distances = np.sqrt(x_indexings**2 + y_indexings**2) # (N,len_indexings^2)

        x_circlePts = np.tile(inds[:,1],(x_indexings.shape[1],1)).T + x_indexings # (N,len_indexings^2)
        y_circlePts = np.tile(inds[:,0],(y_indexings.shape[1],1)).T  + y_indexings # (N,len_indexings^2)

        x_circlePts = x_circlePts[distances < radius].reshape((N,-1))
        x_circlePts[x_circlePts >= self.map_width] = self.map_width-1
        x_circlePts[x_circlePts<0]=0

        y_circlePts = y_circlePts[distances < radius].reshape((N,-1))
        y_circlePts[y_circlePts >= self.map_height] = self.map_height-1
        y_circlePts[y_circlePts<0]=0
        
        circlePts = np.stack((y_circlePts,x_circlePts),axis=-1) # (num_pts, num_pts_per_circle, 2)

        return circlePts # (num_pts, num_pts_per_circle, 2)

    def path_has_collision(self, path: Path, collision_fcn):
        for pose in path.poses:
            x1,y1 = get_config_from_pose_stamped(pose)[:2] # get current position (x,y)
            r1, c1 = self.point_to_ind(np.array([x1, y1, 0])[:, None]) # turn those into indices into occ map
            if collision_fcn(np.array([r1, c1])):
                return True
        return False

    def run(self, t_map_d: PoseStamped, t_map_d_goal: PoseStamped) -> Path:

        with self.map_lock:
            if self.map is None:
                return
            occ_map = self.map.copy()[:, :, 0]

        def collision_fn(pt):
            if len(pt.shape) == 2:
                pts = np.round(pt).astype(np.int32)
            else:
                row=int(round(pt[0]))
                col=int(round(pt[1]))
                pts = np.array([row, col])[None, :] # (1, 2)
            circle_pts = self.inds_to_robot_circle(pts)
            rows = circle_pts[..., 0]
            cols = circle_pts[..., 1]
            return np.any(occ_map[rows, cols] > 60, axis = 1)

        if self.current_path is None or len(self.current_path.poses) == 0 or not self.pose_is_close(t_map_d_goal, self.current_path.poses[-1]) or self.path_has_collision(self.current_path, collision_fn):
            self.current_path = self.plan(occ_map, t_map_d, t_map_d_goal, collision_fn)
        else:
            # check if we are near first waypoint on path
            assert len(self.current_path.poses) > 0
            if self.pose_is_close(t_map_d, self.current_path.poses[0]):
                self.current_path.poses = self.current_path.poses[1:]
            if len(self.current_path.poses) == 0:
                self.current_path.poses = [t_map_d_goal]

        #self.path = path_msg

#        x1, y1, z1, _, _, yaw1 = get_config_from_pose_stamped(t_map_d)
#        x2, y2, z2, _, _, yaw2 = get_config_from_pose_stamped(t_map_d_goal)
#        dyaw = np.abs(shortest_signed_angle(yaw1, yaw2))
#        dd = np.sqrt((x1 - x2)**2  + (y1 - y2)**2 + (z1 - z2)**2)
#
#        n = max(1, int((dd / 0.2)) + int((dyaw / np.deg2rad(10))))
#
#        poses = []
#        for i in range(n):
#            p = slerp_pose(t_map_d.pose, t_map_d_goal.pose, rospy.Time(0), rospy.Time(1), rospy.Time((i+1)/n), 'map')
#            p.header.stamp = rospy.Time.now()
#            poses.append(p)
#
#        path_msg = Path()
#        path_msg.header = Header()
#        path_msg.header.stamp = rospy.Time.now()
#        path_msg.header.frame_id = 'map'
#
#        path_msg.poses = poses
#        self.path_pub.publish(path_msg)
#
        self.path_pub.publish(self.current_path)
        return self.current_path
