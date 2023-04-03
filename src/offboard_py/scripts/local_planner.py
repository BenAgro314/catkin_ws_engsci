#!/usr/bin/env python3
from threading import Lock
import time
from functools import partial
from offboard_py.scripts.utils import config_to_pose_stamped, get_config_from_pose_stamped
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
import networkx as nx
from nav_msgs.msg import Path
from std_msgs.msg import Header
import random
from scipy.spatial import KDTree
import heapq

# idea:
# - takes in current waypoint and current drone pose (in map frame)
# - returns a set of poses to follow to reach that waypoint
# - take best step towards that path

def coll_free(p1, p2, coll_fn, steps=10):
    pts = np.linspace(p1, p2, steps)
    return not np.any(coll_fn(pts))

def dijkstra(graph, start, goal, dist_fn):
    # Initialize the priority queue and distances
    queue = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Initialize the dictionary to store the shortest path tree
    shortest_path_tree = {}

    while queue:
        # Get the node with the smallest distance from the priority queue
        current_dist, current_node = heapq.heappop(queue)

        # If the goal node is reached, construct the shortest path
        if current_node == goal:
            path = [goal]
            while current_node != start:
                current_node = shortest_path_tree[current_node]
                path.append(current_node)
            return path[::-1]

        # If the current node hasn't been visited yet
        if current_dist == distances[current_node]:
            for neighbor in graph[current_node]:
                new_dist = distances[current_node] + dist_fn(current_node, neighbor)
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor))
                    shortest_path_tree[neighbor] = current_node

    return None  # No path found

def prm_star(start_ind, goal_ind, coll_fn, samples, k=None):
    # samples.shape = (N, 2)
    num_samples = samples.shape[0]
    if k is None:
        k = int(np.log(num_samples))

    free_samples = [sample for sample in samples if not coll_fn(sample)]
    free_samples_kd_tree = KDTree(free_samples)

    graph = {i: set() for i, _ in enumerate(free_samples)}

    for i, sample in enumerate(free_samples):
        _, neighbors = free_samples_kd_tree.query(sample, k=k)
        for j in neighbors:
            if i != j and coll_free(sample, free_samples[j], coll_fn, steps=int(round(np.linalg.norm(sample - free_samples[j])))):
                graph[i].add(j)
                graph[j].add(i)

    def dist_fn(curr, next):
        return np.linalg.norm(free_samples[curr] - free_samples[next])

    return [free_samples[i] for i in dijkstra(graph, start_ind, goal_ind, dist_fn)]

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
        pass

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

        for point in path:
            x = point[1] * self.map_res + self.map_origin.x
            y = point[0] * self.map_res + self.map_origin.y
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

    def occupancy_grid_to_graph(self, grid, threshold=50):
        height, width = self.map_height, self.map_width
        graph = nx.grid_2d_graph(height, width)
        for y in range(height):
            for x in range(width):
                if grid[y, x] > threshold:
                    graph.remove_node((y, x))

        return graph



    def get_plan(self, t_map_d: PoseStamped, t_map_d_goal: PoseStamped) -> Path:
        with self.map_lock:
            if self.map is None:
                return Path()
            occ_map = self.map.copy()

            
        x1,y1 = get_config_from_pose_stamped(t_map_d)[:2]
        x2,y2,z2, _, _, yaw = get_config_from_pose_stamped(t_map_d_goal)

        r1, c1 = self.point_to_ind(np.array([x1, y1, 0])[:, None])
        r2, c2 = self.point_to_ind(np.array([x2, y2, 0])[:, None])

        def collision_fn(pt):
            if len(pt.shape) == 2:
                pts = np.round(pt).astype(np.int32)
                rows = pts[:, 0]
                cols = pts[:, 1]
                return occ_map[rows, cols] > 50
            else:
                row=int(round(pt[0]))
                col=int(round(pt[1]))
                return occ_map[row, col] > 50

        start = np.array([r1, c1])
        goal = np.array([r2, c2])
        st = coll_free(start, goal, collision_fn, steps=int(round(np.linalg.norm(start - goal))))
        if st: 
            path = [start, goal] 
        else:
            samples = np.random.rand(1000, 2) * np.array([self.map_height, self.map_width])[None]
            print(samples)
            samples = samples.astype(np.int32)
            samples = np.concatenate((start[None], goal[None], samples), axis = 0)
            t = time.time()
            path = prm_star(0, 1, collision_fn, samples)
            print(f"PRM* Time: {time.time() - t}")
        path_msg = self.path_to_path_message(path, z2, yaw, frame_id='map')

        self.path_pub.publish(path_msg)
        return path_msg


