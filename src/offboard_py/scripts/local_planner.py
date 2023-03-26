
from enum import Enum
import rospy
from nav_msgs.msg import Path
from typing import Optional
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from offboard_py.scripts.utils import are_angles_close, pose_stamped_to_numpy, get_config_from_pose_stamped, se2_pose_list_to_path, shortest_signed_angle, transform_twist
import warnings

class LocalPlannerType(Enum):
    NON_HOLONOMIC = 0

class LocalPlanner:
    # idea:
    # we need the drone to travel front-forwards at all times to prevent collisions
    # how: simulate a differential drive robot
    # use trajectory sampling from (v, omega), and select the best one
    # control z independently

    def __init__(self, v_max=0.5, omega_max=1.0, trans_ths=0.15, yaw_ths=0.16, mode=LocalPlannerType.NON_HOLONOMIC):# , num_substeps=10, horizon=1.0):
        self.v_max = v_max
        self.omega_max = omega_max
        self.trans_ths = trans_ths
        self.yaw_ths=yaw_ths # 10 deg

    def get_speed(self, goal_vec: np.array):
        return np.clip(np.linalg.norm(goal_vec), a_min=0, a_max=self.v_max)

    def get_twist(self, t_map_d: PoseStamped, t_map_d_goal: PoseStamped) -> Twist:
        curr_cfg = get_config_from_pose_stamped(t_map_d)
        goal_cfg = get_config_from_pose_stamped(t_map_d_goal)

        goal_vec = goal_cfg[:3] - curr_cfg[:3] # (x, y, z)
        twist_m = Twist()
        goal_vec = self.get_speed(goal_vec) * goal_vec / np.linalg.norm(goal_vec)
        twist_m.linear.x = goal_vec[0]
        twist_m.linear.y = goal_vec[1]
        twist_m.linear.z = goal_vec[2]
        twist_d = transform_twist(twist_m, np.linalg.inv(pose_stamped_to_numpy(t_map_d)))
        return twist_d