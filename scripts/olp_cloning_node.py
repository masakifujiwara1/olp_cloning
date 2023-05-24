#!/usr/bin/env python3
import tf
import sys
import copy
import time
import os
import csv
import math
import random
import numpy as np

from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Twist, Pose
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty, Trigger
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseActionResult

# from torch.utils.tensorboard import SummaryWriter

from olp_cloning_net_off import *

import rospy
import roslib
roslib.load_manifest('olp_cloning')

MODEL_NAME = 'turtlebot3_burger'

EPISODE = 100
TEST = True

# to do random target position 
# X = 1.5
# Y = 1.5

class Train_env:
    def __init__(self):
        # rospy.wait_for_service('/gazebo/rest_world')
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        self.path = roslib.packages.get_pkg_dir('olp_cloning') + '/dataset/'
        self.save_path = roslib.packages.get_pkg_dir('olp_cloning') + '/dataset/' + self.start_time + '/dataset_' + str(EPISODE)
        self.save_scan_path = roslib.packages.get_pkg_dir('olp_cloning') + '/dataset/' + self.start_time + '/dataset_' + str(EPISODE) + '_scan'
        os.makedirs(self.path + self.start_time)

        self.load_path = roslib.packages.get_pkg_dir('olp_cloning') + '/data/20230524_12:41:24/model_gpu.pt'

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.clear_costmap = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        self.set_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_pos = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.start_wp = rospy.Service('/start_wp_nav', Trigger, self.start_wp_callback)
        self.target_pos_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.target_pos_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.target_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.initial_pos = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.cmd_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        self.result_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.result_callback)

        self.dl = deep_learning()
        
        self.episode = 1
        self.step = 1

        self.listener = tf.TransformListener()

        self.collect_flag = True

        self.pos_candidate = []
        self.read_path = '/home/fmasa/catkin_ws/src/olp_cloning/config/set_pose.csv'

        self.cmd_vel = Twist()
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0

        self.sub_cmd_vel = Twist()

        self.ps_map = PoseStamped()
        self.ps_map.header.frame_id = 'map'

        self.ps_estimate = PoseStamped()
        self.ps_estimate.header.frame_id = 'map'

        self.init_pos = PoseWithCovarianceStamped()
        self.init_pos.header.frame_id = 'map'

        self.scan = LaserScan()
        self.min_s = 0
        self.is_goal_reached = False
        self.wait_scan = False

        self.read_csv()

        self.sub_target = False
        self.start_flag = False

        if TEST:
            self.dl.load(self.load_path)

    def loop(self):
        if not TEST:
            if self.episode == EPISODE + 1:
                os.system('killall roslaunch')
                sys.exit()
            print(self.episode, self.step)
            target = self.tf_target2robot()
            # print(target, self.sub_cmd_vel)
            if not self.wait_scan:
                return
            if self.collect_flag:
                self.collect_dataset(target)
                self.step += 1
            self.check_end()
        else:
            if not self.wait_scan:
                return
            if not self.sub_target:
                return
            if not self.start_flag:
                return
            target = self.tf_target2robot()
            v, w = self.dl.act(self.scan.ranges, target)
            self.cmd_vel.linear.x = v
            self.cmd_vel.angular.z = w
            self.cmd_pub.publish(self.cmd_vel)

    def start_wp_callback(self, msg):
        self.start_flag = True

    def cmd_vel_callback(self, msg):
        self.sub_cmd_vel.linear = msg.linear
        self.sub_cmd_vel.angular = msg.angular

    def target_callback(self, msg):
        if TEST:
            self.ps_map.pose = msg.pose
            self.sub_target = True

    def result_callback(self, msg):
        self.is_goal_reached = True

    def scan_callback(self, msg):
        scan_ranges = np.array(msg.ranges)
        scan_ranges[np.isinf(scan_ranges)] = msg.range_max
        # print(len(scan_ranges))
        self.scan.ranges = scan_ranges
        self.min_s = min(self.scan.ranges)
        self.wait_scan = True

    def collect_dataset(self, target):
        # dataset, episode, step, linear.x, angular.z, target_l, scan_data
        line = ["dataset", str(self.episode), str(self.step), str(self.sub_cmd_vel.linear.x), str(self.sub_cmd_vel.angular.z), str(target)]
        with open(self.save_path + '.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(line)
        
        with open(self.save_scan_path + '.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(self.scan.ranges)

    def tf_target2robot(self):
        try:
            (trans, rot) = self.listener.lookupTransform('/odom', '/base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        ps_base = self.listener.transformPose('/odom', self.ps_map)

        dx = ps_base.pose.position.x - trans[0]
        dy = ps_base.pose.position.y - trans[1]
        dist = math.sqrt(dx*dx + dy*dy)

        angle = math.atan2(dy, dx)
        quat = tf.transformations.quaternion_from_euler(0, 0, angle)
        (_, _, yaw) = tf.transformations.euler_from_quaternion(quat)
        angle_diff = yaw - tf.transformations.euler_from_quaternion(rot)[2]
        robot_angle = tf.transformations.euler_from_quaternion(rot)[2]

        # print('Distance: %.2f m' % dist)
        # print('Angle: %.2f rad' % angle_diff)

        # print(robot_angle)

        self.target_l = [dist, angle_diff, robot_angle]
        self.target_l = np.array(self.target_l)

        return self.target_l
    
    def pose_estimate(self, state):
        # self.init_pos.pose.pose.position.x = x
        # self.init_pos.pose.pose.position.y = y
        # self.init_pos.pose.pose.orientation.z = theta
        self.init_pos.pose.pose = state
        self.init_pos.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]
        self.initial_pos.publish(self.init_pos)

    def check_end(self):
        flag = False
        if self.target_l[0] < 0.5:
            self.collect_flag = False
        if self.is_goal_reached:
        # if self.target_l[0] < 0.4 or self.is_goal_reached:
            flag = True
            self.cmd_pub.publish(self.cmd_vel)
        elif self.min_s < 0.12:
            flag = True
        
        if flag:
            self.episode += 1
            self.is_goal_reached = False
            self.set_pos_and_ang()
            # time.sleep(1)
            self.set_target_pos()
            self.clear_costmap()
            self.collect_flag = True
            # set sequence~~~~~

    def set_target_pos(self):
        # model_state = ModelState()
        # model_state = ModelState()
        state = self.get_state()

        random_index = random.randint(0, len(self.pos_candidate) - 1)
        x, y = self.pos_candidate[random_index]

        dx = state.position.x - x
        dy = state.position.y - y
        dist = math.sqrt(dx*dx + dy*dy)

        while dist < 5.0:
            random_index = random.randint(0, len(self.pos_candidate) - 1)
            x, y = self.pos_candidate[random_index]

            dx = state.position.x - x
            dy = state.position.y - y
            dist = math.sqrt(dx*dx + dy*dy)
        
        self.ps_map.pose.position.x = x
        self.ps_map.pose.position.y = y
        self.ps_map.pose.orientation.w = 1.0

        self.clear_costmap()
        self.target_pos_pub.publish(self.ps_map)

        # print(dist)        

    def read_csv(self):
        with open(self.read_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y = map(float, row)
                self.pos_candidate.append((x, y))
        # print(len(self.pos_candidate))

    def reset_env(self):
        self.reset_world()

    def clear_costmap(self):
        self.clear_costmap()

    def get_state(self):
        resp = self.get_pos(MODEL_NAME, '')
        get_state = Pose()
        get_state = resp.pose
        return get_state

    def set_ang(self):
        model_state = ModelState()
        random_ang = random.uniform(-3.14, 3.14)
        # print(random_ang)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, random_ang)
        state = self.get_state()
        state.orientation.x = quaternion[0]   
        state.orientation.y = quaternion[1]
        state.orientation.z = quaternion[2]
        state.orientation.w = quaternion[3]
        model_state.model_name = MODEL_NAME
        model_state.pose = state
        self.set_pos(model_state)
    
    def set_pos_and_ang(self):
        model_state = ModelState()
        random_ang = random.uniform(-3.14, 3.14)
        # print(random_ang)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, random_ang)
        # state = self.get_state()

        random_index = random.randint(0, len(self.pos_candidate) - 1)
        x, y = self.pos_candidate[random_index]

        # state = model_state.pose

        self.ps_estimate.pose.position.x = x
        self.ps_estimate.pose.position.y = y
        self.ps_estimate.pose.orientation.x = quaternion[0]   
        self.ps_estimate.pose.orientation.y = quaternion[1]
        self.ps_estimate.pose.orientation.z = quaternion[2]
        self.ps_estimate.pose.orientation.w = quaternion[3]
        model_state.model_name = MODEL_NAME

        ps_base = self.listener.transformPose('/odom', self.ps_estimate)
        model_state.pose = ps_base.pose
        # print(ps_base)
        # print(model_state)

        # print(self.ps_estimate.pose)

        # model_state.pose = state
        self.set_pos(model_state)
        self.cmd_pub.publish(self.cmd_vel)
        time.sleep(0.2)
        self.pose_estimate(self.ps_estimate.pose)

    def reset_and_set(self):
        self.reset_env()
        self.set_ang()

if __name__ == '__main__':
    rospy.init_node('olp_cloning_node', anonymous=True)
    # rg = olp_cloning_node()
    # DURATION = 0.2
    # r = rospy.Rate(1 / DURATION)
    # while not rospy.is_shutdown():
    #     rg.loop()
    #     r.sleep()

    # rs = olp_cloning_node()
    rg = Train_env()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
    # rg.reset_and_set()
    # rg.clear_costmap()
    # rg.read_csv()
    # rg.set_pos_and_ang()
    # rg.set_target_pos()
    # while True:
    #     rg.set_pos_and_ang()
    # rg.set_target_pos()
    # rg.set_target_pos()