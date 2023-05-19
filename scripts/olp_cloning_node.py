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
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseActionResult

# from torch.utils.tensorboard import SummaryWriter

# from olp_cloning_net_without_action_scale import *
import rospy
import roslib
roslib.load_manifest('olp_cloning')

MODEL_NAME = 'turtlebot3_burger'

# to do random target position 
X = 1.5
Y = 1.5

class olp_cloning_node:
    def __init__(self):
        rospy.init_node('olp_cloning_node', anonymous=True)
        self.target_pos_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.target_callback)
        self.target_pos_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.ps_map = PoseStamped()
        self.ps_map.header.frame_id = 'map'
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.scan = LaserScan()
        self.listener = tf.TransformListener()
        self.wait_scan = False
        self.min_s = 0
        self.target_l = 0

        # self.writer = SummaryWriter(log_dir='/home/fmasa/catkin_ws/src/olp_cloning/runs')

        self.gazebo_env = Gazebo_env()

        self.end_steps_flag = False
        self.eval_flag = False

        self.i_episode = 1
        self.eval_count = 0
        self.train = Trains(self.args, self.reward_args)
        self.make_state = MakeState()

        self.gazebo_env.reset_env()
        self.gazebo_env.reset_env()

        self.hand_set_target(X, Y)
        # print(self.ps_map)
        time.sleep(2)

    def target_callback(self, msg):
        self.ps_map.pose = msg.pose

    def scan_callback(self, msg):
        self.scan.ranges = msg.ranges
        scan_ranges = np.array(self.scan.ranges)
        scan_ranges[np.isinf(scan_ranges)] = msg.range_max
        # print(len(scan_ranges))
        self.scan.ranges = scan_ranges
        self.min_s = min(self.scan.ranges)
        self.wait_scan = True

    def hand_set_target(self, x, y):
        self.ps_map.pose.position.x = x
        self.ps_map.pose.position.y = y
    
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

    def next_episode(self):
        # twist reset
        self.train.action_twist.linear.x = 0
        self.train.action_twist.angular.z = 0
        self.train.action_pub.publish(self.train.action_twist)
        # target set
        self.hand_set_target(X, Y)
        # gazebo reset
        self.gazebo_env.reset_and_set()
        time.sleep(1)
    
    def loop(self):
        if not self.wait_scan:
            return
        self.target_pos_pub.publish(self.ps_map)
        target_l = self.tf_target2robot()
        state = self.make_state.toState(np.array(list(self.scan.ranges)), target_l)

        # #check steps
        # self.end_steps_flag = self.train.check_steps(target_l, self.min_s)

        if self.i_episode == self.args['epochs'] + 1:
            # loop end
            pass

        if not self.eval_flag:
            if self.end_steps_flag:

                episode_avg_reward = self.train.episode_reward / self.train.n_steps
                print(episode_avg_reward, self.i_episode)
                # self.writer.add_scalar("episode per reward", episode_avg_reward, self.i_episode)

                self.train.episode_reward_list.append(self.train.episode_reward)
                if self.i_episode % self.args['eval_interval'] == 0:
                    self.eval_flag = True
                    self.eval_count = 0
                
                # env reset
                # self.gazebo_env.reset_and_set()
                self.next_episode()
                self.train.n_steps = 0
                self.train.episode_reward = 0
                self.train.reward = 0
                self.i_episode += 1
                self.train.init = True

                self.end_steps_flag = False
            else:
                self.train.while_func(state, target_l, self.min_s)
                self.writer.add_scalar("reward", self.train.reward, self.train.total_n_steps)
                print("Train mode,  Episode: {}, Total step: {}, Step: {}, Step reward: {:.2f}, target_d: {:.2f}, target_r: {:.2f}, robot_ang: {:.2f}".format(self.i_episode, self.train.total_n_steps, self.train.n_steps, self.train.reward, self.target_l[0], self.target_l[1], self.target_l[2]))
        else:
            if self.eval_count == self.args['eval_interval']:
                self.eval_flag = False
                print("Episode: {}, Eval Avg. Reward: {:.0f}".format(self.i_episode, self.train.avg_reward))
            else:
                if self.end_steps_flag:
                    self.train.avg_reward /= self.args['eval_interval']
                    self.train.eval_reward_list.append(self.train.avg_reward)

                    # print("Episode: {}, Eval Avg. Reward: {:.0f}".format(self.i_episode, self.train.avg_reward))

                    # env reset
                    # self.gazebo_env.reset_and_set()
                    self.next_episode()
                    self.eval_count += 1
                    self.train.n_steps = 0
                    self.train.episode_reward = 0
                    self.train.reward = 0
                    self.train.init = True

                    self.end_steps_flag = False
                else:
                    # eval func
                    self.train.evaluate_func(state, target_l, self.min_s)
                    print("Evaluate mode Episode: {}, Step: {}, Step reward: {:.0f}".format(self.i_episode, self.train.n_steps, self.train.episode_reward))

        # print("Episode: {}, Step: {}, Step reward: {:.0f}".format(self.i_episode, self.train.n_steps, self.train.episode_reward))
        #check steps
        self.end_steps_flag = self.train.check_steps(target_l, self.min_s)

class Trains:
    def __init__(self, args, reward_args):
        self.args = args
        self.reward_args = reward_args
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        # print(device)
        self.agent = ActorCriticModel(args=args, device=device)
        self.memory = ReplayMemory(args['memory_size'])
        self.action_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.action_twist = Twist()

        self.episode_reward_list = []
        self.eval_reward_list = []

        self.n_steps = 0
        self.total_n_steps = 0
        self.n_update = 0

        self.episode_reward = 0
        self.done = False
        # self.init = True
        self.action = [0.0, 0.0]
        self.init = True
        self.avg_reward = 0.
        self.reward = 0
        # self.state = 1 # robotに対する目標位置

        self.old_action = [0., 0.]
        self.old_state = 0
        self.old_target_l = 0
    
    def while_func(self, state, target_l, min_s):
        if self.args['start_steps'] > self.n_steps:
            # action = env.action_space.sample()
            self.action[0] = random.uniform(0.1, 0.5)
            self.action[1] = random.uniform(-1.0, 1.0)
        else:
            self.action = self.agent.select_action(self.old_state)

        # print(self.action)

        if len(self.memory) > self.args['batch_size']:
            self.agent.update_parameters(self.memory, self.args['batch_size'], self.n_update)
            self.n_update += 1

        # env update
        # if self.action[0] < 0:
        #     self.action[0] = 0
        self.action_twist.linear.x = self.action[0]
        self.action_twist.angular.z = self.action[1]
        self.action_pub.publish(self.action_twist)

        if not self.init:
            # next_state, reward, done, _ = env.step(action)

            self.reward = self.calc_reward(target_l, min_s, self.old_target_l)

            self.n_steps += 1
            self.total_n_steps += 1
            self.episode_reward += self.reward

            self.memory.push(state=self.old_state, action=self.old_action, reward=self.reward, next_state=state, mask=float(not self.done))

        # state = next_state
        self.old_state = state
        self.old_action = self.action
        self.old_target_l = target_l

        self.init = False

    def evaluate_func(self, state, target_l, min_s):
        with torch.no_grad():
            self.action = self.agent.select_action(state, evaluate=True)

        self.action_twist.linear.x = self.action[0]
        self.action_twist.angular.z = self.action[1]
        self.action_pub.publish(self.action_twist)

        if not self.init:
            self.reawrd = self.calc_reward(target_l, min_s, self.old_target_l)
            self.n_steps += 1
            self.total_n_steps += 1
            self.episode_reward += self.reawrd

    def calc_reward(self, target_l, min_s, old_target_l):
        # rt
        if target_l[0] < self.reward_args['Cd']:
            rt = self.reward_args['r_arrive']
        elif min_s < self.reward_args['Co'] + 0.02:
            rt = self.reward_args['r_collision']
        else:
            # rt = self.reward_args['Cr'] * (old_target_l[0] - target_l[0])
            rt = self.reward_args['Cr'] * (old_target_l[0] - target_l[0])
        
        # rpt
        if (old_target_l[0] - target_l[0]) == 0:
            rpt = self.reward_args['r_position']
        else:
            # 0.1m/sで近づかないとペナルティ
            # rpt =  - self.reward_args['Cr'] * (0.1 / 5)
            # rpt =  - (0.1 / 5)
            rpt = 0

        # rwt
        rwt = -1 * abs(target_l[1])

        # self.reward = rt + self.reward_args['ramda_p'] * rpt + self.reward_args['ramda_w'] * rwt
        self.reward = rt + self.reward_args['ramda_r'] * abs(old_target_l[1] - old_target_l[1])
        # self.reward = rt
        # self.reward = rt + self.action[0]

        return self.reward

    def check_steps(self, target_l, min_s):
        flag = False
        if self.args['end_step'] == self.n_steps:
            flag = True
        elif target_l[0] < self.reward_args['Cd']:
            flag = True
        elif min_s < self.reward_args['Co']:
            flag = True
        return flag

class Train_env:
    def __init__(self):
        # rospy.wait_for_service('/gazebo/rest_world')
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.clear_costmap = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        self.set_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_pos = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.target_pos_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.initial_pos = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.result_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.result_callback)
        

        self.pos_candidate = []
        self.read_path = '/home/fmasa/catkin_ws/src/olp_cloning/config/set_pose.csv'

        self.ps_map = PoseStamped()
        self.ps_map.header.frame_id = 'map'

        self.init_pos = PoseWithCovarianceStamped()
        self.init_pos.header.frame_id = 'map'

        self.scan = LaserScan()
        self.min_s = 0
        self.is_goal_reached = False
        self.wait_scan = False

        self.read_csv()

        # self.set_pos_and_ang()
        # self.set_pos_and_ang()

    def loop(self):
        if not self.wait_scan:
            return
        self.check_end()

    def result_callback(self, msg):
        self.is_goal_reached = True

    def scan_callback(self, msg):
        self.scan.ranges = msg.ranges
        scan_ranges = np.array(self.scan.ranges)
        scan_ranges[np.isinf(scan_ranges)] = msg.range_max
        # print(len(scan_ranges))
        self.scan.ranges = scan_ranges
        self.min_s = min(self.scan.ranges)
        self.wait_scan = True
    
    def pose_estimate(self, state):
        # self.init_pos.pose.pose.position.x = x
        # self.init_pos.pose.pose.position.y = y
        # self.init_pos.pose.pose.orientation.z = theta
        self.init_pos.pose.pose = state
        self.initial_pos.publish(self.init_pos)

    def check_end(self):
        flag = False
        if self.is_goal_reached:
            flag = True
        elif self.min_s < 0.12:
            flag = True
        
        if flag:
            self.is_goal_reached = False
            rg.set_pos_and_ang()
            rg.set_target_pos()
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

        state = model_state.pose

        state.position.x = x
        state.position.y = y
        state.orientation.x = quaternion[0]   
        state.orientation.y = quaternion[1]
        state.orientation.z = quaternion[2]
        state.orientation.w = quaternion[3]
        model_state.model_name = MODEL_NAME
        # model_state.pose = state
        self.set_pos(model_state)
        self.pose_estimate(state)

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