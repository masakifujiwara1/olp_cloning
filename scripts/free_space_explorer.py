#!/usr/bin/env python3
import rospy
import tf
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped
import math
import numpy as np

def callback(msg):
    map_origin = PoseStamped()

    free_space = []

    map_data = msg.data
    map_width = msg.info.width
    cellsize = msg.info.resolution
    
    map_origin.pose = msg.info.origin

    map_data = np.array(map_data)
    map_data = map_data.reshape([map_width, -1])
    # print(map_data.shape)
    # print((map_data.reshape([map_width, -1])).shape)
    index = np.where(map_data == 0)

    quat = msg.info.origin.orientation
    print(quat)
    euler = tf.transformations.euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))
    print(euler)

    for i in range(len(index[0])):
        map_x = msg.info.origin.position.x + msg.info.resolution * index[0][i]
        map_y = msg.info.origin.position.y + msg.info.resolution * index[1][i]
        free_space.append((map_x, map_y))
    print(free_space[0], free_space[5000])
    # index[0][n], index[1][n]
    # print(index[0][0], index[1][0])


if __name__ == '__main__':
    rospy.init_node('goal_converter')
    rospy.Subscriber('/map', OccupancyGrid, callback)
    rospy.spin()