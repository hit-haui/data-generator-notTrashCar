#!/usr/bin/python3
import os
import sys
import time

import cv2
import numpy as np

import rospkg
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import message_filters

try:
    os.chdir(os.path.dirname(__file__))
    os.system('clear')
    print("\nWait for initial setup, please don't connect anything yet...\n")
    sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except:
    pass

fourcc = cv2.VideoWriter_fourcc(*'XVID')
rgb_video_out_name = 'rgb_output_{}.avi'.format(time.time())
rgb_video_out = cv2.VideoWriter(rgb_video_out_name, fourcc, 60, (320, 240))

depth_video_out_name = 'depth_output_{}.avi'.format(time.time())
depth_video_out = cv2.VideoWriter(depth_video_out_name, fourcc, 60, (320, 240))


def car_control(angle, speed):
    '''
    Hàm này dùng để gửi tín hiệu đến simulator
    '''
    pub_speed = rospy.Publisher('/set_speed_car_api', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle = rospy.Publisher('/set_steer_car_api', Float32, queue_size=10)
    pub_angle.publish(angle)
    print('Angle:', angle, 'Speed:', speed)


rgb_index = 0
depth_index = 0


def image_callback(rgb_data, depth_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    global rgb_index, depth_index
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    temp = np.fromstring(depth_data.data, np.uint8)
    depth_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    rgb_video_out.write(rgb_img)
    rgb_index += 1
    print('Wrote', rgb_index, 'RGB frame to video.')
    depth_video_out.write(depth_img)
    depth_index += 1
    print('Wrote', depth_index, 'Depth frame to video.')
    print('============================================')


def main():
    rospy.init_node('team705_node', anonymous=True)
    rgb_sub = message_filters.Subscriber(
        '/camera/rgb/image_raw/compressed/', CompressedImage, buff_size=2**32)
    depth_sub = message_filters.Subscriber(
        '/camera/depth/image_raw/compressed/', CompressedImage, buff_size=2**32)

    ts = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub], queue_size=1, slop=0.1)
    ts.registerCallback(image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    rgb_video_out.release()
    depth_video_out.release()
    print('Saved 2 videos')


if __name__ == '__main__':
    main()
