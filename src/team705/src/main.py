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


def rgb_callback(rgb_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    start_time = time.time()
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    # cv2.imshow('rgb_frame', rgb_img)
    cv2.waitKey(1)
    rgb_video_out.write(rgb_img)
    print('RGB Shape:',rgb_img.shape)
    print('FPS_RGB:', 1/(time.time() - start_time))


def depth_callback(depth_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    start_time = time.time()
    temp = np.fromstring(depth_data.data, np.uint8)
    depth_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    # cv2.imshow('depth_frame', depth_img)
    cv2.waitKey(1)
    depth_video_out.write(depth_img)
    print('Depth Shape:', depth_img.shape)
    print('FPS_DEPTH:', 1/(time.time() - start_time))


def main():
    rospy.init_node('team705_node', anonymous=True)
    rgb_sub = rospy.Subscriber(
        '/camera/rgb/image_raw/compressed/', CompressedImage, rgb_callback, queue_size=1, buff_size=2**24)
    depth_sub = rospy.Subscriber(
        '/camera/depth/image_raw/compressed/', CompressedImage, depth_callback, queue_size=1, buff_size=2**24)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    rgb_video_out.release()
    depth_video_out.release()
    print('Saved 2 videos')


if __name__ == '__main__':
    main()
