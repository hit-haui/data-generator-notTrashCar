#!/usr/bin/python3
import os
import sys
import time

import cv2
import numpy as np

import message_filters
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

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_out_name = 'output_{}.avi'.format(time.time())
# video_out = cv2.VideoWriter(video_out_name, fourcc, 20, (320, 240))


def car_control(angle, speed):
    '''
    Hàm này dùng để gửi tín hiệu đến simulator
    '''
    pub_speed = rospy.Publisher('/set_speed_car_api', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle = rospy.Publisher('/set_steer_car_api', Float32, queue_size=10)
    pub_angle.publish(angle)
    print('Angle:', angle, 'Speed:', speed)


def image_callback(rgb_data, depth_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    start_time = time.time()
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    temp = np.fromstring(depth_data.data, np.uint8)
    depth_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    cv2.imshow('rgb_frame', rgb_img)
    cv2.imshow('depth_frame', depth_img)
    cv2.waitKey(1)
    # video_out.write(img)
    print('FPS:', 1/(time.time() - start_time))


def main():
    rospy.init_node('team705_node', anonymous=True)
    rgb_sub = message_filters.Subscriber(
        '/camera/rgb/image_raw', CompressedImage, buff_size=2**24)
    depth_sub = message_filters.Subscriber(
        '/camera/depth/image_raw', CompressedImage, buff_size=2**24)
    ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub], queue_size=1)
    ts.registerCallback(image_callback)
    rospy.spin()
    # video_out.release()
    # print('Saved', video_out_name)


if __name__ == '__main__':
    main()
