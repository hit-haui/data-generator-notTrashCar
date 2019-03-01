#!/usr/bin/python3
import os
import sys
import time
import math
import xbox

import cv2
import numpy as np

import rospkg
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import message_filters
import json

try:
    os.chdir(os.path.dirname(__file__))
    os.system('clear')
    print("\nWait for initial setup, please don't connect anything yet...\n")
    sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except:
    pass

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# rgb_video_out_name = 'rgb_output_{}.avi'.format(time.time())
# rgb_video_out = cv2.VideoWriter(rgb_video_out_name, fourcc, 60, (320, 240))

# depth_video_out_name = 'depth_output_{}.avi'.format(time.time())
# depth_video_out = cv2.VideoWriter(depth_video_out_name, fourcc, 60, (320, 240))


def car_control(angle, speed):
    '''
    Hàm này dùng để gửi tín hiệu đến simulator
    '''
    pub_speed = rospy.Publisher('/set_speed_car_api', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle = rospy.Publisher('/set_steer_car_api', Float32, queue_size=10)
    pub_angle.publish(angle)
    print('Angle:', angle, 'Speed:', speed)


def convert_to_angle(x, y):
    angle = 0.0
    if x == 0 and y == 0 or x == 0 and y > 0:
        angle = 0.0
    if x == 0.0 and y < 0.0:
        angle = 180.0
    if y == 0.0 and x > 0:
        angle = 90.0
    if y == 0.0 and x < 0:
        angle = -90.0
    elif x > 0.0 and y > 0.0:
        angle = math.degrees(math.atan(x/y))
    elif x > 0.0 and y < 0.0:
        angle = math.degrees(math.atan(x/y)) + 180.0
    elif x < 0.0 and y < 0.0:
        angle = math.degrees(math.atan(x/y)) - 180.0
    elif x < 0.0 and y > 0.0:
        angle = math.degrees(math.atan(x/y))
    return angle


joy = xbox.Joystick()
reverse = False
joy_start_time = 0.0
joy_record = []


def joy_stick_controller(index):
    global joy, reverse
    speed = angle = 0.0
    x, y = joy.leftStick()
    angle = convert_to_angle(x, y)
    if joy.X() == 1:
        reverse = True if reverse == False else False
    if reverse:
        if angle > 60 and angle <= 120:
            angle = 60
        elif angle >= -120 and angle < -60:
            angle = -60
    else:
        if angle > 60 and angle <= 179.9:
            angle = 60
        elif angle >= -179.9 and angle < -60:
            angle = -60
    if joy.Y() == 0:
        speed = 100
        car_control(angle=angle, speed=speed)
    else:
        speed = 50
        car_control(angle=angle, speed=speed)
    if joy.A():
        angle = 0
        speed = 0
        car_control(angle=angle, speed=speed)
    joy_record.append({
        'index': index,
        'angle': angle,
        'speed': speed,
    })


rgb_index = 0
depth_index = 0
rgb_path = './recored_data/rgb/'
depth_path = './recored_data/depth/'
try:
    os.makedirs(rgb_path)
    os.makedirs(depth_path)
except:
    pass

def image_callback(rgb_data, depth_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    global rgb_index, depth_index
    rgb_index += 1
    depth_index += 1
    joy_stick_controller(rgb_index)
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    temp = np.fromstring(depth_data.data, np.uint8)
    depth_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    cv2.imwrite(rgb_path + '{}_rgb.jpg'.format(rgb_index), rgb_img)
    print('Wrote', rgb_index, 'RGB frame out.')
    cv2.imwrite(depth_path + '{}_rgb.jpg'.format(depth_index), depth_img)
    print('Wrote', depth_index, 'Depth frame out.')
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
    # rgb_video_out.release()
    # depth_video_out.release()
    # print('Saved 2 videos')
    with open('./recored_data/key_data.json', 'w', encoding='utf-8') as outfile:
        json.dump(joy_record, outfile, ensure_ascii=False,
                  sort_keys=False, indent=4)
        outfile.write("\n")


main()
