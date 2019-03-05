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
from std_msgs.msg import Float32, Bool
import message_filters
import json


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
    if angle == -180 or angle == 180:
        angle = 0
    return float(-angle)


joy = xbox.Joystick()
reverse = False
joy_start_time = 0.0
joy_record = []
emergency_brake = True
proximity_sensor = True

emergency_brake = True


def joy_stick_controller(index):
    global joy, reverse, emergency_brake
    speed = angle = 0.0
    x, y = joy.leftStick()
    angle = convert_to_angle(x, y)
    print("Proximity:", proximity_sensor)
    print('Hand brake:', emergency_brake)
    if joy.B() == 1:
        emergency_brake = True if emergency_brake == False else False

    if emergency_brake or proximity_sensor == False:
        angle = 0
        speed = 0
        car_control(angle=0, speed=0)

    else:
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
            speed = 10
            car_control(angle=angle, speed=speed)
        else:
            speed = 5
            car_control(angle=angle, speed=speed)
    
    rgb_img_path = os.path.join(rgb_path, '{}_rgb.jpg'.format(index))
    depth_img_path = os.path.join(depth_path, '{}_depth.jpg'.format(index))
    joy_record.append({
        'index': index,
        'rgb_img_path': rgb_img_path,
        'depth_img_path': depth_img_path,
        'angle': angle,
        'speed': speed,
        'proximity value': proximity_sensor
    })
    return rgb_img_path, depth_img_path

rgb_index = 0
depth_index = 0
output_path = './dataset_{}/'.format(time.time())
rgb_path = os.path.join(output_path, 'rgb')
depth_path = os.path.join(output_path, 'depth')


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
    rgb_img_path, depth_img_path = joy_stick_controller(rgb_index)
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    temp = np.fromstring(depth_data.data, np.uint8)
    depth_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    cv2.imwrite(rgb_img_path, rgb_img)
    print('Wrote', rgb_index, 'RGB frame out.')
    cv2.imwrite(depth_img_path, depth_img)
    print('Wrote', depth_index, 'Depth frame out.')
    print('============================================')


def proximity_callback(proximity_data):
    global proximity_sensor
    proximity_sensor = proximity_data.data


def main():
    rospy.init_node('team705_node', anonymous=True)
    rgb_sub = message_filters.Subscriber(
        '/camera/rgb/image_raw/compressed', CompressedImage, buff_size=2**32)
    depth_sub = message_filters.Subscriber(
        '/camera/depth/image_raw/compressed/', CompressedImage, buff_size=2**32)
    proximity_sub = rospy.Subscriber(
        '/ss_status', Bool, proximity_callback)
    ts = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub], queue_size=1, slop=0.1)
    ts.registerCallback(image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    with open(os.path.join(output_path, 'label.json'), 'w', encoding='utf-8') as outfile:
        json.dump(joy_record, outfile, ensure_ascii=False,
                  sort_keys=False, indent=4)
        outfile.write("\n")


main()
