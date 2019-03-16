#!/usr/bin/python3
import os
import sys
import time
import math

import numpy as np

import rospkg
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Joy

from std_msgs.msg import Float32, Bool, String
import message_filters
import json
sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
import cv2

def car_control(angle, speed):
    '''
    Hàm này dùng để gửi tín hiệu đến simulator
    '''
    pub_speed = rospy.Publisher('/set_speed_car_api', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle = rospy.Publisher('/set_steer_car_api', Float32, queue_size=10)
    pub_angle.publish(angle)
    print('Angle:', angle, 'Speed:', speed)

def lcd_print(s):
    lcd = rospy.Publisher('/lcd_print', String , queue_size=10)
    lcd.publish(s)


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


reverse = False
joy_start_time = 0.0
joy_record = []
bt1_sensor = False
emergency_brake = True
proximity_sensor = True
emergency_brake = True
start_gendata = False
button_status = 0
# (x,y): Left joystick, left_t: Break button, left_b: Emergency break, right_b: reverse button
x = y = 0.0
left_b = right_t = right_b = y_button = a_button = start_button = 0
default_speed = change_speed = 8
max_speed = 25
min_speed = 8


def joy_stick_controller(index):
    global reverse, emergency_brake, change_speed,button_status
    speed = angle = 0.0
    angle = convert_to_angle(x, y)
    print('Angle before convert:', angle)
    print("Proximity:", proximity_sensor)
    print('Hand brake:', emergency_brake)
    print('Reverse:', reverse)
    if reverse == False:
        lcd_print('1:2: >>>>FORWARD<<<<')
        print('BUTTON:', bt1_sensor)
    else:
        lcd_print('1:2: >>>>REVERSE<<<<')

    if left_b == 1:
        emergency_brake = True if emergency_brake == False else False
    if bt1_sensor == True and button_status == 0 :
        button_status = 1
    elif bt1_sensor == True and button_status == 1 :
        button_status=0
    if emergency_brake or proximity_sensor == False or button_status==1 :
        angle = 0
        speed = 0
        car_control(angle=0, speed=0)
        lcd_print('1:2:STOPED')
    else:
        if right_b == 1:
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

        if y_button == 1:
            change_speed += 5
            if change_speed >= max_speed:
                change_speed = max_speed

        if a_button == 1:
            change_speed -= 5
            if change_speed < min_speed:
                change_speed = min_speed

        if right_t == 1:
            if reverse == True:
                car_control(angle=angle, speed=-90)
            else:
                car_control(angle=angle, speed=change_speed)
                lcd_print('1:2: FORWARD')

        else:
            car_control(angle=angle, speed=0)
        lcd_print('1:2: ') 

    if index != 0:
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
output_path = rgb_path = depth_path = ''


def image_callback(rgb_data, depth_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    global rgb_index, depth_index, output_path, rgb_path, depth_path
    print('start button: ', start_button)
    print('start gen data: ', start_gendata)
    if start_dump == False:
        with open(os.path.join(output_path, 'label.json'), 'w', encoding='utf-8') as outfile:
            json.dump(joy_record, outfile, ensure_ascii=False,
                      sort_keys=False, indent=4)
            outfile.write("\n")
        joy_record.clear()
        rgb_index = depth_index = 0
        output_path = ''
        rgb_path = ''
        depth_path = ''
        joy_stick_controller(0)
        print("Do not write img")
        print('----------------')
    else:
        rgb_index += 1
        depth_index += 1
        if rgb_index == 1 and depth_index == 1:
            output_path = '/home/nvidia/Desktop/dataset_{}/'.format(
                time.time())
            rgb_path = os.path.join(output_path, 'rgb')
            depth_path = os.path.join(output_path, 'depth')
            try:
                os.makedirs(rgb_path)
                os.makedirs(depth_path)
            except:
                pass
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

def bt1_callback(bt1_data):
    global bt1_sensor
    bt1_sensor = bt1_data.data

def proximity_callback(proximity_data):
    global proximity_sensor
    proximity_sensor = proximity_data.data


def joy_callback(joy_data):
    global x, y, left_b, right_b, right_t, y_button, a_button
    global start_button, start_dump

    for index in range(len(joy_data.axes)):
        x = -(joy_data.axes[0])
        y = joy_data.axes[1]
        right_t = -(joy_data.axes[5])
    for index in range(len(joy_data.buttons)):
        a_button = joy_data.buttons[0]
        y_button = joy_data.buttons[3]
        left_b = joy_data.buttons[4]
        right_b = joy_data.buttons[5]
        start_button = joy_data.buttons[7]
    if start_button == 1:
        start_gendata = True if start_gendata == False else False

def main():
    rospy.init_node('team705_node', anonymous=True)
    rgb_sub = message_filters.Subscriber(
        '/camera/rgb/image_raw/compressed', CompressedImage, buff_size=2**16)
    depth_sub = message_filters.Subscriber(
        '/camera/depth/image_raw/compressed/', CompressedImage, buff_size=2**16)
    proximity_sub = rospy.Subscriber(
        '/ss_status', Bool, proximity_callback)
    bt1_sub = rospy.Subscriber(
        '/bt1_status', Bool, bt1_callback)
    
    joy_sub = rospy.Subscriber("joy", Joy, joy_callback)
    ts = message_filters.ApproximateTimeSynchronizer(
        [rgb_sub, depth_sub], queue_size=1, slop=0.1)
    ts.registerCallback(image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        car_control(angle=0, speed=0)
        pass


main()
