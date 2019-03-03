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


def car_control(angle, speed):
    '''
    Hàm này dùng để gửi tín hiệu đến simulator
    '''
    pub_speed = rospy.Publisher('/set_speed_car_api', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle = rospy.Publisher('/set_steer_car_api', Float32, queue_size=10)
    pub_angle.publish(-angle)
    print('Angle:', -angle, 'Speed:', speed)


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
    if angle==-180 or angle==180:
        angle = 0
    return angle


joy = xbox.Joystick()
reverse = False
emergency_brake = True


def joy_stick_controller():
    global joy, reverse, emergency_brake
    speed = angle = 0.0
    x, y = joy.leftStick()
    angle = convert_to_angle(x, y)
    if joy.B():
        emergency_brake = True if emergency_brake == False else False

    if emergency_brake:
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



def main():
    rospy.init_node('team705_node', anonymous=True)
    try:
        while True:
            joy_stick_controller()
    except Exception as ex:
        if ex == KeyboardInterrupt:
            car_control(angle=0, speed=0)
            pass
        else:
            print(ex)


main()
