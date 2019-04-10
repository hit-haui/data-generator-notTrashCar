#!/usr/bin/python3
import math
import os
import sys
import time
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Bool, String
from param import *
from lane_detect import *
from predict_traffic_sign import *
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.backend.tensorflow_backend import set_session
import math

#left, center, right
traffic_status_list = [0,0,0]
flow_lane = 0


def car_control(angle, speed):
    '''
    Hàm này dùng để gửi tín hiệu đến simulator
    '''
    pub_speed = rospy.Publisher('/set_speed_car_api', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle = rospy.Publisher('/set_steer_car_api', Float32, queue_size=10)
    pub_angle.publish(angle)
    print('Angle:', angle, 'Speed:', speed)

def process_frame(raw_img):
    global traffic_status_list,flow_lane

    traffic = 0

    traffic_status = predict_traffic(raw_img)
    if traffic_status == 'Left':
        traffic_status_list[0] = traffic_status_list[0] +1
    elif traffic_status == 'Right':
        traffic_status_list[2] = traffic_status_list[2] +1
    elif traffic_status == 'No traffic':
        traffic_status_list[1] = traffic_status_list[1] +1
    
    if traffic_status_list[1] == no_traffic_size_count:
        traffic_status_list = [0,0,0]

    if traffic_status_list[0] >= 1:
        flow_lane = -1
        traffic = -1
    elif traffic_status_list[2] >=1:
        traffic = 1
        flow_lane = 1

    # Crop from sky line down
    raw_img = raw_img[sky_line:, :]
    # Hide sensor and car's hood
    # raw_img = cv2.rectangle(raw_img, top_left_proximity,
    #                         bottom_right_proximity, hood_fill_color, -1)
    # raw_img = cv2.rectangle(raw_img, top_left_hood,
    #                         bottom_right_hood, hood_fill_color, -1)
    # cv2.imshow('raw', raw_img)


    # Simple color filltering + Canny Edge detection
    combined, combined_gray = detect_gray(raw_img)

    # Handle shadow by using complex sobel operator
    
    # combined = get_combined_binary_thresholded_img(
    #     cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)) * 25


    print(traffic)
    if traffic == -1:
        flow_lane = -1
        combined = combined[:combined.shape[0], :combined.shape[1]//2]
        combined[:combined.shape[0], combined.shape[1]-20:combined.shape[1]-5] = 255
    elif traffic == 1:
        flow_lane = 1
        combined = combined[:combined.shape[0], combined.shape[1]//2:combined.shape[1]]
        combined[:combined.shape[0], 5:20] = 255

    cv2.imshow('combined',combined)
    # combined = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 51, 2)
    # combined = cv2.bitwise_not(combined)



    # Line detection here
    line_image, angle = hough_lines(combined, rho, theta,
                                    threshold, min_line_length, max_line_gap)

    # Hanlde turn ?
    test_img = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
    annotated_image = cv2.cvtColor(weighted_img(weighted_img(line_image, test_img), cv2.COLOR_RGB2BGR))
    return annotated_image, angle
    

def image_callback(rgb_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    print('call back')
    start_time = time.time()
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    # rgb_img = cv2.resize(rgb_img, (480, 640))
    # print(rgb_img.shape)
    #annotated_image, angle = process_frame(rgb_img)
    angle = detect_angle_lane_right(rgb_img)
    #cv2.imshow('processed_frame', annotated_image)
    #cv2.waitKey(1)
    car_control(angle=angle, speed= 80)
    # rgb_img = cv2.resize(rgb_img, img_size[:-1])
    print("FPS:", 1/(time.time()-start_time))
    print('-----------------------------------')


def main():
    rospy.init_node('team705_node', anonymous=True)
    rospy.Subscriber(    # Printing array dimensions (axes)
        '/camera/rgb/image_raw/compressed/', CompressedImage, buff_size=2**32, queue_size=1, callback=image_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


main()
