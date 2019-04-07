#!/usr/bin/python3
import cv2
import math
import os
import sys
import time
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
from lane_detect import *
from param import *
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

# model_path = '/home/dejavu/read_data_2chanel-054-524.97807.hdf5'
# img_size = (320, 240, 1)
# model = load_model(model_path)
# graph = tf.get_default_graph()

# print('Loaded model')

# try:
#    os.chdir(os.path.dirname(__file__))
#    os.system('clear')
#   print("\nWait for initial setup, please don't connect anything yet...\n")
try:
    sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except:
   pass


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
    # Crop from sky line down
    print(sky_line)
    raw_img = raw_img[sky_line:, :]
    print(raw_img.shape)
    # Hide sensor and car's hood
    # raw_img = cv2.rectangle(raw_img, top_left_proximity,
    #                         bottom_right_proximity, hood_fill_color, -1)
    # raw_img = cv2.rectangle(raw_img, top_left_hood,
    #                         bottom_right_hood, hood_fill_color, -1)
    cv2.imshow('raw', raw_img)

    # Handle shadow by using complex sobel operator
    combined = get_combined_binary_thresholded_img(
        cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)) * 255
    # combined = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 51, 2)
    # combined = cv2.bitwise_not(combined)

    # Simple color filltering + Canny Edge detection
    # combined = easy_lane_preprocess(raw_img)
    line_image, angle = hough_lines(combined, rho, theta,
                                    threshold, min_line_length, max_line_gap)
    test_img = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
    annotated_image = cv2.cvtColor(weighted_img(
        line_image, test_img), cv2.COLOR_RGB2BGR)
    return annotated_image, angle


def image_callback(rgb_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    start_time = time.time()
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    # rgb_img = cv2.resize(rgb_img, (480, 640))
    # print(rgb_img.shape)
    annotated_image, angle = process_frame(rgb_img)
    cv2.imshow('processed_frame', annotated_image)
    cv2.waitKey(1)
    # car_control(angle=angle, speed=25)
    # rgb_img = cv2.resize(rgb_img, img_size[:-1])
    print("FPS:", 1/(time.time()-start_time))
    print('Angle:', angle)
    print('-----------------------------------')

def main():
    rospy.init_node('team705_node', anonymous=True)
    rospy.Subscriber(
        '/camera/rgb/image_raw/compressed/', CompressedImage, buff_size=2**32, queue_size=1, callback=image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


main()
