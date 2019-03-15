#!/usr/bin/python3
import math
import os
import sys
import time

#import cv2
import message_filters
import numpy as np
import rospkg
import rospy
import tensorflow as tf
from keras.models import load_model
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

model_path = './model/real-dropout-466-202.25459.hdf5'
img_size = (320, 240, 1)
model = load_model(model_path)
graph = tf.get_default_graph()

print('Loaded model')

#try:
#    os.chdir(os.path.dirname(__file__))
#    os.system('clear')
#   print("\nWait for initial setup, please don't connect anything yet...\n")
sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
#except:
#    pass

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


def get_prediction(img):
    global model, graph
    with graph.as_default():
        angle = model.predict(np.array([img]))[0][0] - 60
    return angle


def image_callback(rgb_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    global rgb_index, depth_index
    start_time = time.time()
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.resize(rgb_img, img_size[:-1]) 
    #print(rgb_img.shape)
    rgb_img = np.expand_dims(rgb_img,axis=3)
    car_control(angle=get_prediction(rgb_img), speed=9)
    print("FPS:",1/(time.time()-start_time))

def main():
    rospy.init_node('team705_node', anonymous=True)
#    rgb_sub = message_filters.Subscriber(
#        '/camera/rgb/image_raw/compressed/', CompressedImage, buff_size=2**32)
#   depth_sub = message_filters.Subscriber(
#      '/camera/depth/image_raw/compressed/', CompressedImage, buff_size=2**32)

#    ts = message_filters.ApproximateTimeSynchronizer(
#        rgb_sub, queue_size=1, slop=0.1)
#    ts.registerCallback(image_callback)
    rgb_sub = rospy.Subscriber('/camera/rgb/image_raw/compressed/', 
     CompressedImage, image_callback , buff_size=2**16, queue_size=1)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        car_control(angle = 0, speed = 0 )
        pass


main()
