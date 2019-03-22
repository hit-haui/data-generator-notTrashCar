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

graph = tf.get_default_graph()
model_traffic = load_model('/home/nvidia/detect2_traffic-016-0.98212.hdf5')
model_cnn = load_model('/home/nvidia/read_data_2chanel-054-524.97807.hdf5')


print('Loaded model')

try:
    os.chdir(os.path.dirname(__file__))
    os.system('clear')
    print("\nWait for initial setup, please don't connect anything yet...\n")
    sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except:
   pass

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



def get_predict(img):
    s = img.shape
    img = img[:s[0]//2, :]
    output = img.copy()
    raw = output.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 70, 70])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img, img, mask=mask)
    color = res.copy()
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 17, 2)
    # detect circles in the image
    circles = cv2.HoughCircles(
        res, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=50)
    left = 0
    none = 0
    right = 0
    # ensure at least some circles were found
    if circles is not None and np.sum(circles) > 0:
        print('tes')
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # print('Got', len(circles), 'circles')

        # loop over the (x, y) coordinates and radius of the circles
        for index_phu, (x, y, r) in enumerate(circles):

            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 0, 255), 4)
            top_y = max(y - r - 10, 0)
            top_x = max(x - r - 10, 0)
            y_size = min(top_y+r*2+20, img.shape[0])
            x_size = min(top_x+r*2+20, img.shape[1])
            img = img[top_y:y_size, top_x:x_size, :]
            
            h,w,c = img.shape
            if h and w !=0:
                if c != 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                img = cv2.resize(img,(80,80))
            
                img = np.expand_dims(img,axis=-1)
                with graph.as_default():
                    traffic_list = model_traffic.predict(np.array([img]))[0]
                # print('predict:',traffic_list)
                l = traffic_list[0]
                n = traffic_list[1]
                r = traffic_list[2]
                print(l,r,n)
                if max(l,max(n,r)) == traffic_list[0]:
                    left +=1
                elif max(l, max(n, r)) == traffic_list[1]:
                    none +=1
                elif max(l, max(n, r)) == traffic_list[2]:
                    right +=1
    if left > right:
        print('Left traffic')
        return np.array([0,1,0])
    elif left < right:
        print('Right traffic')
        return np.array([0,0,1])
    else:
        print('no traffic')
        return np.array([0, 0, 0])


def predict_angle(img_rgb, img_depth):
    img_list = []
    traffics_list = []
    #predict traffic

    traffic_status = get_predict(img_rgb)
    traffics_list.append(traffic_status)

    h,w,_ = img_rgb.shape
    img_rgb = img_rgb[h//2:h,:w]
    img_rgb = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
    img_depth = cv2.cvtColor(img_depth,cv2.COLOR_RGB2GRAY)
    hd,wd = img_depth.shape
    img_depth = img_depth[hd//2:hd,:wd]
    img_list.append(np.dstack((img_rgb,img_depth)))
    with graph.as_default():
        angle = model_cnn.predict([img_list,traffics_list])[0][0]
    return angle


def image_callback(rgb_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    global rgb_index, depth_index
    start_time = time.time()
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    depth_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    angle = predict_angle(rgb_img,depth_img)
    car_control(angle=angle-60, speed=9)
    print("FPS:",1/(time.time()-start_time))


def proximity_callback(proximity_data):
    global proximity_sensor
    proximity_sensor = proximity_data.data


def main():
    rospy.init_node('team705_node', anonymous=True)
    # rgb_sub = message_filters.Subscriber(
    #    '/camera/rgb/image_raw/compressed/', CompressedImage, buff_size=2**16,queue_size = 1)
    # depth_sub = message_filters.Subscriber(
    #  '/camera/depth/image_raw/compressed/', CompressedImage, buff_size=2**16,queue_size = 1)

    # ts = message_filters.ApproximateTimeSynchronizer(
    #    rgb_sub, queue_size=1, slop=0.1)
    # ts.registerCallback(image_callback)
    rgb_sub = rospy.Subscriber('/camera/rgb/image_raw/compressed/', 
     CompressedImage, image_callback , buff_size=2**16, queue_size=1)
    depth_sub = rospy.Subscriber('/camera/depth/image_raw/compressed/', 
     CompressedImage, image_callback , buff_size=2**16, queue_size=1)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        car_control(angle = 0, speed = 0 )
        pass


main()
