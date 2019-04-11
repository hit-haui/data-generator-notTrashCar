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


try:
    sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except:
   pass

from param import *
from lane_detect import *
import math
import cv2
# from predict_traffic_sign import *

default_speed = 15  #8
x_need_right = 250 #220 #200 #40

#default_speed = 10
#x_need_right = 210 #200 #40

start_run_time = 0.0
count = 0

def car_control(angle, speed):
    '''
    Hàm này dùng để gửi tín hiệu đến simulator
'''
    global count, start_run_time, x_need_right
    if speed != 0.0:
        count += 1
        if count == 1:
            start_run_time = time.time()
        else:
            time_now = time.time() - start_run_time
            if time_now  >= 18:
               print('time now:', time_now >= 17)
               speed = 8
               x_need_right = 220
    else:
       count = 0
       start_run_time = 0.0
       x_need_right = 250
    print('start_run_time:', start_run_time)
    print('x_need_right: ', x_need_right)
    pub_speed = rospy.Publisher('/set_speed_car_api', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle = rospy.Publisher('/set_steer_car_api', Float32, queue_size=10)
    pub_angle.publish(angle)
    print('Angle:', angle, 'Speed:', speed)

def detect_angle_lane_right(img):
    img = img[sky_line:,:]
    res,combined = detect_gray(img)
    no_cut = res
    print(res.shape)
    res = res[:res.shape[0]-lane_right_pixel, lane_right_pixel_height:res.shape[1]]
    x,y,check = find(res)
    print(x,y)
    if check == True:
        cv2.line(res , (x, y), (x, y), (255, 255, 255), 5)
        cv2.line(img , (lane_right_pixel_height+x, y), (lane_right_pixel_height+x, y), (90, 0, 255), 5)
        cv2.line(img , (img.shape[1]//2, y ), (img.shape[1]//2, y), (90, 0, 255), 5)

        x_mid = img.shape[1]//2
        y_mid = img.shape[0]
        cv2.line(img , (x_mid, y_mid ), (x_mid, y_mid ), (90, 0, 255), 5)

        x_need = lane_right_pixel_height+x-x_need_right #x+img.shape[1]//2 #(img.shape[1]//2 + x) //2
        y_need = y

        cv2.line(img , (x_need, y_need ), (x_need, y_need ), (90, 90, 255), 5)

        cv2.line(img , (x_need, y_need ), (x_mid, y_mid ), (90, 90, 255), 5)

        angle = math.degrees(math.atan((x_mid - x_need)/(y_mid-y_need)))
    else :
        angle = 0
    print(angle)
    #cv2.imwrite('/home/nvidia/Desktop/data_visual/no_cut/'+str(time.time())+'.jpg', no_cut)
    #cv2.imwrite('/home/nvidia/Desktop/data_visual/image/'+str(time.time())+'.jpg',img)
    #cv2.imshow('img',img)
    #cv2.waitKey(1)
    return angle

#left, center, right
def process_frame(raw_img):
    # traffic_status = predict_traffic(raw_img)

    # Crop from sky line down
    bottom_raw_img = raw_img[sky_line:, :]
    # above_raw_img = raw_img[:sky_line, :]
    # Hide sensor and car's hood
    # raw_img = cv2.rectangle(raw_img, top_left_proximity,
    #                         bottom_right_proximity, hood_fill_color, -1)
    # raw_img = cv2.rectangle(raw_img, top_left_hood,
    #                         bottom_right_hood, hood_fill_color, -1)
    # cv2.imshow('raw', raw_img)

    # Object detect

    detections = detect(image=raw_img, thresh=0.05)
    confident = {}
    if detections:
        for each_detection in detections:
            # print('{}: {}%'.format(each_detection[0], each_detection[1]*100))
            if each_detection[0] in confident:
                confident[each_detection[0]].append(each_detection[1]*100)
            else:
                confident[each_detection[0]] = [each_detection[1]*100]

            x_center = each_detection[-1][0]
            y_center = each_detection[-1][1]
            width = each_detection[-1][2]
            height = each_detection[-1][3]
            x_top = int(x_center - width/2)
            y_top = int(y_center - height/2)
            x_bot = int(x_top + width)
            y_bot = int(y_top + height)
            cv2.rectangle(raw_img, (x_top, y_top), (x_bot, y_bot), (0, 255, 0), 2)
    cv2.imshow('traffic_sign_detection', raw_img)
    traffic = 0
    mean_confident_left = np.mean(confident['turn_left']) if 'turn_left' in confident.keys() else 0
    mean_confident_right = np.mean(confident['turn_right']) if 'turn_right' in confident.keys() else 0
    if mean_confident_left > mean_confident_right and mean_confident_left != 0:
        print('Turning LEFT: {}%'.format(mean_confident_left))
        traffic = -1
    elif mean_confident_right !=0:
        print('Turning RIGHT: {}%'.format(mean_confident_right))
        traffic = 1

    # Simple color filltering + Canny Edge detection
    combined,combined_gray = detect_gray(bottom_raw_img)


    # Handle shadow by using complex sobel operator
    
    # combined = get_combined_binary_thresholded_img(

    #     cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)) * 255
    # print(traffic)
    # if traffic == -1:
    #     combined = combined[:combined.shape[0], :combined.shape[1]//2]
    #     combined[:combined.shape[0], combined.shape[1]-20:combined.shape[1]-5] = 255
    # elif traffic == 1:
    #     combined = combined[:combined.shape[0], combined.shape[1]//2:combined.shape[1]]
    #     combined[:combined.shape[0], 5:20] = 255

    # cv2.imshow('combined',combined)

    # combined = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 51, 2)
    # combined = cv2.bitwise_not(combined)



    # Line detection here
    line_image, angle = hough_lines(combined, rho, theta,
                                    threshold, min_line_length, max_line_gap)

    # Hanlde turn ?
    test_img = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)

    annotated_image = cv2.cvtColor(weighted_img(
        line_image, test_img), cv2.COLOR_RGB2BGR)
    return annotated_image, angle, raw_img


def lcd_print(s):
    lcd = rospy.Publisher('/lcd_print', String, queue_size=10)
    lcd.publish(s)

proximity_sensor = True
bt1_sensor = bt2_sensor = bt3_sensor = bt4_sensor = False
hand_brake = True
#default_speed = 15
max_speed = 15
max_speed_mode = False


def image_callback(rgb_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    global hand_brake
    print('call back')
    start_time = time.time()
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    # rgb_img = cv2.resize(rgb_img, (480, 640))
    # print(rgb_img.shape)
    
    #annotated_image, angle = process_frame(rgb_img)
    #status = traffic_detect(rgb_img)
    #print(status)
    angle = 0
    #if status == 1:
    #    angle = detect_angle_lane_right(rgb_img)
    #elif status == -1:
    angle = detect_angle_lane_right(rgb_img)
    #angle = detect_angle_lane_right(rgb_img)    
    #cv2.imshow('processed_frame', annotated_image)
    #cv2.waitKey(1)
    if proximity_sensor == False:
        hand_brake = True if hand_brake == False else False
        lcd_print('1:2:                    ')
        while proximity_sensor == False:
            lcd_print('1:2: PROXIMITY')
            car_control(angle = 0, speed = 0)
        lcd_print('1:2:                   ')
    if hand_brake == True:
        car_control(angle=0, speed=0)
    if hand_brake == False and  proximity_sensor == True:
        car_control(angle=angle, speed=default_speed)
    # rgb_img = cv2.resize(rgb_img, img_size[:-1])
    #cv2.waitKey(1)
    print("FPS:", 1/(time.time()-start_time))
    print('Angle:', angle)
    #print('status:', tree_detect(raw_img, rgb_img))

    print('-----------------------------------')

def proximity_callback(proximity_data):
    global proximity_sensor
    proximity_sensor = proximity_data.data

def bt1_callback(bt1_data):
    global bt1_sensor
    bt1_sensor = bt1_data.data

def bt2_callback(bt2_data):
    global bt2_sensor
    bt2_sensor = bt2_data.data

def bt3_callback(bt3_data):
    global bt3_sensor
    bt3_sensor = bt3_data.data

def bt4_callback(bt4_data):
    global bt4_sensor
    bt4_sensor = bt4_data.data


def main():
    rospy.init_node('team705_node', anonymous=True)
    rospy.Subscriber(    # Printing array dimensions (axes)
        '/camera/rgb/image_raw/compressed/', CompressedImage, buff_size=2**32, queue_size=1, callback=image_callback)
    proximity_sub = rospy.Subscriber('/ss_status', Bool, proximity_callback)
    bt1_sub = rospy.Subscriber('/bt1_status', Bool, bt1_callback)
    bt2_sub = rospy.Subscriber('/bt2_status', Bool, bt2_callback)
    bt3_sub = rospy.Subscriber('/bt3_status', Bool, bt3_callback)
    bt4_sub = rospy.Subscriber('/bt4_status', Bool, bt4_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


main()
