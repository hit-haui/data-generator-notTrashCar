#!/usr/bin/python3
<<<<<<< HEAD
import cv2
import os
import sys
import time
import math

=======
import math
import os
import sys
import time

#import cv2
import message_filters
>>>>>>> 14387370ccb9a4fc1659b50f9c98ba2f7607c522
import numpy as np
import rospkg
import rospy
import tensorflow as tf
from keras.models import load_model
from sensor_msgs.msg import CompressedImage
<<<<<<< HEAD
from sensor_msgs.msg import Joy

from std_msgs.msg import Float32, Bool, String
import message_filters
import json
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
=======
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
>>>>>>> 14387370ccb9a4fc1659b50f9c98ba2f7607c522

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


<<<<<<< HEAD
def lcd_print(s):
    lcd = rospy.Publisher('/lcd_print', String, queue_size=10)
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
proximity_sensor = True
hand_brake = True
start_gendata = False
button_status = 0
# (x,y): Left joystick, left_t: Break button, left_b: Emergency break, right_b: reverse button
x = y = 0.0
left_b = right_t = right_b = y_button = a_button = start_button = 0
default_speed = change_speed = 8
max_speed = 25
min_speed = 8
start_gendata = False


def joy_stick_controller(index):
    global reverse, hand_brake, change_speed, button_status, proximity_sensor
    speed = angle = 0.0
    angle = convert_to_angle(x, y)
    print("Proximity:", proximity_sensor)
    print('Hand brake:', hand_brake)
    print('Reverse:', reverse)
    if reverse == False:
        lcd_print('1:2: ')
        lcd_print('1:2: >>>>FORWARD<<<<')
        print('BUTTON:', bt1_sensor)
    else:
        lcd_print('1:2: ')
        lcd_print('1:2: >>>>REVERSE<<<<')

    if left_b == 1:
        hand_brake = True if hand_brake == False else False
    if bt1_sensor == True and button_status == 0:
        button_status = 1
    elif bt1_sensor == True and button_status == 1:
        button_status = 0
    if hand_brake or proximity_sensor == False or button_status == 1:
        angle = 0
        speed = 0
        car_control(angle=0, speed=0)
        lcd_print('1:2: ')
        if hand_brake:
            lcd_print('1:2: HAND BRAKE')
        elif proximity_sensor == False:
            lcd_print('1:2: SENSOR BRAKE')
        elif button_status == 1:
            lcd_print('1:2: BUTTON BOARD BRAKE')
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
            change_speed += 2
            if change_speed >= max_speed:
                change_speed = max_speed

        if a_button == 1:
            change_speed -= 2
            if change_speed < min_speed:
                change_speed = min_speed

        if right_t == 1:
            if reverse == True:
                car_control(angle=angle, speed=-90)
            else:
                car_control(angle=angle, speed=change_speed)
                lcd_print('1:2: ')
                lcd_print('1:2: FORWARD')

        else:
            car_control(angle=angle, speed=0)
            #car_control(angle=angle, speed=default_speed)
            
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
            'proximity value': proximity_sensor,
            'hand brake': hand_brake,
            'reverse': reverse,
            'button in board': button_status,
            'right_t': right_t
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
    if start_gendata == False:
        lcd_print('1:2: ')
        lcd_print('1:2: STOPGENDATA')
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
        lcd_print('1:2: ')
        lcd_print('1:2: STARTGENDATA')
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
    global start_button, start_gendata

    for index in range(len(joy_data.axes)):
        x = -(joy_data.axes[2])
        y = joy_data.axes[3]
    for index in range(len(joy_data.buttons)):
        a_button = joy_data.buttons[0]
        y_button = joy_data.buttons[3]
        left_b = joy_data.buttons[4]
        right_b = joy_data.buttons[5]
        right_t = joy_data.buttons[7]
        start_button = joy_data.buttons[9]
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
=======
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
>>>>>>> 14387370ccb9a4fc1659b50f9c98ba2f7607c522
        pass


main()
