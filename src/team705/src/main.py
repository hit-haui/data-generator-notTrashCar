#!/usr/bin/python3
import os
import sys
import time

import cv2
import numpy as np

import rospkg
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

try:
    os.chdir(os.path.dirname(__file__))
    os.system('clear')
    print("\nWait for initial setup, please don't connect anything yet...\n")
    sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except:
    pass

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_out_name = 'output_{}.avi'.format(time.time())
# video_out = cv2.VideoWriter(video_out_name, fourcc, 20, (320, 240))


def car_control(angle, speed):
    '''
    Hàm này dùng để gửi tín hiệu đến simulator
    '''
    pub_speed = rospy.Publisher('/team705_speed', Float32, queue_size=10)
    pub_speed.publish(speed)
    pub_angle = rospy.Publisher('/team705_steerAngle', Float32, queue_size=10)
    pub_angle.publish(angle)
    print('Angle:', angle, 'Speed:', speed)


def image_callback(data):
    '''
    Hàm này được gọi mỗi khi simulator trả về một bức ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    start_time = time.time()
    temp = np.fromstring(data.data, np.uint8)
    img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
    cv2.imshow('raw_frame', img)
    # video_out.write(img)
    cv2.waitKey(1)
    print('FPS:', 1/(time.time() - start_time))


'''
Nên viết thêm một hàm để nhận tín hiệu từ bàn phím và chuyển hoá thành góc quay và tốc độ, call nó trong `image_callback`,
để ghi nhận được 1 chuỗi phím thì nên dùng biến toàn cục để lưu lại, sao cho khi đang ở trong callback cũng có thể truy cập được.
Nhận phím thế nào thì chịu khó google nhé.
'''


def main():
    rospy.init_node('team705_node', anonymous=True)
    rospy.Subscriber("/team705_image/compressed", CompressedImage,
                     image_callback, queue_size=1, buff_size=2**24)
    rospy.spin()

    # video_out.release()
    # print('Saved', video_out_name)


if __name__ == '__main__':
    main()
