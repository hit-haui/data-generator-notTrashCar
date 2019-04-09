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
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.backend.tensorflow_backend import set_session
import math
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

graph = tf.get_default_graph()

model_traffic = load_model(
    '/home/dejavu/Downloads/traffic_sign_019_0.98794.hdf5')
print('Loaded model')

try:
    sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except:
   pass
import cv2

def predict_traffic(img):
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
        print('have circle')
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
                #print(l,r,n)
                if max(l,max(n,r)) == traffic_list[0]:
                    left +=1
                elif max(l, max(n, r)) == traffic_list[1]:
                    none +=1
                elif max(l, max(n, r)) == traffic_list[2]:
                    right +=1
    if left > right:
        print('Left: <-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-<-< :Left') 
        return 'Left'
    elif left < right:
        print('Right: ->->->->->->->->->->->->->->->->->->->->->->->-> :Right')
        return 'Right'
    else:
        return 'No traffic'

#left, center, right
traffic_status_list = [0,0,0]
flow_lane = 0

kernel_size = 5
def to_hls(img):
    """
    Returns the same image in HLS format
    The input image must be in RGB format
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def to_lab(img):
    """
    Returns the same image in LAB format
    Th input image must be in RGB format
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def abs_sobel(gray_img, x_dir=True, kernel_size=kernel_size, thres=(0, 255)):
    """
    Applies the sobel operator to a grayscale-like (i.e. single channel) image in either horizontal or vertical direction
    The function also computes the asbolute value of the resulting matrix and applies a binary threshold
    """
    sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) if x_dir else cv2.Sobel(
        gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))

    gradient_mask = np.zeros_like(sobel_scaled)
    gradient_mask[(thres[0] <= sobel_scaled) & (sobel_scaled <= thres[1])] = 1
    return gradient_mask


def mag_sobel(gray_img, kernel_size=kernel_size, thres=(0, 255)):
    """
    Computes sobel matrix in both x and y directions, merges them by computing the magnitude in both directions
    and applies a threshold value to only set pixels within the specified range
    """
    sx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sxy = np.sqrt(np.square(sx) + np.square(sy))
    scaled_sxy = np.uint8(255 * sxy / np.max(sxy))

    sxy_binary = np.zeros_like(scaled_sxy)
    sxy_binary[(scaled_sxy >= thres[0]) & (scaled_sxy <= thres[1])] = 1

    return sxy_binary


def dir_sobel(gray_img, kernel_size=kernel_size, thres=(0, np.pi/2)):
    """
    Computes sobel matrix in both x and y directions, gets their absolute values to find the direction of the gradient
    and applies a threshold value to only set pixels within the specified range
    """
    sx_abs = np.absolute(
        cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sy_abs = np.absolute(
        cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size))

    dir_sxy = np.arctan2(sx_abs, sy_abs)

    binary_output = np.zeros_like(dir_sxy)
    binary_output[(dir_sxy >= thres[0]) & (dir_sxy <= thres[1])] = 1

    return binary_output


def compute_hls_white_binary(rgb_img):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB format
    """
    hls_img = to_hls(rgb_img)

    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_bin = np.zeros_like(hls_img[:, :, 0])
    img_hls_white_bin[((hls_img[:, :, 0] >= 0) & (hls_img[:, :, 0] <= 255))
                      & ((hls_img[:, :, 1] >= 220) & (hls_img[:, :, 1] <= 255))
                      & ((hls_img[:, :, 2] >= 0) & (hls_img[:, :, 2] <= 255))
                      ] = 1

    return img_hls_white_bin


def combined_sobels(sx_binary, sy_binary, sxy_magnitude_binary, gray_img, kernel_size=kernel_size, angle_thres=(0, np.pi/2)):
    sxy_direction_binary = dir_sobel(
        gray_img, kernel_size=kernel_size, thres=angle_thres)

    combined = np.zeros_like(sxy_direction_binary)
    # Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels
    combined[(sx_binary == 1) | ((sy_binary == 1) & (
        sxy_magnitude_binary == 1) & (sxy_direction_binary == 1))] = 1

    return combined


def get_combined_binary_thresholded_img(undist_img):
    """
    Applies a combination of binary Sobel and color thresholding to an undistorted image
    Those binary images are then combined to produce the returned binary image
    """
    undist_img_gray = to_lab(undist_img)[:, :, 0]
    sx = abs_sobel(undist_img_gray, kernel_size=15, thres=(20, 120))
    sy = abs_sobel(undist_img_gray, x_dir=False,
                   kernel_size=15, thres=(20, 120))
    sxy = mag_sobel(undist_img_gray, kernel_size=15, thres=(80, 200))
    sxy_combined_dir = combined_sobels(
        sx, sy, sxy, undist_img_gray, kernel_size=15, angle_thres=(np.pi/4, np.pi/2))

    hls_w_y_thres = compute_hls_white_binary(undist_img)

    combined_binary = np.zeros_like(hls_w_y_thres)
    combined_binary[(sxy_combined_dir == 1) | (hls_w_y_thres == 1)] = 1

    return combined_binary


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # line_img = np.zeros(img.shape, dtype=np.uint8)  # this produces single-channel (grayscale) image
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
    angle = draw_lines(line_img, lines)
    #draw_lines_debug2(line_img, lines)
    return line_img, angle


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # In case of error, don't draw the line
    draw_right = True
    draw_left = True

    # Find slopes of all lines
    # But only care about lines where abs(slope) > slope_threshold
    slope_threshold = 0.5
    slopes = []
    new_lines = []
    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]

        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding division by 0
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)

        # Filter lines based on slope
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)

    lines = new_lines

    # Split lines into right_lines and left_lines, representing the right and left lane lines
    # Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2  # x coordinate of center of image
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)

    # Run linear regression to find best fit line for right and left lane lines
    # Right lane lines
    right_lines_x = []
    right_lines_y = []

    for line in right_lines:
        x1, y1, x2, y2 = line[0]

        right_lines_x.append(x1)
        right_lines_x.append(x2)

        right_lines_y.append(y1)
        right_lines_y.append(y2)

    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(
            right_lines_x, right_lines_y, 1)  # y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False

    # Left lane lines
    left_lines_x = []
    left_lines_y = []

    for line in left_lines:
        x1, y1, x2, y2 = line[0]

        left_lines_x.append(x1)
        left_lines_x.append(x2)

        left_lines_y.append(y1)
        left_lines_y.append(y2)

    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(
            left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
        left_m, left_b = 1, 1
        draw_left = False

    # Find 2 end points for right and left lines, used for drawing the line
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - trap_height)

    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m

    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m

    # Convert calculated end points from float to int
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    # print('Left:')
    # print('(', left_x1, ',', y1, '), (', left_x2, ',', y2, ')')
    # print('Right:')
    # print('(', right_x1, ',', y1, '), (', right_x2, ',', y2, ')')

    # Draw the right and left lines on image
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

    angle = calculate_angle(img, draw_left, draw_right,
                            left_x1, left_x2, right_x1, right_x2, x1, y1)
    return angle


def calculate_angle(img, draw_left, draw_right, left_x1, left_x2, right_x1, right_x2, x1, y1):
    if draw_left and draw_right:
        x_des = int((left_x1 + right_x1)/2)
        y_des = int(y1-destination_line_height)
    elif draw_left and not draw_right:
        x_des = img.shape[1]//2 + destination_left_right_slope
        y_des = img.shape[0] - destination_line_height
    elif draw_right and not draw_left:
        x_des = img.shape[1]//2 - destination_left_right_slope
        y_des = img.shape[0] - destination_line_height
    else:
        x_des = img.shape[1]//2
        y_des = img.shape[0] - destination_line_height

    # print('Center:')
    # print('(', x_des, ',', y_des, ')')
    car_pos_x = img.shape[1]//2
    car_pos_y = img.shape[0]
    dx = x_des - car_pos_x
    dy = car_pos_y - y_des

    if dx < 0:
        angle = -np.arctan(-dx/dy) * 180/math.pi
    elif dx == 0:
        angle = 0
    else:
        angle = np.arctan(dx/dy) * 180/math.pi
    # print(angle)
    try:
        img = cv2.line(img, (car_pos_x, car_pos_y),
                       (x_des, y_des), (0, 0, 255), 3)
        # img[y_des-5:y_des+5, x_des-5:x_des+5] = (0, 0, 255)
    except:
        pass
    return angle


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def easy_lane_preprocess(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray_img)
    mask_white = cv2.inRange(gray_img, lower_white, upper_white)
    combined = cv2.bitwise_and(gray_img, mask_white)
    combined = gaussian_blur(combined, kernel_size)
    combined = canny(combined, canny_low_threshold, canny_high_threshold)
    # cv2.imshow('combined', combined)
    return combined

def detect_gray(img):
    combined_white = easy_lane_preprocess(img)
    hsv = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)

    mask_white = cv2.inRange(hsv, lower_gray, upper_gray)

    combined = cv2.bitwise_and(img,img, mask=mask_white)
    combined = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
    combined = canny(combined, canny_low_threshold, canny_high_threshold)

    res = cv2.add(combined,combined_white)
    
    return res,combined
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

    if traffic_status_list[0] >= 2:
        flow_lane = -1
        traffic = -1
    elif traffic_status_list[2] >=2:
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
    annotated_image = cv2.cvtColor(weighted_img(detect white color in threshold image
        line_image, test_img), cv2.COLOR_RGB2BGR)
    return annotated_image, angle
    


def snow_detech(raw_image):
    combined = get_combined_binary_thresholded_img(
        cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)) * 255
    height, width = combined.shape[:2]
    pixel_sum_value = np.sum(combined)
    rate = (pixel_sum_value / (height * width*255)) * 100
    print('rate', rate)
    if rate < 10.5:
        snow = False
    else:
        snow = True
    return snow

def lcd_print(s):
    lcd = rospy.Publisher('/lcd_print', String, queue_size=10)
    lcd.publish(s)

proximity_sensor = True
bt1_sensor = bt2_sensor = bt3_sensor = bt4_sensor = False
hand_brake = True
default_speed = 8
max_speed = 15
max_speed_mode = False
    
    

def image_callback(rgb_data):
    '''
    Hàm này được gọi mỗi khi simulator trả về ảnh, vậy nên có thể gọi điều khiển xe ở đây
    '''
    global hand_brake, max_speed_mode
    print('call back')
    start_time = time.time()
    temp = np.fromstring(rgb_data.data, np.uint8)
    rgb_img = cv2.imdecode(temp, cv2.IMREAD_COLOR)
   # cv2.imwrite('/home/vicker/Desktop/data/'+str(start_time)+'.jpg',rgb_img)
    # rgb_img = cv2.resize(rgb_img, (480, 640))
    # print(rgb_img.shape)
    annotated_image, angle = process_frame(rgb_img)
    snow = snow_detech(rgb_img)

    if bt1_sensor == True:
        hand_brake = True if hand_brake == False else False
    print(hand_brake)

    if hand_brake == True:
        lcd_print('1:2: HANDBRAKE')
        car_control(angle=0, speed=0)

    if hand_brake == False and proximity_sensor == False:
        lcd_print('1:2:                    ')
        while proximity_sensor == False:
            lcd_print('1:2: PROXIMITY')
            car_control(angle = 0, speed = 0)
        lcd_print('1:2:                   ')

    if hand_brake == False and proximity_sensor == True:
        if bt3_sensor == True:
            max_speed_mode = True if max_speed_mode == False else False
        if max_speed_mode:
            lcd_print('1:2:  MAXSPEED')
            car_control(angle=angle, speed=max_speed)
        else:
            lcd_print('1:2:  MINSPEED')
            car_control(angle=angle, speed=default_speed)
    

    cv2.imshow('processed_frame', annotated_image)
    cv2.waitKey(1)
    # rgb_img = cv2.resize(rgb_img, img_size[:-1])
    print("FPS:", 1/(time.time()-start_time))
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
