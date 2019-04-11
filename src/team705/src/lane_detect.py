import numpy as np
import cv2
from param import *
import math
import time



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

def find(img):
    check = True
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, binary = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    binary = img
    #return np.nonzero(binary)[0][-1] if len(np.nonzero(binary)[0]) > 0 else img.shape[0]
    if len(np.nonzero(binary)[0]) > 0:
        x = np.nonzero(binary)[1][-1]
        y = np.nonzero(binary)[0][-1]
        return x,y, check
    else:
        check =False
        return 1,1, check
    

def detect_angle_lane_left(img):
    start_time = time.time()
    img = img[sky_line:,:]
    res,combined = detect_gray(img)
    no_cut = res
    print(res.shape)
    res = res[:res.shape[0]//2, :res.shape[1]//2]
    x,y,check = find(res)
    print(x,y)
    if check == True:
        cv2.line(res , (x, y), (x, y), (255, 255, 255), 5)
        cv2.line(img , (x, y), (x, y), (90, 0, 255), 5)
        cv2.line(img , (img.shape[1]//2, y ), (img.shape[1]//2, y), (90, 0, 255), 5)

        x_mid = img.shape[1]//2
        y_mid = img.shape[0]
        cv2.line(img , (x_mid, y_mid ), (x_mid, y_mid ), (90, 0, 255), 5)

        x_need = x+x_need_left #(img.shape[1]//2 + x) //2
        y_need = y

        cv2.line(img , (x_need, y_need ), (x_need, y_need ), (90, 90, 255), 5)

        cv2.line(img , (x_need, y_need ), (x_mid, y_mid ), (90, 90, 255), 5)

        angle = math.degrees(math.atan((x_mid - x_need)/(y_mid-y_need)))
    else :
        angle = 0
    print(angle)
    #cv2.imshow('no_cut', no_cut)
    debug = False
    if debug:
        cv2.imwrite('/home/nvidia/Desktop/data_visual/no_cut/'+str(time.time())+'.jpg',no_cut)
        cv2.imwrite('/home/nvidia/Desktop/data_visual/img/'+str(time.time())+'.jpg',img)
    #cv2.waitKey(1)
    return angle
'''
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
'''
def traffic_detect(raw_img):
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
            cv2.putText(raw_img, each_detection[0], (x_bot, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
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
    return traffic

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

def tree_detect(raw, rgb):
    height, weight = raw.shape[:2]
    weightmiddle = weight // 2
    green_img_left = rgb[:height, :weightmiddle//2][2].sum()
    green_img_right = rgb[:height, weightmiddle+weightmiddle//2:weight][2].sum()


    sum_white_pixels = np.sum(raw == 255)
    sum_green_pixels = green_img_left + green_img_right
    if sum_white_pixels <= 1 and green_img_left + green_img_right <= green_counting:
        status = True
    else:
        status = False
    return status
