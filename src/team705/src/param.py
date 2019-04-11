import numpy as np

'''
PARAM WORLD
'''
#default_speed = 15
#x_need_right = 250 #200 #40
# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
# width of bottom edge of trapezoid, expressed as percentage of image width
trap_bottom_width = 1
trap_top_width = 1  # ditto for top edge of trapezoid
trap_height = 1  # height of the trapezoid expressed as percentage of image height
sky_line = 150 #70+20

# # Hide proximity sensor and car's hood
# hood_fill_color = (91, 104, 119)
# top_left_proximity = (255, 150)
# bottom_right_proximity = (400, 250)
# top_left_hood = (99, 210)
# bottom_right_hood = (515, 250)

''' LANE DETECT '''
# Color filtering
lower_white = 250 #200
upper_white = 255
kernel_size = 11
canny_low_threshold = 160
canny_high_threshold = 170

#Color gray filtering 
lower_gray = np.array([75,81,80])
upper_gray = np.array([140,255,255])

# Hough Transform
rho = 2  # distance resolution in pixels of the Hough grid
theta = np.pi/180  # angular resolution in radians of the Hough grid

threshold = 130  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line

max_line_gap = 50    # maximum gap in pixels between connectable line segments

# Angle calculation
# Height of destination points line calcualte from bottom of the frame
destination_line_height = 50
# Slope for left, right angle calculation when we only can find a single lane
destination_left_right_slope = 30

#tree
green_counting = 50000

# remember traffic status


no_traffic_size_count = 20


#Left:
##chieu ngang anh:
lane_left_pixel = 60 # 30
##chieu doc anh:
lane_left_pixel_height = 100 #50
x_need_left = 150 #100
#Right
##chieu ngang anh:
lane_right_pixel =100 #50
##chie doc anh
lane_right_pixel_height = 200 #100
#x_need_right = 240 #200 #40


'''
PARAM WORLD
'''
