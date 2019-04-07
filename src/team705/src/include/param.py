import numpy as np

'''
PARAM WORLD
'''
# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
# width of bottom edge of trapezoid, expressed as percentage of image width
trap_bottom_width = 1
trap_top_width = 1  # ditto for top edge of trapezoid
trap_height = 1  # height of the trapezoid expressed as percentage of image height
sky_line = 235+30

# Hide proximity sensor and car's hood
hood_fill_color = (91, 104, 119)
top_left_proximity = (255, 150)
bottom_right_proximity = (400, 250)
top_left_hood = (99, 210)
bottom_right_hood = (515, 250)

''' LANE DETECT '''
# Color filtering
lower_white = 200
upper_white = 255
kernel_size = 11
canny_low_threshold = 130
canny_high_threshold = 150

# Hough Transform
rho = 2  # distance resolution in pixels of the Hough grid
theta = np.pi/180  # angular resolution in radians of the Hough grid
threshold = 50  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line
max_line_gap = 25    # maximum gap in pixels between connectable line segments

# Angle calculation
# Height of destination points line calcualte from bottom of the frame
destination_line_height = 50
# Slope for left, right angle calculation when we only can find a single lane
destination_left_right_slope = 30

'''
PARAM WORLD
'''
