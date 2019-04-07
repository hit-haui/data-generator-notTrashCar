import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

graph = tf.get_default_graph()

model_traffic = load_model(
    '/home/vicker/Downloads/traffic_sign_019_0.98794.hdf5')
print('Loaded model')

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
