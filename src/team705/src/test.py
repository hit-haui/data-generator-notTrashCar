import rospy
from sensor_msgs.msg import Joy

def joy_callback(joy_data):
    for index in range(len(joy_data.axes)):
        print("axes:", index , "is:", joy_data.axes[index])
    for index in range(len(joy_data.buttons)):
        print("button:", index , "is:", joy_data.buttons[index])


    

def start():
    rospy.Subscriber("/joy", Joy, joy_callback)
    rospy.init_node('Joy2Turtle')
    rospy.spin()

if __name__ == "__main__":
    start()