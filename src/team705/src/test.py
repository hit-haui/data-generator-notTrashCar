import math, xbox

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
    elif x > 0.0  and y < 0.0:
        angle = math.degrees(math.atan(x/y)) + 180.0
    elif x < 0.0 and y < 0.0:
        angle = math.degrees(math.atan(x/y)) - 180.0
    elif x < 0.0 and y > 0.0:
        angle = math.degrees(math.atan(x/y))
    return angle

def joy_stick_controller():
    joy = xbox.Joystick()
    while not joy.Back():
        angle = 0.0
        x, y = joy.leftStick()
        angle = convert_to_angle(x, y)
        print(angle)

if __name__ == '__main__':
    joy_stick_controller()