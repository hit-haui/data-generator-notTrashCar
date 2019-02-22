# Data generator for HAUI.notTrashCar
Data generator dành cho xe RC của team HAUI.notTrashCar

## Dependency

- Ubuntu 16.04 or newer
- [Melodic Morenia](http://wiki.ros.org/melodic) version of [ROS](https://ros.org)
```bash
sudo apt-get install ros-melodic-desktop-full
```
Plese follow full instruction at [Melodic Morenia wiki](http://wiki.ros.org/melodic)
- rosbridge-suite
```bash
sudo apt-get install ros-melodic-rosbridge-server
```
- Python 3.6+ (recommend Python 3.6.7) and required package:
```bash
pip3 install -r requirements.txt
```
- Newest version of Unity Simulator from FPT: https://goo.gl/EcHGUs

## Cách run

- Đầu tiên build lại package `team705`:
```bash
git clone https://github.com/lamhoangtung/data-generator-notTrashCar
cd data-generator-notTrashCar
catkin_make
```

- Package được chạy theo hướng dẫn trong file [`team705.launch`](/src/team705/launch/team705.launch), đầu tiên chạy `ros_bridge` khởi tạo kết nối đến simulator sau đó chạy đến [`main.py`](/src/team705/src/main.py) như một node của ROS

```bash
source ./devel/setup.bash
roslaunch team705 team705.launch
```

- Chạy simulator theo hướng dẫn [này](https://drive.google.com/open?id=14vCOzUO6_-6fyv0eypql1owZz3NIRiRY) với port là `127.0.0.1:9005`. Lưu ý một số ROS topic mới:
    - Topic nhận ảnh màu: `/camera/rgb/image_raw`
    - Topic nhận ảnh Depth: `/camera/depth/image_raw`
    - Topic set tốc độ: `/set_speed_car_api`
    - Topic set góc lái: `/set_steer_car_api`