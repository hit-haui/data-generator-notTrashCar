# Data generator for HAUI.notTrashCar
Data generator dành cho xe RC của team HAUI.notTrashCar trên simulator

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
- Tay cầm Xbox 360

## Cách run

- Đầu tiên build lại package `team705`:
```bash
git clone https://github.com/lamhoangtung/data-generator-notTrashCar
cd data-generator-notTrashCar
catkin_make
```
- Kết nối tay cầm Xbox 360 vào PC
- Chạy file [`team705.launch`](/src/team705/launch/team705.launch) với quyền root
```bash
sudo su
source ./devel/setup.bash
roslaunch team705 team705.launch
```

- Chạy simulator theo hướng dẫn [này](https://drive.google.com/open?id=14vCOzUO6_-6fyv0eypql1owZz3NIRiRY) với port là `127.0.0.1:9005`. 

## Cách sử dụng Data Generator
- Giữ phím `A` trên controller để phanh
- Giữ phím `Y` để giảm tốc độ của xe
- Dùng `left joystick` để điều khiển góc lái của xe
- Nhấn phím `X` để khởi động/thoát chế độ Reverse
- Nhấn `Ctrl + C` trên cửa sổ chạy file python để dừng quá trình generate data.

## Data format
Dữ liệu được sinh ra sẽ được lưu tại folder `recorded_data` tại vị trí chạy câu lệnh run (trong trường hợp là này root của repo). Bên trong có chứa 2 folder chứa ảnh RGB và Depth, cộng với 1 file `.json` tương ứng là label cho các ảnh.