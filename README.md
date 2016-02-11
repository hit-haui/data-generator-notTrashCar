# Data generator for HAUI.notTrashCar (Simulator Only)
Data generator dành cho xe RC của team HAUI.notTrashCar trên simulator

## Dependency

- Ubuntu 16.04 or newer
- [Melodic Morenia](http://wiki.ros.org/melodic) version of [ROS](https://ros.org)
```bash
sudo apt-get install ros-melodic-desktop-full
```
Plese follow full instruction at [Melodic Morenia wiki](http://wiki.ros.org/melodic)

- Python 3.6+ (recommend Python 3.6.7) and required package:
```bash
pip3 install -r requirements.txt
```
- Newest version of Unity Simulator from FPT: https://goo.gl/EcHGUs
- Tay cầm Xbox 360


- Router wifi để thiết lập mạng local cho xe và MASTER PC, ưu tiên sử dụng mạng 5GHz

- Tay cầm Xbox 360

- Download code điều khiển của xe tại đây, giải nén ra tại thư mục `HOME` của board Jeston trên xe.

- Config MASTER PC và board mạch của xe bằng cách thêm vào cuối file `.bashrc` nội dung sau:
    - Trên MASTER PC:
    ```bash
    export ROS_MASTER_URL=http://localhost:11311/
    export ROS_HOSTNAME=<Local IP của MASTER PC>
    export ROS_IP=<Local IP của MASTER PC>
    ```
    Lập lại bước trên với `.bashrc` nằm trên vị trí `~/.bashrc` của user `root` của MASTER PC
    ```bash
    sudo su
    nano ~/.bashrc
    ...
    ```
    `Ctrl + X` -> `Y` -> `Enter` để save.
    - Trên mạch xe (có thể SSH vào để chỉnh sửa):
    ```bash
    export ROS_MASTER_URL=http://<Local IP của MASTER PC>:11311/
    export ROS_HOSTNAME=<Local IP của xe>
    export ROS_IP=<Local IP của xe>
    ```
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
>>>>>>> 55a7353161a1f46a4668693a6920e6de17df2279
source ./devel/setup.bash
source ~/.bashrc
rosrun team705 main.py
```
## Cách sử dụng Data Generator
- Dùng phím `B` trên controller để chuyển qua trạng thái Dừng hoặc Di chuyển
- Giữ phím `Y` để giảm tốc độ của xe
- Dùng `left joystick` để điều khiển góc lái của xe
- Nhấn phím `X` để khởi động/thoát chế độ Reverse
- Nhấn `Ctrl + C` trên cửa sổ chạy file python để dừng quá trình generate data.

- Chạy simulator theo hướng dẫn [này](https://drive.google.com/open?id=14vCOzUO6_-6fyv0eypql1owZz3NIRiRY) với port là `127.0.0.1:9005`. 

## Cách sử dụng Data Generator
- Giữ phím `A` trên controller để phanh
- Giữ phím `Y` để giảm tốc độ của xe
- Dùng `left joystick` để điều khiển góc lái của xe
- Nhấn phím `X` để khởi động/thoát chế độ Reverse
- Nhấn `Ctrl + C` trên cửa sổ chạy file python để dừng quá trình generate data.

## Data format
Dữ liệu được sinh ra sẽ được lưu tại folder `recorded_data` tại vị trí chạy câu lệnh run (trong trường hợp là này root của repo). Bên trong có chứa 2 folder chứa ảnh RGB và Depth, cộng với 1 file `.json` tương ứng là label cho các ảnh.

