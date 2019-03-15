# Data generator for HAUI.notTrashCar (required MASTER PC)
Data generator dành cho xe RC của team HAUI.notTrashCar cần thêm master PC

# Data generator for HAUI.notTrashCar
Data generator dành cho xe RC của team HAUI.notTrashCar

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
- Kết nối PC và board Jeston TX2 của xe vào cùng một mạng local ở trên
- Khởi tạo server `roscore` trên MASTER PC
```bash
sudo su
cd data-generator-notTrashCar
source ./devel/setup.bash
source ~/.bashrc
roscore
```
- Khởi chạy tất cả các node ROS để giao tiếp với phần cứng trên mạch Jeston
```bash
./run_all_node.sh
```
- Kết nối tay cầm Xbox 360 vào MASTER PC
- Chạy script [`main.py`](/src/team705/src/main.py) với quyền root

```bash
sudo su
cd data-generator-notTrashCar
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

## Data format
Dữ liệu được sinh ra sẽ được lưu tại folder `recorded_data` tại vị trí chạy câu lệnh run (trong trường hợp là này root của repo). Bên trong có chứa 2 folder chứa ảnh RGB và Depth, cộng với 1 file `.json` tương ứng là label cho các ảnh.
