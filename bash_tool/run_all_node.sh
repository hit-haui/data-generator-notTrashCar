trap "exit" INT TERM ERR SIGINT SIGTERM
trap "kill 0" EXIT

#cd catkin_ws
#source ~/.bashrc &
#source ~/catkin_ws/devel/setup.bash &
rosparam set /lcd_adr 63
rosrun car_controller car_controller_node &
rosrun hal hal_node &
roslaunch astra_launch astra.launch &
rostopic pub /lcd_print std_msgs/String "data: '1:2:Haui.NotTrashCar'"
wait
