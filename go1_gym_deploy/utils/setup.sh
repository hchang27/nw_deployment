cd /home/unitree/go2_hardware/scripts/build && sudo ./lcm_receive
sudo ifconfig lo multicast && sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev lo && sudo ./lcm_position_go2 eth0
sudo ./lcm_position_go2 eth0