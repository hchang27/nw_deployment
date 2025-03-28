echo Killing previous instance...
ps aux | grep lcm | grep -v grep | awk '{print $2}' | xargs -r kill -9


echo 123 | sudo -S ifconfig eth0 multicast
echo 123 | sudo -S route add -net 224.0.0.0 netmask 240.0.0.0 dev eth0
echo 123 | sudo -S ip route add 239.255.76.67 dev eth0
export LCM_DEFAULT_URL=udpm://239.255.76.67:7667?ttl=255

cd /home/unitree/mit/go1_gym_deploy/unitree_legged_sdk_bin
./lcm_position
