if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <interface>"
    exit 1
fi

INTERFACE=$1

echo Using interface $INTERFACE

cd ~/mit/parkour/go1_gym_deploy/launch

./setup_network.sh $INTERFACE
wait

echo sleeping a bit...
sleep 2

echo running!

tmux new-session -d "./deploy.sh $INTERFACE; bash"
