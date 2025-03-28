cd ~/mit/parkour/go1_gym_deploy/launch

echo Entering pi. Will take a bit...

echo "Before SSH"
nohup sshpass -p "123" ssh -t go1-pi "~/kill_nodes.sh; bash -i" > /dev/null 2>&1
echo "After SSH"

echo Starting LCM...

SESSION_NAME=$(date +'%Y%m%d%H%M%S')

sshpass -p "123" ssh go1-nx "screen -dmS $SESSION_NAME"
sshpass -p "123" ssh go1-nx "screen -S $SESSION_NAME -p 0 -X stuff $'~/setup_lcm.sh\n'"
sshpass -p "123" ssh go1-nx "screen -S $SESSION_NAME -p 0 -X stuff $'\n'"


echo Deploying policy in current session...
conda activate torch

cd ~/mit/parkour

export PYTHONPATH=$PWD

which python
cd go1_gym_deploy/scripts
python deploy_vision.py
