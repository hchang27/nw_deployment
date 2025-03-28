import time
from go1_gym_deploy.modules.base.state_estimator import BasicStateEstimator

se = BasicStateEstimator("cpu")
se.spin_process()

while True:
    print(se.data)
    time.sleep(0.001)
