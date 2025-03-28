import time
import torch
from detr.policy import get_n_act_policy
TRACING_DEVICE = 'cuda'
checkpoint_path = "/home/unitree/nw_deploy/parkour/go1_gym_deploy/scripts/ckpts/go1_test/policy_last_alan.pt"
state_dict = torch.load(checkpoint_path, map_location=TRACING_DEVICE)
my_model = get_n_act_policy(10)
my_model.load_state_dict(state_dict)  # Load weights (key may vary)
my_model.eval().to(TRACING_DEVICE)  # Set the model to evaluation mode
ego_view = torch.randn(1, 10, 3, 180, 320).to(TRACING_DEVICE)
obs_input = torch.randn(1, 753).to(TRACING_DEVICE)
input_data = (ego_view, obs_input)

# warm up 
print("starting warm up")
for _ in range(10):
    ret = my_model(*input_data)
print("warm up done")

N_iters = 100
start_cp = time.time()
for _ in range(N_iters):
    ret = my_model(*input_data)

end_cp = time.time()
print(f"Time taken for {N_iters} iterations: {end_cp - start_cp} seconds")
print(f"Average FPS: {N_iters / (end_cp - start_cp)}")
