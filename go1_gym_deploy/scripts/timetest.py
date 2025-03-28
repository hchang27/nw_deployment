import torch
import time



policy = torch.jit.load("/home/unitree/nw_go1_test.jit")

print("warm up")
for _ in range(10):
    img = torch.randn(1, 10, 3, 180, 320).to("cuda")
    obs = torch.randn(1, 753).to("cuda")
    actions = policy(img, obs)

print("finish warming up")

stime = time.time()
for _ in range(100):
    img = torch.randn(1, 10, 3, 180, 320).to("cuda")
    obs = torch.randn(1, 753).to("cuda")
    actions = policy(img, obs)

print("this used ", time.time() - stime)