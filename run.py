import torch, math
from mlp.simulation import *
pop = torch.rand(4, N_PARAMS) * 2 - 1
start = torch.linspace(-math.pi, math.pi, 8)
print(run_episodes(pop, start))   # 4 numbers between 0 and MAX_STEPS