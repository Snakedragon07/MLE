import config as c
import policy
import evolution
import numpy as np

from environment import CartPoleBatch
from visualize import animate_cartpoles

env = CartPoleBatch(n=c.N_PENDULUMS)
population = evolution.init_population()
population = evolution.train(env, population, c.N_GENERATIONS)


env.theta_threshold = np.inf
env.reset()
get_forces = lambda env: policy.MLP(env, population, evolution.layout)
animate_cartpoles(env, get_forces)