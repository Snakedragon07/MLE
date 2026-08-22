import config as c
import policy

from environment import CartPoleBatch
from visualize import animate_cartpoles


env = CartPoleBatch(n=c.N_PENDULUMS)
animate_cartpoles(env, policy.P)