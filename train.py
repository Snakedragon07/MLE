import config
from environment import CartPoleBatch
from visualize import animate_cartpoles

env = CartPoleBatch(n=15)
animate_cartpoles(env, get_forces=None)