"""Load a saved population and animate the best controllers (no training)."""
import numpy as np

from mle import config as c, evolution, policy
from mle.environment import CartPoleBatch
from mle.visualize import animate_cartpoles

population = evolution.load_population()
env = CartPoleBatch(n=population.shape[0])
env.theta_threshold = np.inf
env.reset()
weights_and_biases = policy.build_weights(population, evolution.layout)
get_forces = lambda env: policy.forward(env, weights_and_biases)
animate_cartpoles(env, get_forces)
