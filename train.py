import numpy as np
import winsound

from mle import config as c, evolution, policy, visualize
from mle.environment import CartPoleBatch
from mle.visualize import animate_cartpoles

env = CartPoleBatch(n=c.N_PENDULUMS)
population = evolution.load_population()
on_generation = visualize.live_fitness_plot()
population = evolution.train(env, population, c.N_GENERATIONS, on_generation=on_generation)
evolution.save_population(population)
winsound.MessageBeep()

env.theta_threshold = np.inf
env.reset()
weights_and_biases = policy.build_weights(population, evolution.layout)
get_forces = lambda env: policy.forward(env, weights_and_biases)
animate_cartpoles(env, get_forces)