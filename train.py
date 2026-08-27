import config as c
import policy
import evolution
import numpy as np
import winsound

import visualize
from environment import CartPoleBatch
from visualize import animate_cartpoles

env = CartPoleBatch(n=c.N_PENDULUMS)
population = evolution.load_population()
on_generation = visualize.live_fitness_plot()
population = evolution.train(env, population, c.N_GENERATIONS, on_generation=on_generation)
evolution.save_population(population)
winsound.MessageBeep()

env.theta_threshold = np.inf
env.reset()
get_forces = lambda env: policy.MLP(env, population, evolution.layout)
animate_cartpoles(env, get_forces)