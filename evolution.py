import environment
import config as c
import policy as p
import numpy as np
import os

layout = c.NETWORK_LAYOUT  # e.g. [5, 8, 1] -- worth defining this in config.py

def save_population(population, path=c.POPULATION_FILE):
    np.save(path, population)

def load_population(path=c.POPULATION_FILE):
    if os.path.exists(path):
        pop = np.load(path)
        if pop.shape[1] == p.p_count(layout):
            return pop
    return init_population()

# 1. POPULATION REPRESENTATION
def init_population():
    return np.random.uniform(-1, 1, size=(c.N_PENDULUMS, p.p_count(layout)))

# 2. Fitness per Step
def fitness(env):
    return ((1+np.cos(env.theta))/2) * (1-(env.x/env.x_threshold)**2)

# 3. FITNESS EVALUATION (one full episode)
def run_episode(env, population, N_times = c.N_Simulations):
    total_fitness = np.zeros(env.n)
    for i in range(N_times):
        env.reset()
        for step in range(c.MAX_STEPS):
            forces = p.MLP(env, population, layout)
            env.step(forces)
            total_fitness += np.where(env.alive, fitness(env), 0.0)
            if not env.alive.any():
                break
    total_fitness /= N_times
    return total_fitness

# 4. SELECTION & MUTATION
def evolve(population, fitness, evo_step, elite_frac=c.elite_frac, mutation_std=c.mutation_std, random_frac=c.random_frac, mutation_rate=c.mutation_rate):
    pop_size = population.shape[0]
    n_elites = int(pop_size * elite_frac)
    n_random = int(pop_size * random_frac)
    n_offspring = pop_size - n_elites - n_random

    elite_indices = np.argsort(fitness)[::-1][:n_elites]
    elites = population[elite_indices]

    new_population = np.zeros_like(population)
    new_population[:n_elites] = elites

    # rank-weighted parent selection: best elite reproduces more often than
    # the weakest elite, instead of every elite being equally likely (old:
    # np.random.randint(0, n_elites, ...) gave the #1 and #15 elite the same
    # odds of being a parent, which slows down exploitation of good solutions)
    elite_fitness = fitness[elite_indices]
    ranks = np.arange(n_elites, 0, -1)  # best elite = n_elites, worst = 1
    parent_probs = ranks / ranks.sum()
    parent_indices = np.random.choice(n_elites, size=n_offspring, p=parent_probs)
    parents = elites[parent_indices]

    # only mutate a subset of genes per offspring instead of nudging all
    # 129 parameters at once -- a full-vector random nudge in a high-dim
    # space rarely points toward an improvement, so most of it is wasted;
    # perturbing fewer genes with the same std gives more useful, localized
    # steps (classic bit-mutation-rate idea, applied to real-valued genes)
    gene_mask = np.random.rand(n_offspring, population.shape[1]) < mutation_rate

    noise = np.random.randn(n_offspring, population.shape[1]) * mutation_std * (1 - elite_fitness[parent_indices] / c.MAX_STEPS)[:, None]
    noise *= gene_mask
    new_population[n_elites:n_elites+n_offspring] = parents + noise

    new_population[n_elites+n_offspring:] = np.random.uniform(-1, 1, size=(n_random, population.shape[1]))

    return new_population

# 6. OUTER GENERATION LOOP
def train(env, population, n_generations):
    for generation in range(n_generations):
        scores = run_episode(env, population)
        population = evolve(population, scores, generation)
        print(generation, scores.max())
    return population