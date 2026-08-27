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
def evolve(population, fitness, evo_step, elite_frac=c.elite_frac, mutation_std=c.mutation_std, random_frac=c.random_frac):
    pop_size = population.shape[0]
    n_elites = int(pop_size * elite_frac)
    n_random = int(pop_size * random_frac)
    n_offspring = pop_size - n_elites - n_random

    elite_indices = np.argsort(fitness)[::-1][:n_elites]
    elites = population[elite_indices]

    new_population = np.zeros_like(population)
    new_population[:n_elites] = elites

    parent_indices = np.random.randint(0, n_elites, size=n_offspring)
    parents = elites[parent_indices]
    quality = fitness[elite_indices].mean() / c.MAX_STEPS
    gen_decay = 1 - evo_step / c.MAX_STEPS* quality

    noise = np.random.randn(n_offspring, population.shape[1]) * mutation_std * (1-fitness[elite_indices][parent_indices]/c.MAX_STEPS)[:,None] * gen_decay**2
    new_population[n_elites:n_elites+n_offspring] = parents + noise
    new_population[n_elites+n_offspring:] = np.random.uniform(-1, 1, size=(n_random, population.shape[1]))
    return new_population

# 6. OUTER GENERATION LOOP
def train(env, population, n_generations, on_generation=None):
    for generation in range(n_generations):
        scores = run_episode(env, population)
        n_random = int(len(scores) * c.random_frac)
        tracked = scores[:len(scores) - n_random]
        print(generation, "Max:", tracked.max()/c.MAX_STEPS, "Mean:", tracked.mean()/c.MAX_STEPS)
        if on_generation is not None:
            on_generation(generation, tracked)
        if tracked.max()/c.MAX_STEPS > c.Quality_threshhold and tracked.mean()/c.MAX_STEPS > c.Quality_mean_threshhold:
            break
        if getattr(on_generation, "stop_requested", False):
            print("Training stopped by user (Escape).")
            break
        population = evolve(population, scores, generation)
    return population