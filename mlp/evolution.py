"""Evolution: population + selection/mutation on the GPU, training loop + files on the CPU."""
import os
import time

import numpy as np
import torch

from . import config as c
from .simulation import DEVICE, DTYPE, N_PARAMS, run_episodes


# ============================== GPU side ==============================

def init_population(n=c.POP_SIZE):
    """(n, N_PARAMS) random parameters in [-1, 1] on the GPU."""
    return torch.rand(n, N_PARAMS, device=DEVICE, dtype=DTYPE) * 2 - 1


def start_angles(n_sims=c.N_SIMULATIONS):
    """(n_sims,) start angles, one random angle per equal slice of THETA_INIT_RANGE."""
    lo, hi = c.THETA_INIT_RANGE
    edges = torch.linspace(lo, hi, n_sims + 1, device=DEVICE, dtype=DTYPE)
    return edges[:-1] + torch.rand(n_sims, device=DEVICE, dtype=DTYPE) * (edges[1:] - edges[:-1])


def decay_schedule(generation):
    """Mutation strength shrinks slowly over the generations."""
    return c.DECAY_RATE ** generation


@torch.no_grad()
def evolve(population, fitness, generation):
    """Pure CEM: row 0 = cloud mean (the current best guess), rest = samples from the cloud."""
    P = population.shape[0]
    n_elites = int(P * c.ELITE_FRAC)

    order = torch.argsort(fitness, descending=True)[:n_elites]
    elites = population[order]

    mean = elites.mean(dim=0)
    std = elites.std(dim=0) + c.CEM_EXTRA_STD * decay_schedule(generation)

    samples = mean + std * torch.randn(P - 1, N_PARAMS, device=DEVICE, dtype=DTYPE)
    return torch.cat([mean[None, :], samples])          # (P, N_PARAMS)


# ============================== CPU side ==============================

def save_population(population, path=c.POPULATION_FILE):
    """GPU tensor -> .npy file (numpy needs the data on the CPU)."""
    np.save(path, population.cpu().numpy())


def load_population(path=c.POPULATION_FILE, n=c.POP_SIZE):
    """Saved population if it fits the network, otherwise a fresh one. Always n rows, on DEVICE."""
    if not os.path.exists(path):
        print("No saved population found, starting fresh.")
        return init_population(n)

    pop = torch.as_tensor(np.load(path), device=DEVICE, dtype=DTYPE)
    if pop.ndim != 2 or pop.shape[1] != N_PARAMS:
        print(f"{path} has shape {tuple(pop.shape)}, network needs {N_PARAMS} parameters. Starting fresh.")
        return init_population(n)

    if pop.shape[0] < n:                          # e.g. an old file with 50 rows: fill up with random ones
        pop = torch.cat([pop, init_population(n - pop.shape[0])])
    print(f"Loaded {path}")
    return pop[:n]


def train(population, n_generations=c.N_GENERATIONS, on_generation=None):
    """Main loop. Returns the population sorted best-first.

    on_generation(generation, best, mean) is called every generation (e.g. to update a plot).
    If it has an attribute stop_requested that becomes True, training stops.
    """
    P = population.shape[0]
    n_random = int(P * c.RANDOM_FRAC)
    scores = None                                 # scores that belong to the CURRENT population

    try:
        for gen in range(n_generations):
            t0 = time.perf_counter()

            # GPU: evaluate everyone
            scores = run_episodes(population, start_angles())

            # CPU: only two numbers come back (.item() waits for the GPU -> timing is real)
            tracked = scores[:P - n_random] / c.MAX_STEPS        # ignore the random newcomers
            best, mean = tracked.max().item(), tracked.mean().item()
            ms = (time.perf_counter() - t0) * 1000
            print(f"{gen:6d}  best {best:.4f}  mean {mean:.4f}  decay {decay_schedule(gen):.4f}  {ms:5.0f} ms")

            if on_generation is not None:
                on_generation(gen, best, mean)
            if best > c.QUALITY_BEST and mean > c.QUALITY_MEAN:
                print("Quality threshold reached.")
                break
            if getattr(on_generation, "stop_requested", False):
                print("Stopped by user.")
                break

            # GPU: next generation
            population = evolve(population, scores, gen)
            scores = None                         # the new population has no scores yet

            if gen % c.SAVE_EVERY == 0:
                save_population(population)

    except KeyboardInterrupt:
        print("Interrupted (Ctrl+C).")

    # best first: after evolve() the elites are already on top; otherwise sort by the last scores
    if scores is not None:
        population = population[torch.argsort(scores, descending=True)]
    return population
