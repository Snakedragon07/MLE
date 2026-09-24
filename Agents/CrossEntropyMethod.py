import os
import time

import numpy as np
import torch

from mlp import simulation as sim
from mlp import config as c

def decay_schedule(generation):
    """Mutation strength shrinks slowly over the generations."""
    return c.DECAY_RATE ** generation

@torch.no_grad()
def evolve(population, fitness, generation):
    P = population.shape[0]
    n_elites = int(P * c.ELITE_FRAC)

    order = torch.argsort(fitness, descending=True)[:n_elites]
    elites = population[order]

    mean = elites.mean(dim=0)
    std = elites.std(dim=0) + c.CEM_EXTRA_STD * decay_schedule(generation)

    samples = mean + std * torch.randn(P - 1, sim.N_PARAMS, device=sim.DEVICE, dtype=sim.DTYPE)
    return torch.cat([mean[None, :], samples])          # (P, N_PARAMS)