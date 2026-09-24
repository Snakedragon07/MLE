import math
import torch
from . import config as c

DEVICE = torch.device(c.DEVICE if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

LAYERS = list(zip(c.NETWORK_LAYOUT, c.NETWORK_LAYOUT[1:]))   # [(5,10), (10,10), (10,1)]
SIZES = [(n_in + 1) * n_out for n_in, n_out in LAYERS]         # [60, 110, 11]
N_PARAMS = sum(SIZES)                                           # 181

def unpack(population):
    """(P, N_PARAMS) -> one tensor per layer, (P, n_in + 1, n_out). Last row = biases."""
    P = population.shape[0]
    return [p.reshape(P, n_in + 1, n_out)
        for p, (n_in, n_out) in zip(population.split(SIZES, dim=1), LAYERS)]

def mlp_forces(weights, x, x_dot, th, th_dot):
    """States (P, S) -> forces (P, S)."""
    # 1. inputs: 5 features per cart-pole -> (P, S, 5)
    h = torch.stack([torch.sin(th), torch.cos(th), th_dot, x / c.X_THRESHOLD, x_dot], dim=-1)

    # 2. layers: (P,S,5) -> (P,S,10) -> (P,S,10) -> (P,S,1)
    for W in weights:
        h = torch.tanh(torch.bmm(h, W[:, :-1]) + W[:, -1:])

    # 3. output in [-1, 1] -> force in [-FORCE_MAG, FORCE_MAG]
    return h[..., 0] * c.FORCE_MAG

def physics_step(x, x_dot, th, th_dot, alive, force):
    """One Euler step for all cart-poles. All inputs (P, S). Returns x, x_dot, th, th_dot, alive."""
    # 1. dead cart-poles get no force
    force = torch.where(alive, force, torch.zeros_like(force))

    # 2. accelerations
    x_ddot = (force - c.CART_FRICTION * x_dot) / c.CART_MASS
    th_ddot = (x_ddot / c.POLE_LENGTH * torch.cos(th)
               + c.GRAVITY / c.POLE_LENGTH * torch.sin(th)
               - c.POLE_DAMPING * th_dot)

    # 3. integrate: velocity first, then position
    new_x_dot = x_dot + x_ddot * c.DT
    new_x = x + new_x_dot * c.DT
    new_th_dot = th_dot + th_ddot * c.DT
    new_th = th + new_th_dot * c.DT

    # 4. wrap angle to [-pi, pi)
    new_th = torch.remainder(new_th + math.pi, 2 * math.pi) - math.pi

    # 5. only alive cart-poles move, dead ones keep their old state
    x_dot = torch.where(alive, new_x_dot, x_dot)
    x = torch.where(alive, new_x, x)
    th_dot = torch.where(alive, new_th_dot, th_dot)
    th = torch.where(alive, new_th, th)

    # 6. who is still alive? once dead, stays dead
    alive = alive & (th.abs() < c.THETA_THRESHOLD) & (x.abs() < c.X_THRESHOLD)

    return x, x_dot, th, th_dot, alive

def step_fitness(x,th):
    return ((1 + torch.cos(th)) / 2) * (1- (x/c.X_THRESHOLD)**2)

@torch.no_grad()
def run_episodes(population, start_th):
    """population (P, N_PARAMS), start_th (S,) -> mean fitness per individual (P,)."""
    P, S = population.shape[0], start_th.shape[0]
    weights = unpack(population)

    # start state, all (P, S): every individual gets the same S start angles
    th = start_th.expand(P, S).clone()
    x = torch.zeros_like(th)
    x_dot = torch.zeros_like(th)
    th_dot = torch.zeros_like(th)
    alive = torch.ones_like(th, dtype=torch.bool)
    total = torch.zeros_like(th)                  # summed fitness per episode

    for step in range(c.MAX_STEPS):
        force = mlp_forces(weights, x, x_dot, th, th_dot)
        x, x_dot, th, th_dot, alive = physics_step(x, x_dot, th, th_dot, alive, force)
        total += torch.where(alive, step_fitness(x, th), torch.zeros_like(total))

        if (step + 1) % c.EARLY_EXIT_CHECK == 0 and not alive.any():
            break

    return total.mean(dim=1)                      # average over the S episodes