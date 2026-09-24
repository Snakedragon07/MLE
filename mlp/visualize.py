"""CPU part: live fitness plot and cart-pole animation (matplotlib)."""
import matplotlib

matplotlib.use("TkAgg")                 # real window, not PyCharm's static plot panel
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as c
from .simulation import mlp_forces, physics_step, unpack


def live_fitness_plot():
    """Open the fitness window and return update(generation, best, mean).

    Plots 1 - fitness on a log scale (inverted, so "up" = better).
    Esc in the window sets update.stop_requested = True, which train() checks.
    """
    plt.ion()
    fig, ax = plt.subplots()
    best_line, = ax.plot([], [], label="best")
    mean_line, = ax.plot([], [], label="mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("1 - fitness (log scale)")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title("Esc = stop training")

    gens, bests, means = [], [], []

    def update(generation, best, mean):
        gens.append(generation)
        bests.append(max(1 - best, 1e-6))
        means.append(max(1 - mean, 1e-6))
        if generation % c.PLOT_EVERY == 0:
            best_line.set_data(gens, bests)
            mean_line.set_data(gens, means)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
        fig.canvas.flush_events()           # keeps the window responsive (and catches Esc)

    update.stop_requested = False

    def on_key(event):
        if event.key == "escape":
            update.stop_requested = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=False)
    return update


def animate_best(population, n_show=c.N_SHOW):
    """Animate the best n_show controllers on the CPU. Row 0 = best, drawn black.

    Uses the same mlp_forces / physics_step as training, just on CPU tensors of shape (n, 1).
    Left / right arrow keys kick all carts.
    """
    plt.ioff()
    params = population[:n_show].detach().cpu().float()
    weights = unpack(params)
    n = params.shape[0]
    L = c.POLE_LENGTH

    # state, all (n, 1): random start angles, cart at rest in the middle
    s = {
        "x": torch.zeros(n, 1),
        "x_dot": torch.zeros(n, 1),
        "th": torch.empty(n, 1).uniform_(*c.THETA_INIT_RANGE),
        "th_dot": torch.zeros(n, 1),
        "alive": torch.ones(n, 1, dtype=torch.bool),
    }

    fig, ax = plt.subplots(figsize=c.PLOT_SIZE)
    ax.set_xlim(-1.2 * c.X_THRESHOLD, 1.2 * c.X_THRESHOLD)
    ax.set_ylim(-1.2 * L, 1.2 * L)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5)
    for edge in (-c.X_THRESHOLD, c.X_THRESHOLD):
        ax.axvline(edge, color="red", lw=0.5, ls="--")      # track limits
    ax.set_title("left / right arrow = kick the carts")

    poles, carts = [], []
    for i in range(n):
        style = dict(color="black", alpha=1.0, zorder=10) if i == 0 else dict(alpha=c.ALPHA)
        pole, = ax.plot([], [], "-", lw=2, **style)
        cart, = ax.plot([], [], "s", markersize=c.MARKER_SIZE, **style)
        poles.append(pole)
        carts.append(cart)

    def on_key(event):
        if event.key == "left":
            s["x_dot"] -= c.KICK
        elif event.key == "right":
            s["x_dot"] += c.KICK

    fig.canvas.mpl_connect("key_press_event", on_key)

    @torch.no_grad()
    def update(_frame):
        force = mlp_forces(weights, s["x"], s["x_dot"], s["th"], s["th_dot"])
        s["x"], s["x_dot"], s["th"], s["th_dot"], s["alive"] = physics_step(
            s["x"], s["x_dot"], s["th"], s["th_dot"], s["alive"], force)

        xs, ths, alive = s["x"][:, 0].numpy(), s["th"][:, 0].numpy(), s["alive"][:, 0].numpy()
        bob_x, bob_y = xs - L * np.sin(ths), L * np.cos(ths)
        for i in range(n):
            if alive[i]:
                poles[i].set_data([xs[i], bob_x[i]], [0, bob_y[i]])
                carts[i].set_data([xs[i]], [0])
            else:                                    # cart left the track: hide it
                poles[i].set_data([], [])
                carts[i].set_data([], [])
        return poles + carts

    ani = animation.FuncAnimation(fig, update, interval=c.INTERVAL_MS,
                                  blit=True, cache_frame_data=False)
    plt.show()
    return ani                                       # keep a reference, or the animation stops
