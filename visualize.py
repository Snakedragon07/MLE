import config as c

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def live_fitness_plot():
    plt.ion()
    fig, ax = plt.subplots()
    best_line, = ax.plot([], [], label="best")
    mean_line, = ax.plot([], [], label="mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("1 - fitness (log scale)")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.legend()

    gens, bests, means = [], [], []

    def update(generation, scores):
        gens.append(generation)
        bests.append(max(1 - scores.max()/c.MAX_STEPS, 1e-6))
        means.append(max(1 - scores.mean()/c.MAX_STEPS, 1e-6))
        best_line.set_data(gens, bests)
        mean_line.set_data(gens, means)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    update.stop_requested = False

    def on_key(event):
        if event.key == 'escape':
            update.stop_requested = True

    fig.canvas.mpl_connect('key_press_event', on_key)

    return update

def animate_cartpoles(env, get_forces = None, alpha=c.alpha, interval = c.interval, plotsize = c.plotsize, kick = 1.0):
    plt.ioff()

    fig, ax = plt.subplots(figsize=plotsize)

    ax.set_xlim(-1.2*env.x_threshold, 1.2*env.x_threshold)
    ax.set_ylim(-1.2*c.POLE_LENGTH, 1.2*c.POLE_LENGTH)
    ax.set_aspect('equal')

    poles = []
    carts = []

    #The best cart of last iteration has alpha 1
    pole_line, = ax.plot([], [], "-",color= "black", lw=2, alpha=1, zorder = 10)
    cart_marker, = ax.plot([], [], "s",color = "black", markersize=c.markersize, alpha=1, zorder = 10)
    poles.append(pole_line)
    carts.append(cart_marker)

    for i in range(env.n-1):
        pole_line, = ax.plot([], [], "-", lw=2, alpha=alpha)
        cart_marker, = ax.plot([], [], "s", markersize=c.markersize, alpha = alpha)
        poles.append(pole_line)
        carts.append(cart_marker)

    def on_key(event):
        if event.key == 'left':
            env.x_dot -= kick
        elif event.key == 'right':
            env.x_dot += kick

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame):
        forces = get_forces(env) if get_forces is not None else np.zeros(env.n)
        env.step(forces)

        for i in range(env.n):
            if env.alive[i]:
                pivot_x, pivot_y = env.x[i], 0
                bob_x, bob_y = env.x[i] - env.l*np.sin(env.theta[i]), env.l*np.cos(env.theta[i])
                poles[i].set_data([pivot_x, bob_x], [pivot_y, bob_y])
                carts[i].set_data([pivot_x], [pivot_y])
            else:
                poles[i].set_data([], [])
                carts[i].set_data([], [])

        return poles + carts

    ani = animation.FuncAnimation(fig, update, interval=interval, blit=True, cache_frame_data=False)
    plt.show()