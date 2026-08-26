import config as c

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def animate_cartpoles(env, get_forces = None, alpha=c.alpha, interval = c.interval, plotsize = c.plotsize, kick = 1.0):

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