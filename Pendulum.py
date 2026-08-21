"""
Physically accurate pendulum simulation.

Models a simple (point-mass) pendulum using the full nonlinear equation of
motion (no small-angle approximation), integrated numerically with SciPy.
Includes optional damping, an energy-conservation check to validate the
simulation's accuracy, and a matplotlib animation.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ---------------------------------------------------------------------------
# Physical parameters -- change these to match your setup
# ---------------------------------------------------------------------------
g = 9.81          # gravitational acceleration [m/s^2]
L = 1.0           # pendulum length [m]
m = 1.0           # bob mass [kg] (cancels out of the motion, only used for energy)
b = 0.0           # damping coefficient [1/s]  (0 = ideal pendulum; try e.g. 0.2 for a damped one)

theta0 = np.radians(80)    # initial angle from vertical [rad]
omega0 = 0.0                # initial angular velocity [rad/s]

t_span = (0, 20)            # simulate 20 seconds
t_eval = np.linspace(*t_span, 2000)


# ---------------------------------------------------------------------------
# Equation of motion: theta'' = -(g/L) sin(theta) - b * theta'
#
# This is the EXACT nonlinear equation, valid for any swing amplitude --
# unlike the small-angle approximation theta'' = -(g/L) theta, which only
# holds for angles below about 10-15 degrees.
# ---------------------------------------------------------------------------
def pendulum_ode(t, y):
    theta, omega = y
    theta_dot = omega
    omega_dot = -(g / L) * np.sin(theta) - b * omega
    return [theta_dot, omega_dot]


sol = solve_ivp(
    pendulum_ode,
    t_span,
    [theta0, omega0],
    t_eval=t_eval,
    method="DOP853",   # high-order integrator -> very low numerical error
    rtol=1e-10,
    atol=1e-10,
)

theta = sol.y[0]
omega = sol.y[1]
t = sol.t


# ---------------------------------------------------------------------------
# Energy check: with b = 0, total mechanical energy must stay constant.
# This is the real test of "physically accurate" -- if energy drifts
# noticeably, the integrator or timestep is too coarse.
# ---------------------------------------------------------------------------
KE = 0.5 * m * (L * omega) ** 2
PE = m * g * L * (1 - np.cos(theta))
E = KE + PE

print(f"Energy drift over simulation: {E.max() - E.min():.3e} J "
      f"({(E.max() - E.min()) / E[0] * 100:.4f}% of initial energy)")


# ---------------------------------------------------------------------------
# Diagnostic plots: angle, angular velocity, and energy vs time
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

axes[0].plot(t, np.degrees(theta))
axes[0].set_ylabel("theta [deg]")

axes[1].plot(t, omega)
axes[1].set_ylabel("omega [rad/s]")

axes[2].plot(t, E)
axes[2].set_ylabel("Energy [J]")
axes[2].set_xlabel("time [s]")

fig.tight_layout()
fig.savefig("pendulum_diagnostics.png", dpi=150)


# ---------------------------------------------------------------------------
# Animation of the swinging pendulum
# ---------------------------------------------------------------------------
x = L * np.sin(theta)
y = -L * np.cos(theta)

fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.set_xlim(-1.2 * L, 1.2 * L)
ax2.set_ylim(-1.2 * L, 0.2 * L)
ax2.set_aspect("equal")
ax2.grid(True)

line, = ax2.plot([], [], "o-", lw=2, markersize=10)
trace, = ax2.plot([], [], "-", lw=1, alpha=0.4)

trace_x, trace_y = [], []

def animate(i):
    line.set_data([0, x[i]], [0, y[i]])
    trace_x.append(x[i])
    trace_y.append(y[i])
    trace.set_data(trace_x[-200:], trace_y[-200:])
    return line, trace

ani = animation.FuncAnimation(
    fig2, animate, frames=len(t), interval=1000 * (t[1] - t[0]), blit=True
)

plt.show()