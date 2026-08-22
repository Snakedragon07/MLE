import config as c
import numpy as np

class CartPoleBatch:
    def __init__(self, n, g=c.GRAVITY, m=c.CART_MASS, l=c.POLE_LENGTH,
                 force_mag=c.FORCE_MAG, damp=c.DAMPING_COEFF, dt=c.DT,
                 theta_threshold=c.THETA_THRESHOLD, x_threshold=c.X_THRESHOLD,
                 seed=c.SEED):
        self.n = n
        self.g = g
        self.m = m
        self.l = l
        self.force_mag = force_mag
        self.damp = damp
        self.dt = dt
        self.theta_threshold = theta_threshold
        self.x_threshold = x_threshold
        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.theta = self.rng.uniform(*c.THETA_INIT_RANGE, size = self.n)
        self.theta_dot = self.rng.uniform(*c.THETA_DOT_INIT_RANGE, size = self.n)
        self.x = self.rng.uniform(*c.X_INIT_RANGE, size = self.n)
        self.x_dot = self.rng.uniform(*c.X_DOT_INIT_RANGE, size = self.n)
        self.alive = np.ones(self.n, dtype=bool)

    def step(self, forces):
        forces = np.where(self.alive, forces, 0.0)
        x_ddot = forces/self.m

        # 3. pole angular acceleration -- your corrected CCW equation
        theta_ddot = x_ddot/self.l*np.cos(self.theta)+ self.g/self.l*np.sin(self.theta)-self.theta_dot*(1-self.damp)

        # 4. integrate: velocity first, then position, using dt
        new_x_dot = self.x_dot + x_ddot*self.dt
        new_x = self.x + new_x_dot*self.dt
        new_theta_dot = self.theta_dot + theta_ddot*self.dt
        new_theta = self.theta + new_theta_dot*self.dt

        # 5. only apply the update where alive -- frozen pendulums keep old state
        self.x_dot = np.where(self.alive, new_x_dot, self.x_dot)
        self.x = np.where(self.alive, new_x, self.x)
        self.theta_dot = np.where(self.alive, new_theta_dot, self.theta_dot)
        self.theta = np.where(self.alive, new_theta, self.theta)

        # 6. recompute alive -- once dead, stays dead
        still_ok = (abs(self.theta) < self.theta_threshold) & (abs(self.x) < self.x_threshold)
        self.alive = self.alive & still_ok
