import config as c

import numpy as np

def P(env):
    P_coeff = 0.7
    return  2 * (env.theta-np.pi) * P_coeff * c.FORCE_MAG / np.pi

def PI(env):
    P_coeff = 0.7
    I_coeff = 0.2

