import numpy as np

# --- physics ---
GRAVITY = 9.81
CART_MASS = 0.2
POLE_LENGTH = 1
DAMPING_COEFF = 0.5

# --- actuation ---
FORCE_MAG = 2.0
DT = 0.02

# --- failure thresholds ---
THETA_THRESHOLD = np.inf   # radians
X_THRESHOLD = 2.4       # meters

# --- initial state ranges (used in reset()) ---
THETA_INIT_RANGE = (-3.14,3.14)      # e.g. (-x, x) radians
THETA_DOT_INIT_RANGE = (0,0)
X_INIT_RANGE = (0,0)
X_DOT_INIT_RANGE = (0,0)

# --- simulation ---
N_PENDULUMS = 50       # how many run in parallel
MAX_STEPS = 500          # episode length before it auto-ends
SEED = None

# --- visualization ---
plotsize = (12,6)
alpha = 0.2
markersize = 15
fps = 60
interval = 1000/fps

#--- MLP ---
#[5,x,..,y,1]
NETWORK_LAYOUT = [5,8,8,1]

#--- Evolution ---
elite_frac = 0.3
mutation_std = 0.5
mutation_rate = 0.2  # fraction of genes perturbed per offspring
random_frac = 0.1
N_GENERATIONS = 500
N_Simulations = 8

POPULATION_FILE = "population.npy"