# --- physics ---
GRAVITY = 9.81
CART_MASS = 0.2
POLE_LENGTH = 0.3
DAMPING_COEFF = 1.0

# --- actuation ---
FORCE_MAG = 1.0
DT = 0.02

# --- failure thresholds ---
THETA_THRESHOLD = 0.21   # radians
X_THRESHOLD = 2.4       # meters

# --- initial state ranges (used in reset()) ---
THETA_INIT_RANGE = (-0.05,0.05)      # e.g. (-x, x) radians
THETA_DOT_INIT_RANGE = (0,0)
X_INIT_RANGE = (0,0)
X_DOT_INIT_RANGE = (0,0)

# --- simulation ---
N_PENDULUMS = 25        # how many run in parallel
MAX_STEPS = 500          # episode length before it auto-ends
SEED = None