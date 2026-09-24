import math

# --- device ---
DEVICE = "cuda"                     # "cuda" = GPU, "cpu" = test without GPU

# --- physics ---
GRAVITY = 9.81
CART_MASS = 0.2
POLE_LENGTH = 1.0
POLE_DAMPING = 0.0
CART_FRICTION = 0.0
FORCE_MAG = 2.0
DT = 0.02                           # seconds per simulation step
THETA_THRESHOLD = math.inf          # radians (inf = pole may swing freely)
X_THRESHOLD = 2.4                   # meters (cart dies outside +-X_THRESHOLD)

# --- start states (ranges for each episode) ---
THETA_INIT_RANGE = (-3.14, 3.14)
THETA_DOT_INIT_RANGE = (0.0, 0.0)
X_INIT_RANGE = (0.0, 0.0)
X_DOT_INIT_RANGE = (0.0, 0.0)

# --- episode ---
MAX_STEPS = 500
EARLY_EXIT_CHECK = 50

# --- network ---
NETWORK_LAYOUT = [5, 10, 10, 1]     # [inputs, hidden..., output]

# --- evolution ---
POP_SIZE = 8192                     # individuals per generation
N_SIMULATIONS = 32                  # episodes per individual (same start angles for everyone)
N_GENERATIONS = int(1e6)
ELITE_FRAC = 0.3                    # best fraction kept unchanged
RANDOM_FRAC = 0.1                   # fraction of brand-new random individuals
MUTATION_STD = 0.3                  # noise size for children
DECAY_RATE = 0.999                  # mutation shrinks by this factor per generation

# --- stop criteria ---
QUALITY_BEST = 0.99
QUALITY_MEAN = 0.8

# --- files ---
SAVE_EVERY = 25                     # save population every N generations
POPULATION_FILE = "population.npy"

# --- visualization ---
PLOT_EVERY = 1                      # redraw fitness plot every N generations
N_SHOW = 50                         # how many of the best controllers to animate
PLOT_SIZE = (12, 6)
ALPHA = 0.2                         # transparency of the other carts
MARKER_SIZE = 15
INTERVAL_MS = DT * 1000             # ms per animation frame -> real time
KICK = 1.0                          # velocity added by the arrow keys
