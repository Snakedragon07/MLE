# MLE: Cart-pole swing-up with neuroevolution

A population of small neural networks (MLPs) learns to swing up and balance an
inverted pendulum on a cart. Training uses a simple evolutionary algorithm:
the best controllers are kept, mutated copies fill the population, and a few
random newcomers keep exploration going.

## Project structure

```
MLE/
├── train.py          # train a population, save it, then animate the best ones
├── play.py           # load a saved population and animate it (no training)
├── requirements.txt
└── mle/              # the package
    ├── config.py     # physics, network, evolution and visualization settings
    ├── environment.py# batched cart-pole physics
    ├── policy.py     # MLP controller
    ├── evolution.py  # population, fitness, selection/mutation, training loop
    └── visualize.py  # live fitness plot and cart-pole animation
```

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python train.py   # train (Esc in the fitness plot stops training)
python play.py    # watch a saved population
```

In the animation, the **left / right arrow keys** give all carts a kick.
The best controller is drawn in black.

Settings live in `mle/config.py`. The trained population is saved as
`population.npy` (not tracked by git).
