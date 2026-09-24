"""Train a population on the GPU, save it, then animate the best controllers."""
import torch

from mlp import config as c
from mlp.evolution import load_population, save_population, train
from mlp.simulation import DEVICE
from mlp.visualize import animate_best, live_fitness_plot


def main(debug=False):
    if debug:
        device_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU (no CUDA found)"
        print(f"Device: {device_name}")
        print(f"{c.POP_SIZE} individuals x {c.N_SIMULATIONS} episodes = "
              f"{c.POP_SIZE * c.N_SIMULATIONS:,} cart-poles per generation")

    population = load_population()
    plot = live_fitness_plot()
    population = train(population, c.N_GENERATIONS, on_generation=plot)

    save_population(population)
    print(f"Saved to {c.POPULATION_FILE}")

    animate_best(population)


if __name__ == "__main__":
    main()
