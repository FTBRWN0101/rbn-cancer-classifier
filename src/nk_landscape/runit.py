import numpy as np
from .nk_landscape import NKLandscape
from .fitness_evaluator import FitnessEvaluator
from .SA import SimulatedAnnealing
from .ruggedness import autocorrelation

def main():
    print("\nDefining NK landscape")
    n = 20
    k = 2
    random_state = 42
    nk_landscape = NKLandscape(n, k, random_state=random_state)

    print("\nEvaluating fitness")
    genome = np.random.randint(0, 2, size=n)
    fitness_evaluator = FitnessEvaluator(nk_landscape)
    fitness = fitness_evaluator.evaluate(genome)
    print(f"Initial fitness: {fitness}")

    print("\nRunning Simulated Annealing")
    sa = SimulatedAnnealing(fitness_evaluator, initial_temperature=1.0, cooling_rate=0.995, steps=1000)
    optimized_genome = sa.optimise(genome)

    optimized_fitness = fitness_evaluator.evaluate(optimized_genome)
    print(f"Optimized fitness: {optimized_fitness}")

    print("\nEvaluating ruggedness...")
    fitness_values = [fitness_evaluator.evaluate(np.random.randint(0, 2, size=n)) for _ in range(100)]
    ruggedness = autocorrelation(np.array(fitness_values))
    print(f"Ruggedness (autocorrelation): {ruggedness}")

if __name__ == "__main__":
    main()
