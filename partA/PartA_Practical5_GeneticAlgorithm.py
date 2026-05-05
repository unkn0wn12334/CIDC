# ============================================================
# FILE: PartA_Practical5_GeneticAlgorithm.py
# STANDALONE FILE — No other files needed.
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# STEP 1 — Install required libraries (only once):
#   pip install numpy matplotlib
#
# OPTION A — Jupyter Notebook / Google Colab (RECOMMENDED):
#   1. Open Jupyter or go to colab.research.google.com
#   2. Paste entire code into a cell
#   3. Press Shift+Enter
#   4. Watch generation-by-generation output print below the cell:
#      Gen  1: Best Fitness = X.XXXX, Best x = X.XXXX, Avg = X.XXXX
#      Gen 10: ...
#      ...
#      Gen 100: ...
#   5. Two plots appear inline:
#      Left: The fitness function curve with GA's best x marked in red
#      Right: Convergence graph (fitness improving over generations)
#   6. 'genetic_algorithm.png' saved in current directory
#
# OPTION B — PyCharm:
#   1. Open this file → Install numpy/matplotlib if prompted
#   2. Click ▶ Run
#   3. Generation logs print in Run console
#   4. Plot opens as a popup window after all 100 generations
#
# OPTION C — Terminal:
#   python PartA_Practical5_GeneticAlgorithm.py
#
# EXPECTED OUTPUT:
#   - Generation logs every 10 generations showing best fitness improving
#   - Final best x value found (should be near 7.9 where x*sin(x)+1 is max)
#   - Final best fitness value
#   - Two-panel plot saved as genetic_algorithm.png
#
# TO CHANGE PARAMETERS: Edit the genetic_algorithm() call at the bottom:
#   pop_size=50        → number of individuals in population
#   num_generations=100 → how many evolution cycles to run
#   crossover_rate=0.8 → probability of crossover occurring
#   mutation_rate=0.01 → probability of each bit flipping
# ============================================================

# Import random for random number generation (mutation, crossover, selection)
import random
# Import numpy for numerical operations
import numpy as np
# Import matplotlib for visualizing GA convergence over generations
import matplotlib.pyplot as plt
# Import copy for deep copying individuals to avoid reference issues
import copy

# ============================================================
# STEP 1: PROBLEM DEFINITION AND FITNESS FUNCTION
# ============================================================
# We are optimizing a simple function: f(x) = x * sin(x) + 1 in range [0, 10]
# In the coconut milk spray drying context, this represents optimizing
# parameters like inlet temperature, feed rate, atomization pressure, etc.
# GA will find the x value that maximizes f(x)

def fitness_function(x):
    """
    Fitness function to be MAXIMIZED by the Genetic Algorithm.
    Higher return value = better individual = higher survival probability.
    In spray drying context: evaluates quality of parameter combination.
    """
    # Compute x * sin(x) + 1 — a non-linear multimodal function
    return x * np.sin(x) + 1  # Returns fitness value for individual x


# ============================================================
# STEP 2: ENCODING — Represent solutions as binary chromosomes
# ============================================================
def encode(x, x_min=0, x_max=10, num_bits=16):
    """
    Encodes a real value x into a binary chromosome (list of 0s and 1s).
    Genotype (binary string) represents the Phenotype (real value).
    num_bits: precision of encoding — more bits = higher precision
    """
    # Scale x to integer range based on number of bits
    # Formula: integer = (x - x_min) / (x_max - x_min) * (2^num_bits - 1)
    max_int = 2 ** num_bits - 1  # Maximum integer that can be represented
    int_val = int((x - x_min) / (x_max - x_min) * max_int)  # Scale to integer
    # Convert integer to binary string with leading zeros
    binary_str = format(int_val, f'0{num_bits}b')
    # Convert binary string to list of integers [0, 1, 0, 1, ...]
    return [int(bit) for bit in binary_str]

def decode(chromosome, x_min=0, x_max=10, num_bits=16):
    """
    Decodes a binary chromosome back to a real value x.
    Converts binary list → integer → scaled real value.
    """
    # Convert binary list to integer value
    binary_str = ''.join(map(str, chromosome))  # Join bits into string "01001..."
    int_val = int(binary_str, 2)  # Convert binary string to decimal integer
    # Scale integer back to real value range [x_min, x_max]
    max_int = 2 ** num_bits - 1
    x = x_min + (int_val / max_int) * (x_max - x_min)  # Reverse the encoding formula
    return x


# ============================================================
# STEP 3: INITIALIZE POPULATION
# ============================================================
def initialize_population(pop_size, num_bits=16, x_min=0, x_max=10):
    """
    Creates the initial random population of individuals.
    Each individual is a binary chromosome representing a solution.
    Population size determines diversity and search coverage.
    """
    population = []  # Empty list to hold all individuals
    for _ in range(pop_size):
        # Generate a random x value in the allowed range
        x = random.uniform(x_min, x_max)
        # Encode x as a binary chromosome
        chromosome = encode(x, x_min, x_max, num_bits)
        # Add this individual to the population
        population.append(chromosome)
    return population  # Returns list of binary chromosomes


# ============================================================
# STEP 4: SELECTION — Roulette Wheel (Fitness Proportionate)
# ============================================================
def roulette_wheel_selection(population, fitnesses):
    """
    Selects individuals based on their relative fitness.
    Higher fitness = higher probability of being selected.
    Like spinning a roulette wheel where better individuals have larger sectors.
    """
    # Calculate total fitness of entire population
    total_fitness = sum(fitnesses)

    # Avoid division by zero if all fitnesses are 0
    if total_fitness == 0:
        return random.choice(population)

    # Normalize fitnesses to get selection probabilities (must sum to 1)
    probabilities = [f / total_fitness for f in fitnesses]

    # Use numpy's choice with weighted probabilities to select one individual
    # np.random.choice selects index based on probabilities
    selected_idx = np.random.choice(len(population), p=probabilities)
    return population[selected_idx]  # Return the selected chromosome


# ============================================================
# STEP 5: CROSSOVER — Single Point Crossover
# ============================================================
def single_point_crossover(parent1, parent2, crossover_rate=0.8):
    """
    Creates two offspring by swapping genetic material between two parents.
    Single point: one random point divides the chromosome into two halves.
    Crossover rate: probability that crossover actually happens.
    """
    # Only perform crossover with probability = crossover_rate
    if random.random() < crossover_rate:
        # Choose a random crossover point (not at the very start or end)
        point = random.randint(1, len(parent1) - 1)

        # Create child1: head of parent1 + tail of parent2
        child1 = parent1[:point] + parent2[point:]
        # Create child2: head of parent2 + tail of parent1
        child2 = parent2[:point] + parent1[point:]
    else:
        # If no crossover, children are exact copies of parents
        child1 = parent1.copy()
        child2 = parent2.copy()

    return child1, child2  # Return two offspring chromosomes


# ============================================================
# STEP 6: MUTATION — Bit Flip Mutation
# ============================================================
def mutate(chromosome, mutation_rate=0.01):
    """
    Randomly flips bits in a chromosome with small probability.
    Mutation introduces diversity and prevents premature convergence.
    Low mutation rate: preserves good solutions; high rate: too random.
    """
    mutated = chromosome.copy()  # Copy chromosome to avoid modifying original
    for i in range(len(mutated)):
        # For each bit, flip it with probability = mutation_rate
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]  # Flip: 0→1 or 1→0
    return mutated  # Return the (possibly) mutated chromosome


# ============================================================
# STEP 7: MAIN GENETIC ALGORITHM LOOP
# ============================================================
def genetic_algorithm(pop_size=50, num_generations=100,
                       crossover_rate=0.8, mutation_rate=0.01,
                       x_min=0, x_max=10, num_bits=16):
    """
    Main GA loop implementing the full evolutionary cycle:
    Initialize → Evaluate → Select → Crossover → Mutate → Replace → Repeat
    """
    print("="*60)
    print("GENETIC ALGORITHM — Starting Optimization")
    print(f"Population Size: {pop_size}, Generations: {num_generations}")
    print(f"Crossover Rate: {crossover_rate}, Mutation Rate: {mutation_rate}")
    print("="*60)

    # INITIALIZATION: Create random starting population
    population = initialize_population(pop_size, num_bits, x_min, x_max)

    # Lists to track best fitness over generations (for convergence plot)
    best_fitness_history = []
    avg_fitness_history = []
    best_solution_history = []

    best_individual = None  # Track the overall best individual found
    best_fitness_ever = float('-inf')  # Track the highest fitness seen

    # MAIN EVOLUTIONARY LOOP — runs for num_generations iterations
    for generation in range(num_generations):

        # EVALUATION: Compute fitness for each individual
        fitnesses = []
        for chromosome in population:
            x = decode(chromosome, x_min, x_max, num_bits)  # Decode to real value
            fitness = fitness_function(x)  # Evaluate fitness function
            # Use max(0, fitness) to handle negative fitness values
            fitnesses.append(max(0, fitness))

        # Find best individual in current generation
        max_idx = np.argmax(fitnesses)  # Index of best individual
        gen_best_fitness = fitnesses[max_idx]  # Best fitness in this generation
        gen_best_x = decode(population[max_idx], x_min, x_max, num_bits)  # Decode best x

        # Update overall best if current generation has better individual
        if gen_best_fitness > best_fitness_ever:
            best_fitness_ever = gen_best_fitness
            best_individual = population[max_idx].copy()  # Save best chromosome
            best_x = gen_best_x  # Save best x value

        # Track history for plotting
        best_fitness_history.append(gen_best_fitness)
        avg_fitness_history.append(np.mean(fitnesses))

        # Print progress every 10 generations
        if generation % 10 == 0 or generation == num_generations - 1:
            print(f"Gen {generation+1:>3}: Best Fitness = {gen_best_fitness:.4f}, "
                  f"Best x = {gen_best_x:.4f}, Avg Fitness = {np.mean(fitnesses):.4f}")

        # CREATE NEXT GENERATION through selection, crossover, mutation
        new_population = []

        # ELITISM: Keep the best individual unchanged (preserves best solution)
        new_population.append(population[max_idx].copy())

        # Fill rest of new population
        while len(new_population) < pop_size:
            # SELECTION: Pick two parents using roulette wheel
            parent1 = roulette_wheel_selection(population, fitnesses)
            parent2 = roulette_wheel_selection(population, fitnesses)

            # CROSSOVER: Produce two children from two parents
            child1, child2 = single_point_crossover(parent1, parent2, crossover_rate)

            # MUTATION: Randomly flip bits in children
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            # Add children to new population
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # REPLACEMENT: New population replaces old population
        population = new_population

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    best_x_final = decode(best_individual, x_min, x_max, num_bits)
    best_fitness_final = fitness_function(best_x_final)
    print(f"Best x found: {best_x_final:.6f}")
    print(f"Best fitness f(x) = x*sin(x)+1: {best_fitness_final:.6f}")
    print(f"Best chromosome (first 16 bits): {''.join(map(str, best_individual[:16]))}")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Fitness function and found optimum
    x_range = np.linspace(x_min, x_max, 500)  # 500 points across the range
    y_range = [fitness_function(xi) for xi in x_range]  # Compute fitness for all
    axes[0].plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = x·sin(x)+1')
    axes[0].axvline(x=best_x_final, color='red', linestyle='--', linewidth=2,
                    label=f'GA Solution: x={best_x_final:.3f}')
    axes[0].scatter([best_x_final], [best_fitness_final], color='red', s=150, zorder=5)
    axes[0].set_title('Fitness Function and GA Solution', fontsize=13)
    axes[0].set_xlabel('x (Parameter Value)')
    axes[0].set_ylabel('f(x) (Fitness)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Convergence — best and average fitness over generations
    gen_range = range(1, num_generations + 1)  # x-axis: generation numbers
    axes[1].plot(gen_range, best_fitness_history, 'g-', linewidth=2, label='Best Fitness')
    axes[1].plot(gen_range, avg_fitness_history, 'orange', linewidth=2,
                 linestyle='--', label='Average Fitness')
    axes[1].set_title('GA Convergence Over Generations', fontsize=13)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Fitness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Genetic Algorithm Optimization', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('genetic_algorithm.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'genetic_algorithm.png'")

    return best_x_final, best_fitness_final

# Run the Genetic Algorithm
if __name__ == "__main__":
    best_x, best_fit = genetic_algorithm(
        pop_size=50,          # Population of 50 individuals
        num_generations=100,  # Run for 100 generations
        crossover_rate=0.8,   # 80% chance of crossover
        mutation_rate=0.01,   # 1% chance of each bit flipping
        x_min=0,              # Minimum value of search space
        x_max=10,             # Maximum value of search space
        num_bits=16           # 16-bit chromosome precision
    )

# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. ENCODING: Real values encoded to binary chromosomes (Genotype)
# 2. INITIALIZATION: Random population of 50 binary chromosomes created
# 3. EVALUATION: Each chromosome decoded and fitness_function evaluated
# 4. SELECTION: Roulette wheel selects parents proportional to fitness
# 5. CROSSOVER: Single-point crossover creates new child chromosomes
# 6. MUTATION: Random bit flips introduce diversity (prevent local optima)
# 7. REPLACEMENT: New generation replaces old (with elitism — best preserved)
# 8. REPEAT: Steps 3-7 repeat for 100 generations
# 9. RESULT: Best chromosome decoded to get optimal x value
# Elitism ensures best solution never regresses between generations
# Convergence plot shows fitness improving and stabilizing over time
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: Genetic Algorithms (GA)
# GA is an optimization algorithm inspired by biological evolution.
# Key biological analogy:
#   Chromosome = Candidate solution (encoded as binary string)
#   Gene = Individual bit in the chromosome
#   Population = Set of candidate solutions
#   Fitness = How good a solution is (quality measure)
#   Selection = Survival of the fittest (better solutions reproduce more)
#   Crossover = Recombination of parents to create offspring
#   Mutation = Random changes to maintain diversity
# GA is useful for non-convex, discontinuous, high-dimensional optimization.
# Application in coconut milk spray drying: Find optimal inlet temperature,
# outlet temperature, feed rate combination that maximizes product quality.
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. Define Phenotype and Genotype in GA.
# A1. Genotype: The encoded representation of a solution — the binary chromosome.
#     This is what the GA actually manipulates (crossover, mutation applied here).
#     Phenotype: The actual solution in the problem domain — the decoded real value.
#     Example: Genotype = "0101110010001100", Phenotype = x = 3.67 (temperature)
#     GA works in genotype space; fitness is evaluated in phenotype space.
#
# Q2. What is Encoding and Decoding? Explain its techniques.
# A2. Encoding: Converting a solution (phenotype) into a chromosome (genotype).
#     Techniques: Binary encoding (most common), Gray code, Real-value encoding,
#     Permutation encoding (for TSP), Tree encoding (for genetic programming).
#     Decoding: Reverse process — converting chromosome back to solution.
#     Binary: map binary string to real number using scaling formula.
#
# Q3. Define Population, Genes, and Fitness Function in GA.
# A3. Population: Set of all current candidate solutions (chromosomes).
#     Size matters: too small = low diversity; too large = slow convergence.
#     Gene: A single unit/position in a chromosome (a bit in binary encoding).
#     Fitness Function: Evaluates how good a solution is. GA maximizes this.
#     Must reflect the actual problem objective. E.g., f(x) = x*sin(x)+1.
#
# Q4. Explain GA with its architecture (phases).
# A4. GA architecture:
#     1. Initialization: Create random population
#     2. Evaluation: Compute fitness for all individuals
#     3. Selection: Choose parents (roulette wheel, tournament, rank-based)
#     4. Crossover: Combine parents to create offspring (single/two-point/uniform)
#     5. Mutation: Random small changes to offspring
#     6. Replacement: Form new population (generational or steady-state)
#     7. Termination Check: Stop if max generations or fitness threshold reached
#     8. Return best solution found
#
# Q5. What is Crossover? Explain its types.
# A5. Crossover combines genetic material from two parents to create offspring.
#     Types:
#     (i) Single-Point: One cut point; swap tails of two parents
#     (ii) Two-Point: Two cut points; swap middle section
#     (iii) Uniform: Each bit inherited from parent1 or parent2 with 50% probability
#     (iv) Arithmetic: Child = α*parent1 + (1-α)*parent2 (for real-valued)
#     Crossover rate (0.6-0.9) controls how often it occurs.
#
# Q6. What is Mutation in GA? Why is it important?
# A6. Mutation randomly flips bits in a chromosome with a small probability.
#     Importance:
#     - Introduces genetic diversity lost through selection
#     - Allows exploration of new regions in the search space
#     - Prevents premature convergence to local optima
#     - Recovers genetic material that might have been lost
#     Mutation rate too high: Random search (loses good solutions)
#     Mutation rate too low: Premature convergence. Typical rate: 0.001 to 0.05
#
# Q7. What is Selection in GA? What are its types?
# A7. Selection chooses which individuals reproduce (survive to next generation).
#     Types:
#     (i) Roulette Wheel: Selection probability proportional to fitness
#     (ii) Tournament: Pick k random individuals, best wins
#     (iii) Rank Selection: Rank individuals, probability based on rank
#     (iv) Elitism: Best individuals always survive to next generation
#
# Q8. What are advantages of Genetic Algorithms over traditional optimization?
# A8. (i) No need for gradient information — works with any fitness function
#     (ii) Handles multimodal functions (multiple peaks) better
#     (iii) Works with discrete, continuous, or mixed variables
#     (iv) Less likely to get stuck in local optima (diversity maintained)
#     (v) Naturally parallelizable — evaluate population members simultaneously
#     (vi) Robust to noise in fitness evaluation
#
# Q9. What is the Stopping Criteria in GA?
# A9. GA stops when one of these conditions is met:
#     (i) Maximum number of generations reached
#     (ii) Fitness threshold achieved (good enough solution found)
#     (iii) Population converged (all individuals are similar)
#     (iv) No improvement for many consecutive generations
#     (v) Computational budget exhausted (time limit)
#
# Q10. What is the schema theorem in GA?
# A10. Schema theorem (Holland's theorem) explains why GA works:
#      A schema is a pattern of bits (* = don't care, e.g., "1**0*1").
#      Short, above-average fitness schemas grow exponentially over generations.
#      Building Block Hypothesis: GA combines small, high-fitness sub-patterns
#      (building blocks) via crossover to construct globally optimal solutions.
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# Genetic Algorithm is a population-based metaheuristic inspired by Darwinian evolution.
# It is powerful for non-convex, discontinuous, noisy, black-box optimization.
#
# Core GA cycle:
#   Initialize -> Evaluate -> Select -> Crossover -> Mutate -> Replace -> Repeat.
#
# Why GA works:
#   - Selection exploits good candidates.
#   - Crossover recombines useful traits.
#   - Mutation injects novelty and escapes local optima.
#   - Population keeps multiple hypotheses simultaneously.
#
# Important parameter intuition:
#   - Population too small -> premature convergence.
#   - Mutation too low -> stagnation.
#   - Mutation too high -> random search.
#   - Crossover too low -> slow mixing.
#
# Constraint handling techniques:
#   - Penalty function, repair operator, feasibility-based selection.
#
# CI connection:
#   GA belongs to evolutionary computation family with ES, DE, GP.
#
# Exam-ready line:
# "GA is preferred when objective is complex/non-differentiable and gradient methods fail."
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. What is elitism and why important?
# A11. Elitism copies top individuals directly to next generation, preventing loss of
#      best solutions due to stochastic crossover/mutation.
#
# Q12. What is tournament selection advantage?
# A12. It is simple, fast, and tunable selection pressure via tournament size.
#
# Q13. How does GA handle multimodal fitness landscapes?
# A13. Population diversity allows simultaneous exploration of multiple peaks,
#      reducing chance of getting stuck at one local optimum.
#
# Q14. What is real-coded GA?
# A14. Individuals store real values directly (not binary), useful for continuous
#      optimization and often improves precision/convergence.
#
# Q15. When should GA not be preferred?
# A15. For small convex problems with known derivatives, deterministic methods are
#      usually faster and more precise.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) GA SEARCH PHILOSOPHY
#    - Uses population, not a single solution.
#    - Works even when objective is non-smooth/non-differentiable.
#
# 2) REPRESENTATION
#    - Binary encoding maps real x <-> bit string.
#    - Precision depends on number of bits.
#
# 3) EVOLUTION OPERATORS
#    - Selection favors fitter solutions.
#    - Crossover mixes genetic material from parents.
#    - Mutation introduces new traits and prevents stagnation.
#
# 4) CONVERGENCE BEHAVIOR
#    - Early stage: exploration (high diversity).
#    - Later stage: exploitation (refining best region).
#    - Elitism preserves best candidate each generation.
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "This GA encodes candidate x values as chromosomes, evaluates f(x), evolves
#     population using selection-crossover-mutation, and converges to near-optimal x."
# ============================================================
