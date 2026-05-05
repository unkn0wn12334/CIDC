# ============================================================
# FILE: PartA_Practical5_GeneticAlgorithm.py
# SUBJECT: Computational Intelligence (CI) — Part A, Assignment 5
# TOPIC: Genetic Algorithm — Optimization (Spray Drying of Coconut Milk)
# STANDALONE FILE — No other files needed.
#
# ── HOW TO RUN ────────────────────────────────────────────
# STEP 1 — Install required libraries (only once):
#   pip install numpy matplotlib pandas scikit-learn requests
#
# OPTION A — Google Colab (RECOMMENDED):
#   1. Go to colab.research.google.com → New Notebook
#   2. Cell 1: !pip install numpy matplotlib pandas scikit-learn requests
#   3. Paste entire code into Cell 2 → Shift+Enter
#
# OPTION B — PyCharm / Terminal:
#   python PartA_Practical5_GeneticAlgorithm.py
#
# EXPECTED OUTPUT:
#   - Dataset loaded (real or synthetic with clear label)
#   - Generation-by-generation GA log showing fitness improving
#   - Best x value found (optimized parameter)
#   - Two plots: fitness function curve + convergence graph
#   - 'genetic_algorithm.png' saved
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import random
import copy

# ================================================================
# ██████████████████████████████████████████████████████████████
#          REAL DATASET — SPRAY DRYING COCONUT MILK DATA
#  Source: UCI Machine Learning Repository / Kaggle open datasets
#  We use the "Dry Bean Dataset" (UCI) as a real parameter-optimization
#  proxy — it contains real measured attributes that we optimize using GA.
#  For the spray drying context the GA optimizes a parameter x (e.g. inlet
#  temperature 140–200 °C) to maximise a quality proxy function.
# ██████████████████████████████████████████████████████████████
# ================================================================

# def load_real_dataset():
#     """
#     Loads the UCI Dry Bean Dataset via URL.
#     The dataset has 13 numeric features (area, perimeter, etc.) from 13611 bean images.
#     We use Feature 0 (Area) rescaled to [0,10] as our GA search space proxy —
#     representing a continuous process parameter (e.g. temperature 0→10 normalised).

#     URL source:
#       https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip
#     We use a CSV mirror that is always available:
#       https://raw.githubusercontent.com/dsrscientist/dataset1/master/dry_bean.csv

#     Returns: numpy array of the first numeric column, scaled to [0, 10]
#     """
#     import pandas as pd
#     import requests, io

#     # ── PRIMARY: URL-based loading ──────────────────────────────
#     URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/dry_bean.csv"
#     try:
#         print("Fetching real dataset from URL ...")
#         resp = requests.get(URL, timeout=15)
#         resp.raise_for_status()
#         df = pd.read_csv(io.StringIO(resp.text))
#         # Select the first numeric column (Area) and scale to [0, 10]
#         col = df.select_dtypes(include=[np.number]).iloc[:, 0].dropna().values
#         col_scaled = (col - col.min()) / (col.max() - col.min()) * 10
#         print(f"  ✓ Real dataset loaded: {len(col_scaled)} samples, "
#               f"feature '{df.select_dtypes(include=[np.number]).columns[0]}' scaled to [0,10]")
#         return col_scaled
#     except Exception as e:
#         print(f"  ✗ URL load failed ({e}). Trying local file ...")

#     # ── SECONDARY: Local file (comment/uncomment as needed) ────
#     # LOCAL_PATH = r"C:\Users\YourName\Downloads\dry_bean.csv"
#     # try:
#     #     df = pd.read_csv(LOCAL_PATH)
#     #     col = df.select_dtypes(include=[np.number]).iloc[:, 0].dropna().values
#     #     col_scaled = (col - col.min()) / (col.max() - col.min()) * 10
#     #     print(f"  ✓ Local dataset loaded: {len(col_scaled)} samples")
#     #     return col_scaled
#     # except Exception as e2:
#     #     print(f"  ✗ Local load also failed ({e2}). Falling back to synthetic.")

#     return None   # Signal to caller that real data is unavailable
def load_real_dataset():
    import pandas as pd
    import numpy as np
    import requests, zipfile, io

    URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"

    try:
        print("Downloading dataset from UCI ...")
        resp = requests.get(URL, timeout=20)
        resp.raise_for_status()

        # Extract ZIP in memory
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        file_name = [f for f in z.namelist() if f.endswith(".xlsx")][0]

        # Read Excel file
        df = pd.read_excel(z.open(file_name))

        # Select first numeric column
        col = df.select_dtypes(include=[np.number]).iloc[:, 0].dropna().values

        # Scale to [0,10]
        col_scaled = (col - col.min()) / (col.max() - col.min()) * 10

        print(f"✓ Loaded {len(col_scaled)} samples from UCI dataset")
        return col_scaled

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None

# ── Build fitness landscape from dataset ────────────────────────
def build_fitness_from_data(data_values, n_bins=100):
    """
    Creates a smooth empirical fitness function by binning real data values.
    The idea: data density (normalised histogram) acts as the "quality surface" GA optimises.
    In spray drying terms: regions where real experiments clustered = high-quality parameter zones.

    Returns a callable fitness_fn(x) → float
    """
    counts, bin_edges = np.histogram(data_values, bins=n_bins, density=True)
    def empirical_fitness(x):
        # Find which bin x falls into and return its normalised density
        idx = np.searchsorted(bin_edges[1:], x)
        idx = np.clip(idx, 0, len(counts) - 1)
        return float(counts[idx])
    return empirical_fitness


# ================================================================
# ██████████████████████████████████████████████████████████████
#         SYNTHETIC DATA (FALLBACK — clearly labelled)
#  Used ONLY when the real dataset cannot be fetched.
#  The synthetic fitness function is a well-known multimodal test function.
# ██████████████████████████████████████████████████████████████
# ================================================================

def synthetic_fitness_function(x):
    """
    ── SYNTHETIC FALLBACK ──
    f(x) = x * sin(x) + 1  on [0, 10]
    Represents a hypothetical process-quality curve with multiple local maxima.
    Used only if the real dataset cannot be loaded.
    """
    return x * np.sin(x) + 1


# ================================================================
# ──────────────────────────────────────────────────────────────
#   GA CORE — Encoding, Population, Selection, Crossover, Mutation
# ──────────────────────────────────────────────────────────────
# ================================================================

def encode(x, x_min=0, x_max=10, num_bits=16):
    """
    Converts a real-valued parameter x into a binary chromosome (list of 0/1).
    This is the GENOTYPE — what the GA manipulates.
    The corresponding real value is the PHENOTYPE.

    Formula: int_val = (x - x_min)/(x_max - x_min) * (2^num_bits - 1)
    Then convert int_val to binary string of length num_bits.
    """
    max_int = 2 ** num_bits - 1
    int_val = int((x - x_min) / (x_max - x_min) * max_int)
    return [int(b) for b in format(int_val, f'0{num_bits}b')]


def decode(chromosome, x_min=0, x_max=10, num_bits=16):
    """
    Converts a binary chromosome back to a real-valued parameter.
    Reverse of encode().
    """
    int_val = int(''.join(map(str, chromosome)), 2)
    return x_min + (int_val / (2 ** num_bits - 1)) * (x_max - x_min)


def initialize_population(pop_size, num_bits=16, x_min=0, x_max=10):
    """
    Creates a random starting population.
    Each individual = random real x → encoded to binary chromosome.
    More individuals = better coverage of search space but slower per generation.
    """
    return [encode(random.uniform(x_min, x_max), x_min, x_max, num_bits)
            for _ in range(pop_size)]


def roulette_wheel_selection(population, fitnesses):
    """
    Fitness-proportionate selection (roulette wheel).
    Higher fitness → larger "slice" of the wheel → higher chance of selection.
    Ensures good individuals reproduce more but doesn't completely exclude weaker ones.
    """
    total = sum(fitnesses)
    if total == 0:
        return random.choice(population)
    probs = [f / total for f in fitnesses]
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


def single_point_crossover(p1, p2, rate=0.8):
    """
    Single-point crossover: one random cut point splits both parent chromosomes.
    child1 = head(p1) + tail(p2)
    child2 = head(p2) + tail(p1)
    Only happens with probability = rate (crossover_rate).
    No crossover → children are exact copies of parents.
    """
    if random.random() < rate:
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1.copy(), p2.copy()


def mutate(chromosome, rate=0.01):
    """
    Bit-flip mutation: each bit independently flipped with probability = rate.
    Maintains genetic diversity and prevents premature convergence.
    Rate too high → random search. Rate too low → stagnation.
    Typical: 0.001 – 0.05 (inversely proportional to chromosome length).
    """
    return [b if random.random() >= rate else 1 - b for b in chromosome]


# ================================================================
#   MAIN GA LOOP
# ================================================================

def genetic_algorithm(fitness_fn, x_min=0, x_max=10,
                       pop_size=50, num_generations=100,
                       crossover_rate=0.8, mutation_rate=0.01,
                       num_bits=16):
    """
    Main Genetic Algorithm optimisation loop.

    Phases per generation:
      1. Evaluate    — compute fitness for each individual
      2. Select      — roulette-wheel selection picks parents
      3. Crossover   — produce offspring by swapping chromosome halves
      4. Mutate      — randomly flip bits for diversity
      5. Replace     — new population replaces old (with elitism)
    """
    print("="*60)
    print("GENETIC ALGORITHM — Optimisation")
    print(f"Pop={pop_size}, Gens={num_generations}, "
          f"Cx={crossover_rate}, Mut={mutation_rate}")
    print("="*60)

    population = initialize_population(pop_size, num_bits, x_min, x_max)

    best_fitness_history, avg_fitness_history = [], []
    best_individual, best_fitness_ever = None, float('-inf')

    for gen in range(num_generations):
        # ── Evaluate ─────────────────────────────────────────
        fitnesses = []
        for chrom in population:
            x = decode(chrom, x_min, x_max, num_bits)
            fitnesses.append(max(0.0, fitness_fn(x)))

        # Track best
        max_idx = int(np.argmax(fitnesses))
        if fitnesses[max_idx] > best_fitness_ever:
            best_fitness_ever = fitnesses[max_idx]
            best_individual = population[max_idx].copy()

        best_fitness_history.append(fitnesses[max_idx])
        avg_fitness_history.append(float(np.mean(fitnesses)))

        if gen % 10 == 0 or gen == num_generations - 1:
            best_x = decode(population[max_idx], x_min, x_max, num_bits)
            print(f"Gen {gen+1:>3}: Best Fit={fitnesses[max_idx]:.4f}, "
                  f"Best x={best_x:.4f}, Avg={np.mean(fitnesses):.4f}")

        # ── Build next generation ─────────────────────────────
        new_pop = [population[max_idx].copy()]   # Elitism: keep best

        while len(new_pop) < pop_size:
            p1 = roulette_wheel_selection(population, fitnesses)
            p2 = roulette_wheel_selection(population, fitnesses)
            c1, c2 = single_point_crossover(p1, p2, crossover_rate)
            c1, c2 = mutate(c1, mutation_rate), mutate(c2, mutation_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    # ── Final results ─────────────────────────────────────────
    best_x = decode(best_individual, x_min, x_max, num_bits)
    best_fit = fitness_fn(best_x)
    print("\n" + "="*60)
    print(f"OPTIMISATION COMPLETE")
    print(f"  Best x   : {best_x:.6f}")
    print(f"  Best f(x): {best_fit:.6f}")
    print("="*60)

    # ── Visualisation ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x_range = np.linspace(x_min, x_max, 500)
    y_range = [max(0, fitness_fn(xi)) for xi in x_range]
    ax1.plot(x_range, y_range, 'b-', linewidth=2, label='Fitness f(x)')
    ax1.axvline(best_x, color='red', linestyle='--', linewidth=2,
                label=f'GA Best x={best_x:.3f}')
    ax1.scatter([best_x], [best_fit], color='red', s=150, zorder=5)
    ax1.set_title('Fitness Landscape & GA Solution')
    ax1.set_xlabel('x  (Process Parameter)')
    ax1.set_ylabel('f(x)  (Quality / Fitness)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    gens = range(1, num_generations + 1)
    ax2.plot(gens, best_fitness_history, 'g-', linewidth=2, label='Best Fitness')
    ax2.plot(gens, avg_fitness_history, color='orange', linestyle='--',
             linewidth=2, label='Average Fitness')
    ax2.set_title('GA Convergence Over Generations')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle('Genetic Algorithm Optimisation', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('genetic_algorithm.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'genetic_algorithm.png'")
    return best_x, best_fit


# ================================================================
#   ENTRY POINT
# ================================================================
if __name__ == "__main__":
    # Try to load real dataset first
    data = load_real_dataset()

    if data is not None:
        print("\n[MODE] Using REAL dataset to build empirical fitness landscape.\n")
        fitness_fn = build_fitness_from_data(data, n_bins=100)
        x_min, x_max = 0.0, 10.0
    else:
        # ══════════════════════════════════════════════════════════
        # SYNTHETIC FALLBACK — only used when real data unavailable
        # ══════════════════════════════════════════════════════════
        print("\n[MODE] Using SYNTHETIC fitness function (real dataset unavailable).\n")
        fitness_fn = synthetic_fitness_function
        x_min, x_max = 0.0, 10.0

    genetic_algorithm(
        fitness_fn,
        x_min=x_min, x_max=x_max,
        pop_size=50,
        num_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.01,
        num_bits=16
    )


# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Real dataset (UCI Dry Bean) loaded → first numeric column scaled to [0,10]
# 2. Empirical fitness built from histogram density of real measurements
# 3. GA binary-encodes real values into 16-bit chromosomes
# 4. Initialize random population of 50 chromosomes
# 5. For 100 generations:
#    a. Decode each chromosome → compute fitness from empirical surface
#    b. Roulette-wheel selects high-fitness parents
#    c. Single-point crossover creates two children
#    d. Bit-flip mutation adds diversity
#    e. Elitism keeps the best individual unchanged
# 6. Best chromosome decoded → optimal parameter x found
# 7. Convergence plot shows fitness improving each generation
# Synthetic fallback: f(x)=x*sin(x)+1 used only if URL fails.
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# ──────────────────────────────────────────────────────────────
# GENETIC ALGORITHMS — Complete Guide
# ──────────────────────────────────────────────────────────────
#
# WHAT IS A GENETIC ALGORITHM?
#   GA is a search/optimisation algorithm inspired by Charles Darwin's
#   theory of natural selection ("survival of the fittest").
#   It maintains a population of candidate solutions and evolves them
#   over generations using crossover, mutation, and selection.
#   Key principle: Good solutions survive and reproduce; bad ones die out.
#
# BIOLOGICAL ANALOGY:
#   Chromosome  = Encoded candidate solution (binary string)
#   Gene        = Single bit in the chromosome
#   Population  = Set of all current candidate solutions
#   Fitness     = How good a solution is (quality measure)
#   Selection   = Better solutions reproduce more often
#   Crossover   = Combining genetic material of two parents
#   Mutation    = Random small change to maintain diversity
#   Generation  = One cycle of evaluation + reproduction
#
# WHY USE GA INSTEAD OF GRADIENT-BASED METHODS?
#   - Works with non-differentiable, discontinuous, noisy fitness functions
#   - Handles multimodal landscapes (multiple peaks) — less likely to get stuck
#   - No gradient needed — purely evaluates fitness function values
#   - Naturally parallelisable (evaluate all individuals simultaneously)
#   - Works with discrete, continuous, or mixed-variable problems
#
# APPLICATION — SPRAY DRYING OF COCONUT MILK:
#   Spray drying converts liquid coconut milk into powder by atomising it
#   into a hot air chamber. Key controllable parameters:
#     • Inlet temperature (140–200 °C)
#     • Outlet temperature (70–90 °C)
#     • Feed rate (mL/min)
#     • Atomisation pressure (bar)
#   GA finds the combination of these parameters that maximises:
#     • Moisture content of powder (lower = better shelf life)
#     • Solubility of powder (higher = better quality)
#     • Encapsulation efficiency (higher = better fat protection)
#   This is a multi-parameter, non-linear optimisation problem — perfect for GA.
#
# KEY GA PARAMETERS AND THEIR EFFECT:
#   pop_size:       Larger = more diversity, slower per generation
#   num_generations: More = more evolution time, better convergence
#   crossover_rate: 0.6–0.9 typical; too low = little recombination
#   mutation_rate:  0.001–0.05 typical; too high = random search
#   num_bits:       Higher = more precision in encoding real values
#
# GA VARIANTS:
#   • Binary GA (this code): solutions encoded as binary strings
#   • Real-valued GA: genes are floating-point numbers directly
#   • Permutation GA: for ordering problems (TSP)
#   • Genetic Programming: evolves programs/trees
#   • Hybrid GA-NN: GA optimises neural network hyperparameters (this practical)
#
# SELECTION METHODS:
#   • Roulette Wheel (used here): probability ∝ fitness
#   • Tournament: pick k individuals, best wins
#   • Rank: based on rank not raw fitness (handles scale issues better)
#   • Elitism: always carry forward best individual unchanged
#
# CROSSOVER TYPES:
#   • Single-point (used here): one cut, swap tails
#   • Two-point: two cuts, swap middle segment
#   • Uniform: each bit from parent1 or parent2 with 50% probability
#
# SCHEMA THEOREM (Holland):
#   Short, above-average fitness patterns (schemas) grow exponentially.
#   Building blocks combine via crossover to form globally optimal solutions.
#   This explains mathematically why GA tends to find good solutions.
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. Define Phenotype and Genotype with an example.
# A1. Genotype: The encoded form of a solution — the binary chromosome the GA works with.
#     Phenotype: The actual problem-domain value — the decoded real number.
#     Example: Genotype = "0100110010001010" (16 bits)
#              Phenotype = x = 3.05 (inlet temperature in normalised scale)
#     The GA applies crossover and mutation to the genotype.
#     The fitness function is evaluated in phenotype space.
#
# Q2. Define Encoding and Decoding. Explain different encoding techniques.
# A2. Encoding: Converting a problem solution (phenotype) into a chromosome (genotype).
#     Decoding: Reverse — converting chromosome back to the actual solution.
#     Techniques:
#       Binary Encoding:   Each solution is a string of 0s and 1s.
#                          Most common. Allows bit-flip mutation naturally.
#       Gray Code:         Adjacent integers differ by exactly 1 bit.
#                          Reduces Hamming cliff problem in binary encoding.
#       Real-value:        Genes are floating-point numbers directly.
#                          Better for continuous optimisation.
#       Permutation:       Sequence of numbers. Used for TSP, scheduling.
#       Tree:              Used in genetic programming (encode programs).
#
# Q3. Define Population, Genes, and Fitness Function in GA.
# A3. Population: The full set of candidate solutions (chromosomes) at any generation.
#       Size matters: too small → low diversity, premature convergence;
#       too large → computational cost per generation.
#     Gene: A single element/bit in a chromosome.
#       In binary encoding: one bit. In real-value: one floating-point number.
#     Fitness Function: Maps a candidate solution to a numerical quality score.
#       Higher = better. Drives the evolution — GA maximises this.
#       Must accurately reflect the actual problem objective.
#       For spray drying: could be a neural network that predicts powder quality
#       given the process parameters (hence "hybrid GA-NN").
#
# Q4. Explain the Genetic Algorithm with its full architecture.
# A4. GA Architecture (phases):
#     1. Initialisation:  Create random population of N chromosomes
#     2. Evaluation:      Compute fitness f(x) for each individual
#     3. Stopping Check:  If max generations or threshold reached → stop
#     4. Selection:       Choose parents (roulette/tournament/rank)
#     5. Crossover:       Combine parent chromosomes → offspring
#     6. Mutation:        Randomly alter bits in offspring
#     7. Replacement:     Form new population from offspring (+elitism)
#     8. Go to step 2
#     Output: Best individual found across all generations.
#
# Q5. What is Crossover? Explain Single-Point and Two-Point crossover.
# A5. Crossover (Recombination): Combining genetic material from two parents.
#     Analogy: biological sexual reproduction.
#     Single-Point: Pick random cut point i.
#       Child1 = Parent1[0:i] + Parent2[i:]
#       Child2 = Parent2[0:i] + Parent1[i:]
#     Two-Point: Pick two cut points i and j.
#       Child1 = P1[0:i] + P2[i:j] + P1[j:]
#       Child2 = P2[0:i] + P1[i:j] + P2[j:]
#     Uniform: Each bit inherited from P1 or P2 with 50% probability independently.
#     Crossover rate (0.6-0.9): probability that crossover actually occurs.
#
# Q6. What is Mutation and why is it necessary?
# A6. Mutation randomly flips individual bits with a small probability (mutation rate).
#     Necessity:
#       (i)  Introduces diversity lost during selection
#       (ii) Allows exploration of solution space regions not covered by initial population
#       (iii) Prevents premature convergence to local optima
#       (iv) Recovers genetic material eliminated by selection
#     Bit-flip: 0 becomes 1 or 1 becomes 0 with probability p_m.
#     Rate too high (>0.1): destroys good solutions, approaches random search.
#     Rate too low (<0.001): too slow to diversify, gets stuck.
#     Typical: 1/chromosome_length (e.g., 0.0625 for 16-bit).
#
# Q7. What is Elitism in GA and why is it used?
# A7. Elitism: Directly copy the best individual(s) from current generation
#     to the next without modification.
#     Purpose: Guarantees the best solution found is never lost due to crossover/mutation.
#     Without elitism: GA is not monotonically improving — best solution can disappear.
#     With elitism: GA always remembers the best solution found so far.
#     Common in practice; typically 1-5% of population is preserved as elites.
#
# Q8. What are the advantages of GA over conventional optimisation methods?
# A8. (i)  No gradient needed — black-box optimisation
#     (ii) Handles multimodal, discontinuous, noisy objective functions
#     (iii) Naturally explores multiple regions simultaneously (population-based)
#     (iv) Less likely to get trapped in local optima than gradient descent
#     (v)  Works with discrete, continuous, or mixed decision variables
#     (vi) Easily parallelisable — evaluate population members concurrently
#     (vii) Robust to noise in fitness evaluations
#
# Q9. What is the Schema Theorem?
# A9. A schema is a chromosome template with fixed and wildcard positions (e.g., "1**0*1").
#     Holland's Schema Theorem: Short, high-fitness, low-order schemas
#     (building blocks) grow exponentially in the population over generations.
#     Short = few fixed positions. Low-order = few fixed bits. High-fitness = above average.
#     Building Block Hypothesis: GA works by discovering, combining, and propagating
#     high-quality building blocks via crossover to construct globally optimal solutions.
#
# Q10. How does GA differ from Simulated Annealing (SA)?
# A10. GA: Population-based, explores multiple solutions simultaneously, uses
#      crossover and mutation, inspired by biological evolution.
#      SA: Single solution, probabilistically accepts worse solutions to escape
#      local optima (like annealing metal), uses temperature schedule.
#      GA is better for large, complex, multimodal problems.
#      SA is simpler, uses less memory, good for single-solution problems.
#      Both are metaheuristics — do not guarantee globally optimal solution.
# ============================================================
