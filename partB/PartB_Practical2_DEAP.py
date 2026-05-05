# ============================================================
# FILE: PartB_Practical2_DEAP.py
# STANDALONE FILE — No other files needed.
# SUBJECT: Computational Intelligence (CI)
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# STEP 1 — Install required libraries (only once):
#   pip install deap numpy matplotlib
#
# OPTION A — Jupyter Notebook / Google Colab (RECOMMENDED):
#   1. Open colab.research.google.com → New Notebook
#   2. First cell: !pip install deap numpy matplotlib
#   3. Paste entire code into next cell → Shift+Enter
#   4. Output: generation-by-generation logs + convergence plot
#   5. 'deap_results.png' saved in session
#
# OPTION B — PyCharm:
#   1. Open Terminal inside PyCharm
#   2. Run: pip install deap numpy matplotlib
#   3. Open this file → Click ▶ Run
#   4. Logs in console, plot opens as popup
#
# OPTION C — Terminal:
#   pip install deap numpy matplotlib
#   python PartB_Practical2_DEAP.py
#
# EXPECTED OUTPUT:
#   - gen 0: min=XX.XX, avg=XX.XX (OneMax problem: count of 1s in chromosome)
#   - Fitness improving each generation toward maximum (=100 for 100-bit chromosome)
#   - Final best individual printed with its fitness
#   - Convergence plot saved as deap_results.png
# ============================================================

# Import DEAP modules for evolutionary algorithm framework
from deap import base       # base.Toolbox — container for all EA operators
from deap import creator    # creator — dynamically creates Fitness and Individual classes
from deap import tools      # tools — standard EA operators (crossover, mutation, selection)
from deap import algorithms # algorithms — ready-made EA algorithms (eaSimple, eaMuPlusLambda)

# Import random for stochastic operations
import random
# Import numpy for numerical operations and statistics
import numpy as np
# Import matplotlib for plotting convergence graphs
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: DEFINE FITNESS AND INDIVIDUAL CLASSES USING CREATOR
# ============================================================

# Create a FitnessMax class — we want to MAXIMIZE fitness
# weights=(1.0,) means single-objective maximization
# weights=(-1.0,) would be minimization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Create Individual class — a list with a fitness attribute of type FitnessMax
# Each individual is a list of bits (0s and 1s) — a binary chromosome
creator.create("Individual", list, fitness=creator.FitnessMax)

# ============================================================
# STEP 2: DEFINE THE PROBLEM — OneMax
# ============================================================
# OneMax: simple benchmark — maximize the number of 1s in a binary string
# Fitness = count of 1s in the chromosome
# Maximum fitness = IND_SIZE (all bits are 1)
# Good for demonstrating GA mechanics clearly

# Length of each individual chromosome (number of bits)
IND_SIZE = 100  # Each individual has 100 bits; perfect fitness = 100

def evaluate_onemax(individual):
    """
    Fitness/evaluation function for the OneMax problem.
    Counts the number of 1s in the binary chromosome.
    Returns a TUPLE — DEAP always requires fitness as a tuple.
    Perfect individual: all 100 bits = 1 → fitness = (100,)
    """
    # sum(individual) counts how many bits are 1
    return (sum(individual),)  # Must return tuple even for single-objective

# ============================================================
# STEP 3: INITIALIZE THE TOOLBOX WITH OPERATORS
# ============================================================

# Create a Toolbox — this is DEAP's central registry for all operators
toolbox = base.Toolbox()

# Register a function to generate a single random bit (0 or 1)
# random.randint(0, 1) returns 0 or 1 with equal probability
toolbox.register("attr_bool", random.randint, 0, 1)

# Register Individual initializer: repeat attr_bool IND_SIZE times
# tools.initRepeat fills a creator.Individual with IND_SIZE random bits
toolbox.register("individual",
                 tools.initRepeat,
                 creator.Individual,   # Container type
                 toolbox.attr_bool,    # Function to call for each element
                 n=IND_SIZE)           # Number of elements

# Register Population initializer: a list of individuals
toolbox.register("population",
                 tools.initRepeat,
                 list,                  # Population is a plain list
                 toolbox.individual)    # Each element is an Individual

# Register the evaluation (fitness) function
toolbox.register("evaluate", evaluate_onemax)

# Register crossover operator: two-point crossover
# tools.cxTwoPoint swaps the section between two random points in two parents
toolbox.register("mate", tools.cxTwoPoint)

# Register mutation operator: flip each bit with probability indpb
# tools.mutFlipBit flips 0→1 or 1→0 at each position with probability indpb
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # 5% per bit

# Register selection operator: tournament selection with tournament size 3
# tools.selTournament picks k=3 random individuals, best one survives
toolbox.register("select", tools.selTournament, tournsize=3)

# ============================================================
# STEP 4: STATISTICS TRACKING SETUP
# ============================================================

# Create a Statistics object to track fitness values across generations
stats = tools.Statistics(lambda ind: ind.fitness.values)

# Register statistics functions to compute each generation
stats.register("avg", np.mean)   # Average fitness of population
stats.register("min", np.min)    # Minimum fitness in population
stats.register("max", np.max)    # Maximum fitness in population
stats.register("std", np.std)    # Standard deviation of fitness

# Hall of Fame: stores the best individual ever seen across all generations
hof = tools.HallOfFame(1)  # Keep only the single best individual

# ============================================================
# STEP 5: MAIN GENETIC ALGORITHM — Using DEAP's eaSimple
# ============================================================
def run_deap_ga():
    """
    Runs the DEAP Genetic Algorithm using the built-in eaSimple algorithm.
    eaSimple implements standard generational GA:
      1. Evaluate initial population
      2. For each generation:
         a. Select offspring (tournament selection)
         b. Clone selected individuals
         c. Apply crossover with probability cxpb
         d. Apply mutation with probability mutpb
         e. Evaluate offspring with invalid fitness
         f. Replace population with offspring
      3. Update statistics and Hall of Fame
    """
    print("="*60)
    print("DEAP — Distributed Evolutionary Algorithms in Python")
    print(f"Problem: OneMax (maximize 1s in {IND_SIZE}-bit chromosome)")
    print("="*60)

    # Create initial population of 300 individuals
    pop_size = 300  # Number of individuals in each generation
    population = toolbox.population(n=pop_size)
    print(f"\nInitial population: {pop_size} individuals, each {IND_SIZE} bits")

    # GA parameters
    CXPB = 0.7    # Crossover probability: 70% chance two parents crossover
    MUTPB = 0.2   # Mutation probability: 20% chance an individual is mutated
    NGEN = 50     # Number of generations to evolve

    print(f"Crossover probability: {CXPB}")
    print(f"Mutation probability:  {MUTPB}")
    print(f"Number of generations: {NGEN}")
    print("\nStarting evolution...\n")

    # Run eaSimple — complete generational GA with statistics and HOF tracking
    # Returns: final population, logbook (statistics per generation)
    population, logbook = algorithms.eaSimple(
        population,        # Initial population
        toolbox,           # Toolbox with all registered operators
        cxpb=CXPB,        # Crossover probability
        mutpb=MUTPB,       # Mutation probability
        ngen=NGEN,         # Number of generations
        stats=stats,       # Statistics object for tracking
        halloffame=hof,    # Hall of Fame for best individual
        verbose=True       # Print stats each generation
    )

    return population, logbook, hof

# ============================================================
# STEP 6: CUSTOM MANUAL GA (to show explicit step-by-step)
# ============================================================
def run_manual_ga():
    """
    Manual implementation of the GA loop (same logic as eaSimple but explicit).
    Shows each step clearly: evaluate → select → clone → crossover → mutate → replace.
    This matches the algorithm structure shown in the DEAP lab manual theory.
    """
    print("\n" + "="*60)
    print("MANUAL GA LOOP — Step-by-step DEAP implementation")
    print("="*60)

    # Initialize a smaller population for manual demo
    pop = toolbox.population(n=50)  # 50 individuals

    # GA parameters for manual run
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate fitness of the entire initial population
    fitnesses = list(map(toolbox.evaluate, pop))  # Apply evaluate to each individual
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit  # Assign fitness tuple to each individual

    print(f"Initial population evaluated. Size: {len(pop)}")

    manual_best_history = []  # Track best fitness per generation

    # EVOLUTION LOOP — runs for NGEN generations
    for g in range(NGEN):
        # SELECTION: Select next generation individuals using tournament selection
        offspring = toolbox.select(pop, len(pop))  # Select len(pop) individuals
        # CLONING: Clone selected individuals (to avoid modifying original pop)
        offspring = list(map(toolbox.clone, offspring))

        # CROSSOVER: Apply crossover to pairs of offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:          # Apply crossover with probability CXPB
                toolbox.mate(child1, child2)     # In-place two-point crossover
                del child1.fitness.values        # Invalidate fitness (needs re-evaluation)
                del child2.fitness.values        # Invalidate fitness of child2 too

        # MUTATION: Apply mutation to each offspring individually
        for mutant in offspring:
            if random.random() < MUTPB:          # Apply mutation with probability MUTPB
                toolbox.mutate(mutant)           # In-place bit-flip mutation
                del mutant.fitness.values        # Invalidate fitness (needs re-evaluation)

        # RE-EVALUATE: Only re-evaluate individuals whose fitness was invalidated
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit  # Assign new fitness values

        # REPLACEMENT: New generation completely replaces the old population
        pop[:] = offspring  # In-place replacement

        # Track best fitness in this generation
        best_fit = max(ind.fitness.values[0] for ind in pop)
        manual_best_history.append(best_fit)

        # Print every 10 generations
        if g % 10 == 0 or g == NGEN - 1:
            avg_fit = np.mean([ind.fitness.values[0] for ind in pop])
            print(f"Gen {g+1:>3}: Best={best_fit}, Avg={avg_fit:.1f}")

    # Find and print best individual from manual run
    best_individual = max(pop, key=lambda ind: ind.fitness.values[0])
    print(f"\nManual GA Best Individual Fitness: {best_individual.fitness.values[0]}/{IND_SIZE}")
    print(f"Best Individual (first 20 bits): {best_individual[:20]}...")

    return manual_best_history

# ============================================================
# MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    # Run eaSimple version (official DEAP algorithm)
    final_pop, logbook, hof = run_deap_ga()

    # Print Hall of Fame (best individual ever)
    print(f"\n{'='*60}")
    print(f"HALL OF FAME — Best Individual Ever Found:")
    print(f"Fitness: {hof[0].fitness.values[0]}/{IND_SIZE}")
    print(f"Individual (first 30 bits): {hof[0][:30]}...")
    print(f"{'='*60}")

    # Run manual step-by-step version
    manual_history = run_manual_ga()

    # ============================================================
    # VISUALIZATION
    # ============================================================
    # Extract statistics from logbook for plotting
    gen = logbook.select("gen")          # Generation numbers
    avg_fits = logbook.select("avg")     # Average fitness per generation
    max_fits = logbook.select("max")     # Best fitness per generation
    min_fits = logbook.select("min")     # Worst fitness per generation

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: eaSimple convergence (avg, max, min)
    axes[0].plot(gen, max_fits, 'g-', linewidth=2, label='Best Fitness')
    axes[0].plot(gen, avg_fits, 'b--', linewidth=2, label='Average Fitness')
    axes[0].plot(gen, min_fits, 'r:', linewidth=2, label='Worst Fitness')
    axes[0].axhline(y=IND_SIZE, color='black', linestyle='--',
                    alpha=0.5, label=f'Perfect Fitness ({IND_SIZE})')
    axes[0].set_title('DEAP eaSimple — OneMax Convergence', fontsize=13)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Fitness (Number of 1s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, IND_SIZE + 5])

    # Plot 2: Manual GA convergence
    axes[1].plot(range(1, len(manual_history)+1), manual_history,
                 'purple', linewidth=2, label='Manual GA Best')
    axes[1].axhline(y=IND_SIZE, color='black', linestyle='--',
                    alpha=0.5, label=f'Perfect ({IND_SIZE})')
    axes[1].set_title('Manual DEAP GA — OneMax Convergence', fontsize=13)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Best Fitness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, IND_SIZE + 5])

    plt.suptitle('DEAP — Distributed Evolutionary Algorithms (OneMax Problem)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('deap_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved as 'deap_results.png'")


# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. creator.create() dynamically builds FitnessMax and Individual classes
# 2. Toolbox registers all operators: attr_bool, individual, population,
#    evaluate, mate (cxTwoPoint), mutate (mutFlipBit), select (tournament)
# 3. eaSimple runs the standard generational GA:
#    - Initial population of 300 individuals, each 100 random bits
#    - Each generation: tournament select → clone → crossover (70%) → mutate (20%)
#    - Re-evaluate only individuals whose fitness was invalidated
#    - Entire population replaced by offspring
# 4. Statistics (avg/min/max/std) tracked every generation via logbook
# 5. Hall of Fame stores the single best individual across all generations
# 6. Manual GA shows the explicit step-by-step loop for clarity
# 7. Convergence plots show fitness improving toward 100 (all bits = 1)
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: DEAP — Distributed Evolutionary Algorithms in Python
# DEAP is an open-source Python framework for evolutionary computation.
# Key philosophy: explicit algorithms, transparent data structures.
# Works with multiprocessing and SCOOP for true parallel/distributed execution.
# Supports: Genetic Algorithms, Genetic Programming, Evolution Strategies,
#           Multi-objective Optimization (NSGA-II, SPEA2), PSO, DE.
# The Toolbox pattern allows complete customization of every EA component.
# OneMax problem: simplest GA benchmark — maximize count of 1s in binary string.
# In real applications: replace evaluate_onemax with your actual objective function.
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. What is DEAP and what are its key features?
# A1. DEAP (Distributed Evolutionary Algorithms in Python) is an open-source
#     framework for rapid prototyping of evolutionary algorithms. Key features:
#     - Genetic Algorithms with any representation (list, array, tree, etc.)
#     - Genetic Programming with prefix trees (loosely/strongly typed)
#     - Evolution Strategies including CMA-ES
#     - Multi-objective optimization: NSGA-II, NSGA-III, SPEA2
#     - Parallelization with multiprocessing and SCOOP
#     - Hall of Fame, checkpoints, benchmarks, genealogy tracking
#
# Q2. Explain the role of the creator module in DEAP.
# A2. creator dynamically creates new Python classes at runtime.
#     creator.create("FitnessMax", base.Fitness, weights=(1.0,)):
#       Creates a class FitnessMax inheriting from base.Fitness.
#       weights=(1.0,) → maximize; (-1.0,) → minimize; (1.0,-1.0) → multi-objective.
#     creator.create("Individual", list, fitness=creator.FitnessMax):
#       Creates Individual class inheriting from list with fitness attribute.
#     This approach avoids predefined rigid types — users define exactly what they need.
#
# Q3. What is a Toolbox in DEAP?
# A3. Toolbox is DEAP's central registry (container) for all EA operators and functions.
#     toolbox.register("name", function, *fixed_args): registers a function with name.
#     Registered functions can be called as toolbox.name(*remaining_args).
#     Standard registrations:
#     - toolbox.register("individual", tools.initRepeat, Individual, attr, n=N)
#     - toolbox.register("evaluate", my_fitness_function)
#     - toolbox.register("mate", tools.cxTwoPoint)
#     - toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
#     - toolbox.register("select", tools.selTournament, tournsize=3)
#
# Q4. Describe the working of a genetic algorithm in DEAP step by step.
# A4. 1. Initialization: pop = toolbox.population(n=300) — creates 300 random individuals
#     2. Evaluation: assign fitness to each individual using toolbox.evaluate
#     3. Selection: offspring = toolbox.select(pop, len(pop)) — tournament selection
#     4. Cloning: offspring = map(toolbox.clone, offspring) — avoid reference sharing
#     5. Crossover: for pairs: if random() < CXPB: toolbox.mate(c1, c2); del fitness
#     6. Mutation: for each: if random() < MUTPB: toolbox.mutate(m); del fitness
#     7. Re-evaluation: evaluate only individuals with invalid (deleted) fitness
#     8. Replacement: pop[:] = offspring — replace entire old population
#     9. Repeat from step 3 for NGEN generations
#
# Q5. What are the crossover operators available in DEAP tools?
# A5. tools.cxOnePoint: single-point crossover — one cut point, swap tails
#     tools.cxTwoPoint: two-point crossover — two cut points, swap middle section
#     tools.cxUniform: uniform crossover — each gene from parent1 or parent2 (50%)
#     tools.cxBlend: blend crossover for real-valued (α parameter)
#     tools.cxSimulatedBinary: SBX crossover for real-valued problems
#     tools.cxOrdered: OX crossover for permutation problems (TSP)
#
# Q6. What are the mutation operators in DEAP?
# A6. tools.mutFlipBit(indpb): flip each bit with probability indpb (binary)
#     tools.mutGaussian(mu, sigma, indpb): Gaussian perturbation (real-valued)
#     tools.mutShuffleIndexes(indpb): shuffle genes randomly (permutation)
#     tools.mutUniformInt(low, up, indpb): random integer in [low, up] (integer)
#     tools.mutPolynomialBounded: polynomial mutation (bounded real-valued)
#
# Q7. What is Hall of Fame in DEAP?
# A7. Hall of Fame (HoF) stores the best individuals that ever lived in the population.
#     tools.HallOfFame(n) keeps the top-n individuals across all generations.
#     It automatically updates when better individuals are found.
#     Important: HoF individuals are NOT part of the active population —
#     they're kept separately so the best solution is never lost (even if the
#     population converges away from it).
#
# Q8. How does DEAP support parallelization?
# A8. DEAP supports two parallelization mechanisms:
#     multiprocessing: toolbox.register("map", multiprocessing.Pool().map)
#     replaces Python's map with Pool.map, evaluating population in parallel.
#     SCOOP (Scalable COncurrent Operations in Python): distributed evaluation
#     across multiple computers in a cluster.
#     Usage: python -m scoop script.py  (distributes across available workers)
#     The key advantage: evaluation function (usually the bottleneck) is parallelized.
#
# Q9. What is the difference between eaSimple and eaMuPlusLambda in DEAP?
# A9. eaSimple: Standard generational GA. μ parents → μ offspring → replace all.
#     eaMuPlusLambda: (μ+λ) strategy. μ parents produce λ offspring.
#     Best μ individuals from (μ+λ) combined survive to next generation.
#     eaMuCommaLambda: (μ,λ) strategy. μ parents produce λ offspring (λ>μ).
#     Only offspring survive (parents are discarded). Prevents stagnation.
#     eaSimple is most common; (μ+λ) is best when you want elitism built-in.
#
# Q10. Why must fitness values be tuples in DEAP?
# A10. DEAP uses tuples for fitness to uniformly support both single and
#      multi-objective optimization. A tuple with one element (e.g., (100,))
#      works for single-objective. Multiple elements (e.g., (0.8, 120.0))
#      support multi-objective — each element corresponds to one objective,
#      and the weights tuple defines whether each is maximized or minimized.
#      This design allows the same code structure for all problem types.
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# DEAP is a flexible evolutionary computation framework, not just a single algorithm.
# It provides reusable building blocks: representation, operators, algorithms, statistics.
#
# Key design idea:
#   creator defines individual/fitness types dynamically,
#   toolbox registers "how evolution happens."
#
# Why DEAP is exam-relevant:
#   - Clear mapping from theory to code.
#   - Easy swap of operators (crossover/mutation/selection) for experiments.
#   - Supports single and multi-objective optimization.
#
# In practice:
#   performance bottleneck is usually fitness evaluation, so parallel map is critical.
#
# Typical workflow:
#   define individual -> register operators -> initialize population ->
#   run algorithm (eaSimple/eaMuPlusLambda) -> inspect logbook + hall of fame.
#
# One-line viva line:
# "DEAP separates algorithm skeleton from problem-specific evaluation for modular EA design."
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. What is the difference between `tools` and `algorithms` in DEAP?
# A11. `tools` provides low-level operators; `algorithms` provides ready-made evolutionary loops.
#
# Q12. Why invalidate fitness after mutation/crossover?
# A12. Because genotype changed; cached fitness is stale and must be recomputed.
#
# Q13. What does HallOfFame protect against?
# A13. Losing historically best solutions due to stochastic generational replacement.
#
# Q14. Why is OneMax often used for teaching GA?
# A14. It has simple fitness interpretation and clearly demonstrates convergence behavior.
#
# Q15. How do you adapt this code to real optimization?
# A15. Replace evaluation function with domain objective and adjust representation/operators
#      to match variable types and constraints.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) WHY DEAP IS POPULAR
#    - Clean separation between problem definition and algorithm mechanics.
#    - Easy experimentation by swapping operators.
#
# 2) IMPORTANT DEAP OBJECTS
#    - creator: builds Fitness/Individual classes.
#    - toolbox: operator registry (evaluate, mate, mutate, select).
#    - algorithms: pre-built loops (eaSimple, eaMuPlusLambda).
#    - logbook/stats: convergence monitoring.
#
# 3) FITNESS TUPLE IDEA
#    - Single objective -> one value tuple.
#    - Multi-objective -> multi-value tuple with weights.
#
# 4) INVALID FITNESS RE-EVALUATION
#    - After crossover/mutation, cached fitness is deleted.
#    - Only invalid individuals are re-evaluated for efficiency.
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "DEAP toolbox registers all GA operators; eaSimple repeatedly selects, varies,
#     evaluates, and replaces population while statistics track convergence."
# ============================================================
