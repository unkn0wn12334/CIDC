# ============================================================
# FILE: PartA_Practical6_ClonalSelection.py
# STANDALONE FILE — No other files needed.
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# STEP 1 — Install required libraries (only once):
#   pip install numpy matplotlib
#   (random and copy are Python built-ins — no install needed)
#
# OPTION A — Jupyter Notebook / Google Colab (RECOMMENDED):
#   1. Open Jupyter or colab.research.google.com, new notebook
#   2. Paste entire code into a cell
#   3. Press Shift+Enter
#   4. Output appears below:
#      - Initialization message with population size
#      - Gen 1, 10, 20... logs showing best affinity and best x improving
#      - Final best antibody (x) and affinity value
#   5. Two-panel plot appears inline:
#      Left:  Affinity function with the best solution marked in red
#      Right: Convergence graph (best affinity vs generations)
#   6. 'clonalg.png' saved in current directory
#
# OPTION B — PyCharm:
#   1. Open this file → install numpy/matplotlib if prompted
#   2. Click ▶ Run
#   3. Logs appear in console, plot opens in popup window
#
# OPTION C — Terminal:
#   python PartA_Practical6_ClonalSelection.py
#
# EXPECTED OUTPUT:
#   - Best x found: somewhere in [0, 20] where sin(x)*cos(x/2)+0.5 is maximum
#   - Affinity improving each generation, converging near 1.5
#   - clonalg.png saved showing the multimodal function and solution
#
# TO CHANGE PARAMETERS: Edit the clonalg() call at the bottom:
#   pop_size=30      → number of antibodies
#   n_selected=10    → how many top antibodies are cloned each generation
#   clone_factor=0.5 → controls how many clones are created
#   n_replace=5      → how many worst antibodies are replaced with random ones
#   n_generations=50 → total evolution cycles
# ============================================================

# Import numpy for mathematical operations and array handling
import numpy as np
# Import random for stochastic operations (selection, mutation)
import random
# Import matplotlib for visualization of convergence
import matplotlib.pyplot as plt
# Import copy for creating independent copies of antibodies
import copy

# ============================================================
# STEP 1: DEFINE THE PROBLEM — AFFINITY (FITNESS) FUNCTION
# ============================================================
def affinity_function(x):
    """
    Affinity function measures the quality of an antibody (candidate solution).
    High affinity = better solution = more likely to be cloned.
    In biological terms: how well an antibody recognizes an antigen.
    Here: maximize f(x) = sin(x) * cos(x/2) + 0.5 over x in [0, 20]
    This is a multimodal function — multiple peaks to test the algorithm.
    """
    # Compute multimodal fitness function with several local maxima
    return np.sin(x) * np.cos(x / 2) + 0.5  # Returns affinity (fitness) value


# ============================================================
# STEP 2: ANTIBODY REPRESENTATION AND INITIALIZATION
# ============================================================
def initialize_population(pop_size, x_min=0, x_max=20):
    """
    Initializes a random population of antibodies.
    Each antibody is a real-valued candidate solution.
    In biological terms: diverse initial immune repertoire.
    """
    # Generate pop_size random values uniformly in [x_min, x_max]
    antibodies = [random.uniform(x_min, x_max) for _ in range(pop_size)]
    return antibodies  # Return list of real-valued antibodies

def compute_affinities(antibodies):
    """
    Computes affinity (fitness) for each antibody in the population.
    Returns list of affinity values corresponding to each antibody.
    """
    # Apply affinity_function to each antibody
    return [affinity_function(ab) for ab in antibodies]


# ============================================================
# STEP 3: SELECTION — Select High-Affinity Antibodies for Cloning
# ============================================================
def select_top_antibodies(antibodies, affinities, n_selected):
    """
    Selects the top n_selected antibodies with highest affinity.
    High-affinity antibodies are more likely to recognize the antigen (solution).
    In biology: only B-cells with sufficient antigen affinity are selected to clone.
    """
    # Pair each antibody with its affinity value
    paired = list(zip(antibodies, affinities))
    # Sort pairs by affinity in descending order (highest first)
    paired_sorted = sorted(paired, key=lambda pair: pair[1], reverse=True)
    # Select top n_selected antibodies
    selected = [ab for ab, _ in paired_sorted[:n_selected]]
    return selected  # Return only the antibody values (not affinities)


# ============================================================
# STEP 4: CLONING — Clone Selected Antibodies
# ============================================================
def clone_antibodies(selected_antibodies, clone_factor, total_antibodies):
    """
    Creates clones of selected antibodies.
    In biology: selected B-cells undergo rapid clonal expansion.
    Number of clones per antibody is proportional to its affinity (rank-based here).
    clone_factor * total_antibodies / rank = number of clones for rank-i antibody.
    """
    clones = []  # List to hold all cloned antibodies
    n_selected = len(selected_antibodies)  # Number of selected antibodies

    for i, antibody in enumerate(selected_antibodies):
        # Compute number of clones for this antibody
        # Higher ranked (better) antibodies get more clones
        # Rank i+1 (1-indexed): clone_count = ceil(clone_factor * total / (i+1))
        num_clones = int(np.ceil(clone_factor * total_antibodies / (i + 1)))
        # Create num_clones copies of this antibody
        for _ in range(num_clones):
            clones.append(copy.copy(antibody))  # Independent copy of antibody

    return clones  # Return all cloned antibodies


# ============================================================
# STEP 5: HYPERMUTATION — Mutate Clones with Inverse-Proportional Rate
# ============================================================
def hypermutate(clones, affinities_clones, mutation_strength=2.0, x_min=0, x_max=20):
    """
    Applies hypermutation to clones with rate INVERSELY proportional to affinity.
    Low-affinity clones: high mutation rate (explore widely)
    High-affinity clones: low mutation rate (exploit neighborhood of good solution)
    In biology: somatic hypermutation generates diversity in B-cell receptors.
    """
    mutated_clones = []  # List to hold mutated versions

    # Find max affinity to normalize mutation rate calculation
    max_affinity = max(affinities_clones) if max(affinities_clones) > 0 else 1

    for clone, affinity in zip(clones, affinities_clones):
        # Mutation rate inversely proportional to normalized affinity
        # High affinity → low mutation rate (small perturbation around good solution)
        # Low affinity → high mutation rate (explore more widely)
        normalized_affinity = affinity / max_affinity  # Normalize to [0, 1]
        mutation_rate = mutation_strength * np.exp(-normalized_affinity)  # Exponential inverse

        # Apply Gaussian perturbation to the clone
        # np.random.normal(0, mutation_rate) generates a random perturbation
        perturbation = np.random.normal(0, mutation_rate)
        mutated_value = clone + perturbation  # Add perturbation to antibody value

        # Ensure mutated value stays within valid bounds [x_min, x_max]
        mutated_value = np.clip(mutated_value, x_min, x_max)

        # Add mutated clone to result
        mutated_clones.append(mutated_value)

    return mutated_clones  # Return list of mutated clones


# ============================================================
# STEP 6: REPLACEMENT — Replace Low-Affinity Antibodies
# ============================================================
def replace_low_affinity(population, affinities, new_antibodies, n_replace, x_min, x_max):
    """
    Replaces the n_replace lowest-affinity antibodies with random new ones.
    Maintains diversity and prevents premature convergence.
    In biology: new naive B-cells are constantly generated in bone marrow.
    """
    # Pair antibodies with their affinities
    paired = list(zip(population, affinities))
    # Sort by affinity ascending (worst first)
    paired_sorted = sorted(paired, key=lambda pair: pair[1])
    # Get indices of n_replace worst antibodies
    worst_antibodies = [ab for ab, _ in paired_sorted[:n_replace]]

    # Replace worst antibodies with new random antibodies
    new_random = [random.uniform(x_min, x_max) for _ in range(n_replace)]

    # Rebuild population: remove worst, add new random
    updated_population = [ab for ab in population if ab not in worst_antibodies]
    updated_population.extend(new_random)  # Add new random antibodies

    return updated_population  # Return updated population


# ============================================================
# STEP 7: MAIN CLONALG LOOP
# ============================================================
def clonalg(pop_size=30, n_selected=10, clone_factor=0.5,
            n_replace=5, n_generations=50, x_min=0, x_max=20):
    """
    Main CLONALG loop implementing the complete clonal selection algorithm.
    Phases: Initialize → Select → Clone → Hypermutate → Evaluate → Replace → Repeat
    """
    print("="*60)
    print("CLONAL SELECTION ALGORITHM (CLONALG)")
    print(f"Population: {pop_size}, Selected: {n_selected}, Generations: {n_generations}")
    print(f"Clone Factor: {clone_factor}, Replacements: {n_replace}")
    print("="*60)

    # PHASE 1: INITIALIZATION — Create random population of antibodies
    antibodies = initialize_population(pop_size, x_min, x_max)
    print(f"\nInitialized {pop_size} antibodies randomly in [{x_min}, {x_max}]")

    # Lists to track best affinity over generations for convergence plot
    best_affinity_history = []
    avg_affinity_history = []

    best_antibody = None       # Track best antibody found overall
    best_affinity_ever = float('-inf')  # Track highest affinity seen

    # MAIN EVOLUTION LOOP
    for generation in range(n_generations):

        # PHASE 2: EVALUATION — Compute affinity for all antibodies
        affinities = compute_affinities(antibodies)

        # Update best solution found so far
        max_idx = np.argmax(affinities)  # Index of best antibody
        if affinities[max_idx] > best_affinity_ever:
            best_affinity_ever = affinities[max_idx]
            best_antibody = antibodies[max_idx]

        # Track history
        best_affinity_history.append(max(affinities))
        avg_affinity_history.append(np.mean(affinities))

        # Print progress every 10 generations
        if generation % 10 == 0 or generation == n_generations - 1:
            print(f"Gen {generation+1:>3}: Best Affinity={max(affinities):.4f}, "
                  f"Best x={antibodies[max_idx]:.4f}, Avg={np.mean(affinities):.4f}")

        # PHASE 3: SELECTION — Select top n_selected high-affinity antibodies
        selected = select_top_antibodies(antibodies, affinities, n_selected)

        # PHASE 4: CLONING — Clone selected antibodies proportional to rank
        clones = clone_antibodies(selected, clone_factor, pop_size)

        # PHASE 5: EVALUATION of clones
        clone_affinities = compute_affinities(clones)

        # PHASE 6: HYPERMUTATION — Mutate clones inversely proportional to affinity
        mutated_clones = hypermutate(clones, clone_affinities,
                                     mutation_strength=2.0, x_min=x_min, x_max=x_max)

        # Re-evaluate mutated clones
        mutated_affinities = compute_affinities(mutated_clones)

        # PHASE 7: SELECT BEST from mutated clones + original population
        # Combine original antibodies and mutated clones
        combined = antibodies + mutated_clones
        combined_affinities = compute_affinities(combined)

        # Keep only the top pop_size - n_replace antibodies
        paired_combined = sorted(zip(combined, combined_affinities),
                                  key=lambda p: p[1], reverse=True)
        antibodies = [ab for ab, _ in paired_combined[:pop_size - n_replace]]

        # PHASE 8: REPLACEMENT — Replace n_replace worst with random new antibodies
        new_random = [random.uniform(x_min, x_max) for _ in range(n_replace)]
        antibodies.extend(new_random)

    # FINAL RESULTS
    print("\n" + "="*60)
    print("CLONALG COMPLETE")
    print("="*60)
    print(f"Best Antibody (x): {best_antibody:.6f}")
    print(f"Best Affinity f(x): {best_affinity_ever:.6f}")

    # VISUALIZATION
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Affinity function with found solution
    x_range = np.linspace(x_min, x_max, 500)
    y_range = [affinity_function(xi) for xi in x_range]
    axes[0].plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = sin(x)cos(x/2)+0.5')
    axes[0].axvline(x=best_antibody, color='red', linestyle='--', linewidth=2,
                    label=f'Best x = {best_antibody:.3f}')
    axes[0].scatter([best_antibody], [best_affinity_ever], color='red', s=150, zorder=5,
                    label=f'Best affinity = {best_affinity_ever:.3f}')
    axes[0].set_title('Affinity Function and CLONALG Solution')
    axes[0].set_xlabel('x (Antibody Value / Parameter)')
    axes[0].set_ylabel('Affinity (Fitness)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Convergence over generations
    axes[1].plot(range(1, n_generations+1), best_affinity_history,
                 'g-', linewidth=2, label='Best Affinity')
    axes[1].plot(range(1, n_generations+1), avg_affinity_history,
                 'orange', linestyle='--', linewidth=2, label='Average Affinity')
    axes[1].set_title('CLONALG Convergence')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Affinity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Clonal Selection Algorithm (CLONALG)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clonalg.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as 'clonalg.png'")

    return best_antibody, best_affinity_ever

# Run CLONALG
if __name__ == "__main__":
    best_x, best_aff = clonalg(
        pop_size=30,       # 30 antibodies in population
        n_selected=10,     # Select top 10 for cloning
        clone_factor=0.5,  # Cloning factor β
        n_replace=5,       # Replace 5 worst with random each generation
        n_generations=50,  # Run 50 generations
        x_min=0,           # Search space minimum
        x_max=20           # Search space maximum
    )

# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Initialize: 30 random antibodies (candidate solutions) in [0, 20]
# 2. Evaluate: Compute affinity (fitness) for each antibody
# 3. Select: Pick top 10 highest-affinity antibodies for cloning
# 4. Clone: Each selected antibody cloned — better ones get more clones
#    (proportional: rank 1 gets most, rank 10 gets fewest)
# 5. Hypermutate: Mutate clones — low affinity = large mutation (exploration),
#    high affinity = small mutation (exploitation near good solution)
# 6. Evaluate mutated clones
# 7. Select best from combined population (originals + mutated clones)
# 8. Replace: 5 worst antibodies replaced with new random ones (diversity)
# 9. Repeat 50 times
# Convergence: Algorithm zooms in on best regions; hypermutation prevents stagnation
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: Clonal Selection Algorithm (CLONALG)
# Inspired by the biological immune system's response to pathogens.
# When an antigen (pathogen) enters the body:
#   1. B-cells that recognize it (high affinity) are selected
#   2. Selected B-cells rapidly multiply (clonal expansion)
#   3. Clones undergo somatic hypermutation (diversification)
#   4. Mutated B-cells with higher affinity are selected (affinity maturation)
#   5. Memory B-cells formed (faster response to future exposure)
# CLONALG maps these principles to optimization:
#   Antigen = Problem to solve
#   Antibody = Candidate solution
#   Affinity = Fitness/quality of solution
#   Clonal expansion = Generating more solutions near good ones
#   Hypermutation = Exploring solution space (more for bad, less for good)
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. What is CLONALG and how is it inspired by the biological immune system?
# A1. CLONALG (Clonal Selection Algorithm) is an optimization algorithm inspired by
#     the clonal selection theory of adaptive immunity. In biology, when an antigen
#     enters, B-cells that recognize it (high affinity) are selected, cloned, and
#     hypermutated to produce higher-affinity antibodies. CLONALG mirrors this:
#     high-affinity solutions (antibodies) are selected, cloned more, and mutated
#     to find even better solutions.
#
# Q2. What is antibody representation and affinity function in CLONALG?
# A2. Antibody Representation: Each antibody encodes a candidate solution.
#     Can be real-valued vector, binary string, or other encoding.
#     Represents a point in the search space (e.g., x = 3.7 in [0, 20]).
#     Affinity Function: Measures quality of a solution (like fitness function in GA).
#     High affinity = good solution = gets more clones and better survival chances.
#     Both directly affect algorithm performance — good encoding + relevant affinity.
#
# Q3. Describe the selection and cloning process in CLONALG.
# A3. Selection: Top n antibodies with highest affinity are chosen for cloning.
#     Only "good enough" solutions participate in reproduction.
#     Cloning: Each selected antibody creates multiple copies.
#     Number of clones is proportional to affinity rank:
#     n_clones_i = ceil(β * N / i) where β=clone factor, N=pop size, i=rank.
#     Rank 1 (best) gets most clones; last rank gets fewest.
#     This focuses computational effort on the most promising regions.
#
# Q4. What is hypermutation in CLONALG? Why is it important?
# A4. Hypermutation applies random perturbations to clones.
#     Rate is INVERSELY proportional to affinity:
#     - Low-affinity clones: HIGH mutation rate (explore new regions)
#     - High-affinity clones: LOW mutation rate (fine-tune near good solution)
#     Importance:
#     - Prevents premature convergence to local optima
#     - Maintains population diversity
#     - Balances exploration (finding new areas) and exploitation (refining good solutions)
#     Biologically: somatic hypermutation in B-cells increases receptor diversity.
#
# Q5. Explain all phases of CLONALG briefly.
# A5. 1. Initialization: Generate random antibody population
#     2. Evaluation: Compute affinity for all antibodies
#     3. Selection: Pick top-n high-affinity antibodies
#     4. Cloning: Create proportional copies of selected antibodies
#     5. Hypermutation: Mutate clones inversely proportional to affinity
#     6. Re-evaluation: Compute affinity of mutated clones
#     7. Replacement: Select best from combined; replace worst with random new ones
#     8. Termination: Stop if max iterations or desired affinity reached
#
# Q6. How does CLONALG differ from Genetic Algorithm?
# A6. CLONALG vs GA:
#     Crossover: GA uses crossover extensively; CLONALG does NOT use crossover
#     Mutation: CLONALG uses inverse-proportional hypermutation; GA uses fixed low rate
#     Cloning: CLONALG clones based on rank; GA copies proportional to fitness
#     Replacement: CLONALG replaces worst with random; GA generational replacement
#     Memory: CLONALG has memory cells for good solutions; GA doesn't
#     CLONALG is generally better for multimodal optimization problems.
#
# Q7. What is affinity maturation in the immune system analogy?
# A7. Affinity maturation is the biological process where B-cell receptor affinity
#     for the antigen increases over time through iterative mutation and selection.
#     In CLONALG: through repeated cycles of cloning + hypermutation + selection,
#     antibodies gradually achieve higher and higher affinity for the "antigen"
#     (i.e., solutions converge toward the optimal value).
#
# Q8. What are the applications of Clonal Selection Algorithm?
# A8. (i) Function optimization (multimodal problems)
#     (ii) Pattern recognition and classification
#     (iii) Computer security: intrusion detection, anomaly detection
#     (iv) Scheduling and combinatorial optimization
#     (v) Machine learning: feature selection, neural network training
#     (vi) Robotics: path planning
#     (vii) Engineering design optimization
#
# Q9. What is the role of the replacement step in CLONALG?
# A9. Replacement removes the d lowest-affinity antibodies from the population
#     and introduces d new randomly generated antibodies.
#     Purpose:
#     - Maintains population diversity (prevents all antibodies converging to one point)
#     - Allows exploration of new, unexplored regions
#     - Mimics bone marrow continuously producing new naive B-cells
#     Without replacement: algorithm may converge prematurely to local optimum.
#
# Q10. What is the difference between clonal selection and affinity maturation?
# A10. Clonal Selection: The process of selecting which B-cells (antibodies) will
#      be cloned based on their affinity for the antigen. Only high-affinity B-cells
#      are selected — this is the "selection pressure" in the algorithm.
#      Affinity Maturation: The process that FOLLOWS clonal selection — clones
#      undergo hypermutation and those with higher affinity are retained.
#      Together they drive the population toward better and better solutions.
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# CLONALG is an Artificial Immune System algorithm for optimization/classification.
# Biological inspiration: immune response learns better antibodies over repeated exposure.
#
# Optimization mapping:
#   Antigen -> objective/problem
#   Antibody -> candidate solution
#   Affinity -> fitness score
#   Clonal expansion -> replicate high-quality candidates
#   Hypermutation -> adaptive neighborhood/global search
#
# Why inverse mutation is important:
#   - Good solutions get small mutations (fine tuning).
#   - Poor solutions get larger mutations (exploration).
#
# Diversity mechanisms:
#   - Random replacement of low-affinity cells
#   - Mutation randomness
#   - Multi-clone sampling
#
# Compared to GA:
#   - No crossover needed.
#   - Strong adaptive mutation behavior.
#   - Often effective in multimodal search spaces.
#
# Exam-ready line:
# "CLONALG balances exploration and exploitation via affinity-dependent hypermutation."
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. Why is CLONALG good for multimodal functions?
# A11. It maintains diversity and performs localized refinement through mutation,
#      helping discover multiple promising regions.
#
# Q12. What is immune memory in AIS optimization?
# A12. Best high-affinity antibodies are retained to preserve high-quality solutions
#      and speed up future convergence.
#
# Q13. What happens if mutation strength is too high?
# A13. Search becomes unstable/random and may destroy useful high-affinity structures.
#
# Q14. What happens if mutation strength is too low?
# A14. Algorithm may stagnate around local optima with poor exploration.
#
# Q15. Which problems suit CLONALG?
# A15. Feature selection, anomaly detection, multimodal numerical optimization,
#      scheduling, and pattern recognition tasks.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) IMMUNOLOGY TO ALGORITHM MAPPING
#    - Antigen = problem.
#    - Antibody = candidate solution.
#    - Affinity = quality score.
#
# 2) CLONAL EXPANSION
#    - Better antibodies are cloned more times.
#    - This allocates search budget to promising regions.
#
# 3) SOMATIC HYPERMUTATION
#    - Low-affinity clones mutate strongly (global exploration).
#    - High-affinity clones mutate lightly (local refinement).
#
# 4) DIVERSITY MAINTENANCE
#    - Replacing worst antibodies with random ones avoids collapse into one region.
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "This code applies CLONALG by repeatedly selecting high-affinity antibodies,
#     cloning and hypermutating them, then retaining best evolved candidates to
#     maximize the objective function."
# ============================================================
