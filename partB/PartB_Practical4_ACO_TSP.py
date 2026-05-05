# ============================================================
# FILE: PartB_Practical4_ACO_TSP.py
# STANDALONE FILE — No other files needed.
# SUBJECT: Computational Intelligence (CI)
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# STEP 1 — Install required libraries (only once):
#   pip install numpy matplotlib
#
# OPTION A — Jupyter Notebook / Google Colab (RECOMMENDED):
#   1. Open colab.research.google.com → New Notebook
#   2. First cell: !pip install numpy matplotlib
#   3. Paste entire code into next cell → Shift+Enter
#   4. Output:
#      - City coordinates and distance matrix printed
#      - Iteration-by-iteration best route length improving
#      - Final best route and its total distance
#      - Two plots: city map with best route + convergence graph
#   5. 'aco_tsp.png' saved in session directory
#
# OPTION B — PyCharm / Terminal:
#   pip install numpy matplotlib
#   python PartB_Practical4_ACO_TSP.py
#
# EXPECTED OUTPUT:
#   Iteration  1: Best Distance = XXX.XX
#   Iteration 10: Best Distance = XXX.XX  (improving)
#   ...
#   Iteration 50: Best Distance = XXX.XX  (converged)
#   Best Route: [0, 3, 7, 2, ...] → Total Distance: XXX.XX
#   Plot saved as aco_tsp.png
#
# TO CHANGE CITY COUNT: Edit NUM_CITIES = 10 near the top
# TO CHANGE ITERATIONS: Edit NUM_ITERATIONS = 50
# ============================================================

# Import numpy for matrix operations (distance matrix, pheromone matrix)
import numpy as np
# Import matplotlib for visualizing the TSP route and convergence
import matplotlib.pyplot as plt
# Import random for probabilistic ant decisions
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================
# STEP 1: PROBLEM SETUP — CITY COORDINATES AND DISTANCES
# ============================================================

# Number of cities the salesman must visit
NUM_CITIES = 10

# ACO Algorithm Parameters
NUM_ANTS = 20          # Number of ants in the colony per iteration
NUM_ITERATIONS = 50    # Number of iterations (pheromone update cycles)
ALPHA = 1.0            # α — pheromone influence (higher = more pheromone-guided)
BETA = 2.0             # β — heuristic influence (higher = more distance-guided)
RHO = 0.5              # ρ — pheromone evaporation rate (0.5 = 50% evaporates)
Q = 100                # Q — pheromone deposit constant
INITIAL_PHEROMONE = 1.0  # Initial pheromone level on all edges

def generate_cities(n_cities, area=100):
    """
    Generates random 2D city coordinates within a square area.
    Returns: (n_cities, 2) array of (x, y) coordinates
    """
    cities = np.random.rand(n_cities, 2) * area  # Random x,y in [0, area]
    return cities

def compute_distance_matrix(cities):
    """
    Computes the Euclidean distance between every pair of cities.
    Returns: (n_cities, n_cities) symmetric distance matrix
    D[i][j] = Euclidean distance from city i to city j
    """
    n = len(cities)
    dist_matrix = np.zeros((n, n))  # Initialize with zeros
    for i in range(n):
        for j in range(n):
            if i != j:
                # Euclidean distance: sqrt((x2-x1)² + (y2-y1)²)
                dist_matrix[i][j] = np.sqrt(
                    (cities[i][0] - cities[j][0])**2 +
                    (cities[i][1] - cities[j][1])**2
                )
    return dist_matrix

# Generate cities and compute distances
cities = generate_cities(NUM_CITIES)
dist_matrix = compute_distance_matrix(cities)

print("="*60)
print("ANT COLONY OPTIMIZATION — Traveling Salesman Problem")
print("="*60)
print(f"\nNumber of Cities: {NUM_CITIES}")
print(f"Number of Ants:   {NUM_ANTS}")
print(f"Iterations:       {NUM_ITERATIONS}")
print(f"Alpha (α):        {ALPHA}  (pheromone weight)")
print(f"Beta  (β):        {BETA}   (heuristic weight)")
print(f"Rho   (ρ):        {RHO}    (evaporation rate)")
print(f"\nCity Coordinates:")
for i, (x, y) in enumerate(cities):
    print(f"  City {i}: ({x:.2f}, {y:.2f})")

# ============================================================
# STEP 2: INITIALIZE PHEROMONE MATRIX
# ============================================================
def initialize_pheromones(n_cities, initial_value=INITIAL_PHEROMONE):
    """
    Initializes the pheromone matrix with equal pheromone on all edges.
    pheromone[i][j] = amount of pheromone on edge from city i to city j.
    In biology: initially, all paths are equally attractive.
    """
    # All edges start with the same initial pheromone level
    pheromone = np.ones((n_cities, n_cities)) * initial_value
    # Diagonal = 0 (no self-loops — a city doesn't connect to itself)
    np.fill_diagonal(pheromone, 0)
    return pheromone

# ============================================================
# STEP 3: ANT TOUR CONSTRUCTION
# ============================================================
def ant_tour(pheromone, dist_matrix, start_city=None):
    """
    Constructs a complete TSP tour for one ant.
    The ant starts at a random city and visits each city exactly once.
    City selection is probabilistic: based on pheromone levels and distances.
    
    Probability formula:
    P(i→j) = (τ_ij^α * η_ij^β) / Σ_k(τ_ik^α * η_ik^β)
    where τ = pheromone, η = 1/distance (heuristic), α and β are weights
    """
    n = len(dist_matrix)

    # Start at random city if not specified
    if start_city is None:
        start_city = random.randint(0, n - 1)

    # Track which cities have been visited
    visited = [False] * n          # All cities unvisited initially
    tour = [start_city]            # Tour starts at the starting city
    visited[start_city] = True     # Mark start city as visited

    # Visit all remaining cities one by one
    for _ in range(n - 1):
        current_city = tour[-1]    # Current position of the ant

        # Compute transition probabilities to all unvisited cities
        probabilities = []
        unvisited_cities = []

        for next_city in range(n):
            if not visited[next_city]:  # Only consider unvisited cities
                # Pheromone factor: τ_ij^α
                pheromone_factor = pheromone[current_city][next_city] ** ALPHA

                # Heuristic factor: η_ij^β = (1/distance)^β
                # Closer cities have higher heuristic value
                if dist_matrix[current_city][next_city] > 0:
                    heuristic_factor = (1.0 / dist_matrix[current_city][next_city]) ** BETA
                else:
                    heuristic_factor = 0

                # Numerator of probability formula: τ^α * η^β
                prob = pheromone_factor * heuristic_factor
                probabilities.append(prob)
                unvisited_cities.append(next_city)

        # Normalize probabilities so they sum to 1
        total_prob = sum(probabilities)
        if total_prob == 0:
            # If all probabilities are 0, choose randomly
            next_city = random.choice(unvisited_cities)
        else:
            # Normalize each probability
            normalized_probs = [p / total_prob for p in probabilities]
            # Probabilistic selection: pick city based on normalized probabilities
            # np.random.choice selects index based on probability weights
            chosen_idx = np.random.choice(len(unvisited_cities), p=normalized_probs)
            next_city = unvisited_cities[chosen_idx]

        # Move ant to chosen city
        tour.append(next_city)          # Add to tour
        visited[next_city] = True       # Mark as visited

    return tour  # Returns the complete tour as list of city indices

# ============================================================
# STEP 4: COMPUTE TOUR LENGTH
# ============================================================
def tour_length(tour, dist_matrix):
    """
    Computes the total distance of a complete TSP tour.
    Includes return trip from last city back to starting city.
    """
    total_distance = 0.0
    n = len(tour)

    for i in range(n):
        # Distance from current city to next city
        current = tour[i]
        next_city = tour[(i + 1) % n]  # Modulo wraps around to start
        total_distance += dist_matrix[current][next_city]

    return total_distance  # Total round-trip distance

# ============================================================
# STEP 5: PHEROMONE UPDATE
# ============================================================
def update_pheromones(pheromone, all_tours, all_distances):
    """
    Updates the pheromone matrix after all ants complete their tours.
    Two steps:
    1. EVAPORATION: Reduce all pheromones by factor (1-ρ)
       Simulates pheromone evaporating over time — prevents stagnation
    2. DEPOSITION: Ants deposit pheromone on edges they used
       Better tours (shorter) deposit more pheromone: Q/distance
    """
    # STEP 1: EVAPORATION — reduce all pheromones
    # (1 - RHO) * old_pheromone — longer paths evaporate, shorter ones reinforced
    pheromone *= (1 - RHO)  # In-place evaporation on entire matrix

    # STEP 2: DEPOSITION — each ant adds pheromone to its path
    for tour, distance in zip(all_tours, all_distances):
        # Pheromone deposit amount: Q / total_distance
        # Shorter tours → larger deposit → stronger reinforcement
        deposit = Q / distance

        n = len(tour)
        for i in range(n):
            city_a = tour[i]
            city_b = tour[(i + 1) % n]   # Next city in tour (wraps around)
            pheromone[city_a][city_b] += deposit  # Add to edge a→b
            pheromone[city_b][city_a] += deposit  # Add to edge b→a (symmetric)

    return pheromone  # Return updated pheromone matrix

# ============================================================
# STEP 6: MAIN ACO LOOP
# ============================================================
def aco_tsp():
    """
    Main Ant Colony Optimization loop for the Traveling Salesman Problem.
    Each iteration: all ants construct tours → update pheromones → track best.
    """
    # Initialize pheromone matrix
    pheromone = initialize_pheromones(NUM_CITIES)

    best_tour = None              # Best tour found across all iterations
    best_distance = float('inf') # Best (shortest) distance found
    best_distance_history = []   # Track best distance per iteration

    print(f"\n{'='*60}")
    print(f"Starting ACO — {NUM_ANTS} ants × {NUM_ITERATIONS} iterations")
    print(f"{'='*60}\n")

    # MAIN ITERATION LOOP
    for iteration in range(NUM_ITERATIONS):
        iteration_tours = []      # Tours of all ants in this iteration
        iteration_distances = []  # Distances of all tours in this iteration

        # Each ant constructs a complete tour
        for ant in range(NUM_ANTS):
            # Ant constructs tour starting from a random city
            tour = ant_tour(pheromone, dist_matrix)
            distance = tour_length(tour, dist_matrix)  # Compute tour length

            iteration_tours.append(tour)         # Store this ant's tour
            iteration_distances.append(distance) # Store this ant's distance

            # Update global best if this ant found a shorter route
            if distance < best_distance:
                best_distance = distance
                best_tour = tour.copy()  # Save copy of best tour

        # Update pheromone matrix based on all ants' tours
        pheromone = update_pheromones(pheromone, iteration_tours, iteration_distances)

        # Track best distance for convergence plot
        best_distance_history.append(best_distance)

        # Print progress every 10 iterations
        if (iteration + 1) % 10 == 0 or iteration == 0:
            iter_best = min(iteration_distances)
            print(f"Iteration {iteration+1:>3}: "
                  f"Iteration Best = {iter_best:.2f}, "
                  f"Global Best = {best_distance:.2f}")

    return best_tour, best_distance, best_distance_history, pheromone

# ============================================================
# MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    # Run ACO
    best_tour, best_distance, history, final_pheromone = aco_tsp()

    # Print final results
    print(f"\n{'='*60}")
    print(f"ACO COMPLETE — OPTIMAL ROUTE FOUND")
    print(f"{'='*60}")
    print(f"Best Tour: {best_tour}")
    print(f"Route: {' → '.join(str(c) for c in best_tour)} → {best_tour[0]}")
    print(f"Total Distance: {best_distance:.4f}")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: City map with best route drawn
    ax = axes[0]

    # Draw the best tour as connected lines
    tour_cities = best_tour + [best_tour[0]]  # Close the tour (return to start)
    tour_x = [cities[c][0] for c in tour_cities]  # X coordinates of route
    tour_y = [cities[c][1] for c in tour_cities]  # Y coordinates of route

    ax.plot(tour_x, tour_y, 'b-o', linewidth=2, markersize=8,
            zorder=2, label='Best Route')

    # Mark each city with its index number
    for i, (x, y) in enumerate(cities):
        ax.annotate(f'  C{i}', (x, y), fontsize=11, fontweight='bold', color='darkblue')

    # Highlight start city with a star marker
    ax.scatter(cities[best_tour[0]][0], cities[best_tour[0]][1],
               color='red', s=200, zorder=5, marker='*', label='Start City')

    ax.set_title(f'Best TSP Route\nTotal Distance: {best_distance:.2f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Convergence — best distance over iterations
    axes[1].plot(range(1, len(history)+1), history,
                 'g-', linewidth=2, label='Best Distance')
    axes[1].fill_between(range(1, len(history)+1), history,
                         alpha=0.2, color='green')
    axes[1].set_title('ACO Convergence\n(Best Distance per Iteration)',
                      fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Best Tour Distance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Ant Colony Optimization — Traveling Salesman Problem',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('aco_tsp.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved as 'aco_tsp.png'")


# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Setup: 10 random cities generated, Euclidean distance matrix computed
# 2. Initialize: All pheromone values set to 1.0 (equal initial attraction)
# 3. For each of 50 iterations:
#    a. 20 ants each construct a complete tour:
#       - Start at random city
#       - Probabilistically choose next city: P ∝ pheromone^α × (1/distance)^β
#       - Continue until all cities visited
#       - Return to start city
#    b. Compute tour length for each ant
#    c. Update pheromones:
#       - Evaporation: multiply all by (1-ρ) = 0.5
#       - Deposition: add Q/distance to each edge used by each ant
#    d. Track global best tour
# 4. Result: Best tour (shortest route visiting all cities exactly once)
# 5. Plots: Route visualization + convergence graph
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: Ant Colony Optimization (ACO) for TSP
# ACO inspired by real ants finding shortest path to food using pheromones.
# TSP: Find shortest route visiting N cities exactly once and returning to start.
# TSP is NP-Hard — no known polynomial-time exact algorithm for large N.
# ACO is a metaheuristic that finds near-optimal solutions efficiently.
# Key parameters: α controls pheromone importance, β controls distance importance.
# Higher α: ants follow existing trails more (exploitation)
# Higher β: ants prefer shorter edges more (greedy heuristic)
# Evaporation (ρ): prevents premature convergence, allows exploration.
# Applications: Vehicle routing, network routing, job scheduling, circuit design.
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. What is the Traveling Salesman Problem (TSP)?
# A1. TSP: Given N cities and distances between every pair of cities,
#     find the shortest route that visits every city exactly once and
#     returns to the starting city. It's an NP-Hard optimization problem.
#     For N cities: (N-1)!/2 possible routes (symmetric TSP).
#     For N=10: 181,440 routes. For N=20: 60.8 trillion routes.
#     Exact solutions only feasible for small N; heuristics used for large N.
#
# Q2. Why is TSP NP-Hard?
# A2. NP-Hard means no known polynomial-time algorithm exists to solve it exactly.
#     As N grows, the number of possible routes grows factorially: O(N!).
#     Even with advanced exact methods (branch-and-bound, dynamic programming),
#     solving TSP with hundreds of cities takes unreasonable time.
#     This makes heuristic/metaheuristic approaches like ACO essential for real-world TSP.
#
# Q3. What is Ant Colony Optimization?
# A3. ACO is a metaheuristic algorithm inspired by ant foraging behavior.
#     Real ants: deposit pheromone while walking; other ants preferentially
#     follow high-pheromone paths; shorter paths accumulate more pheromone
#     (more trips per time unit) → colony converges to shortest path.
#     In ACO: artificial ants construct solutions probabilistically based on
#     pheromone matrix (learned) and heuristic (distance). Pheromones updated
#     to reinforce good solutions. First introduced by Marco Dorigo (1990s).
#
# Q4. What is pheromone evaporation in ACO?
# A4. Pheromone evaporation decreases all pheromone values each iteration:
#     τ_ij ← (1 - ρ) × τ_ij  where ρ ∈ (0,1) is evaporation rate.
#     Purpose:
#     (i) Prevents unlimited pheromone accumulation on early good paths
#     (ii) Allows exploration of new paths as old ones fade
#     (iii) Enables the algorithm to forget suboptimal solutions
#     (iv) Acts as forgetting mechanism — avoids premature convergence
#     If ρ too high: too much evaporation, pheromone information lost quickly
#     If ρ too low: stagnation, ants keep following the same suboptimal path.
#
# Q5. What is the role of α (alpha) and β (beta) in ACO?
# A5. The probability of ant choosing city j from city i:
#     P(i→j) = (τ_ij^α × η_ij^β) / Σ_k(τ_ik^α × η_ik^β)
#     α (alpha): Controls weight of pheromone (τ). High α → follow pheromone trails
#     more → exploitation of known good paths. α=0 → random walk (ignores pheromone).
#     β (beta): Controls weight of heuristic (η = 1/distance). High β → greedy,
#     prefer nearby cities. β=0 → ignore distances entirely. β=1 → proportional.
#     Typical values: α=1, β=2-5. Need balance between exploration and exploitation.
#
# Q6. What is the difference between Genetic Algorithm and ACO?
# A6. Genetic Algorithm:
#     - Population of complete solutions (chromosomes)
#     - Uses crossover and mutation operators
#     - Works in genotype space (encoded solutions)
#     - Selection based on fitness
#     Ant Colony Optimization:
#     - No explicit population — ants construct solutions from scratch each iteration
#     - No crossover — uses probabilistic path construction
#     - Uses pheromone matrix as shared memory (implicit knowledge)
#     - Better suited for combinatorial problems like routing and scheduling
#     ACO: more natural for problems with discrete sequential choices (TSP, VRP)
#     GA: more general purpose, works for continuous and combinatorial problems
#
# Q7. What is pheromone deposition in ACO?
# A7. After completing tours, each ant deposits pheromone on edges it used.
#     Deposit amount: Δτ_ij = Q / L_k  where L_k = length of ant k's tour.
#     Shorter tours → larger deposit → edges in short tours get reinforced.
#     All ants deposit: τ_ij += Σ_k Δτ_ij^k
#     Elite variant: only the best ant deposits (stronger reinforcement of best path).
#     Combined with evaporation, this creates positive feedback that converges
#     the colony toward shorter routes over iterations.
#
# Q8. What are the different variants of ACO?
# A8. AS (Ant System): Original ACO — all ants deposit proportional to tour quality
#     ACS (Ant Colony System): Only best ant deposits; local pheromone update too
#     MMAS (Min-Max AS): Limits pheromone to [τ_min, τ_max] to prevent stagnation
#     RBAS (Rank-Based AS): Ants ranked by tour quality; deposit proportional to rank
#     EAS (Elitist AS): Best-so-far ant gets extra pheromone each iteration
#     Each variant improves convergence speed or solution quality for specific problems.
#
# Q9. How does ACO handle the exploration vs exploitation tradeoff?
# A9. Exploration: Finding new, potentially better routes
#     Exploitation: Following known good pheromone trails
#     ACO balances these through:
#     - Pheromone evaporation (ρ): higher ρ = more exploration (trails fade faster)
#     - α parameter: lower α = more exploration (less pheromone influence)
#     - β parameter: higher β = more exploitation of distance heuristic
#     - Random start cities: each ant explores from different starting point
#     - Stochastic selection: probabilistic (not deterministic) city choice
#
# Q10. What are real-world applications of ACO beyond TSP?
# A10. (i) Vehicle Routing Problem (VRP): delivery truck route optimization
#      (ii) Network Routing: packet routing in telecommunications networks
#      (iii) Job Shop Scheduling: machine scheduling in manufacturing
#      (iv) Feature Selection: choosing relevant features in machine learning
#      (v) Protein Structure Prediction: finding optimal protein configurations
#      (vi) Image Processing: image edge detection, segmentation
#      (vii) Electronic circuit design: component placement on PCB
#      (viii) Airline scheduling: crew scheduling and aircraft routing
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# ACO is a swarm-intelligence metaheuristic for combinatorial optimization.
# It uses distributed agents (ants) + shared pheromone memory.
#
# Core probability:
#   choose edge by pheromone influence (tau^alpha) and heuristic influence ((1/d)^beta).
#
# Why it works:
#   - Positive feedback reinforces high-quality routes.
#   - Evaporation prevents over-commitment to early suboptimal paths.
#   - Many ants explore in parallel each iteration.
#
# TSP objective:
#   shortest Hamiltonian cycle through all cities.
#
# Parameter effects:
#   - alpha high: trust pheromone history more.
#   - beta high: trust nearest-neighbor heuristic more.
#   - rho high: more forgetting/exploration.
#
# Variant insight:
#   MMAS and ACS often improve stability and solution quality in practical routing problems.
#
# Exam-ready line:
# "ACO converts collective local decisions into emergent global optimization."
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. Why is pheromone matrix symmetric in symmetric TSP?
# A11. Because distance(i,j)=distance(j,i), so desirability/deposit applies both directions.
#
# Q12. What causes stagnation in ACO?
# A12. Excessive pheromone concentration on early paths, reducing exploration diversity.
#
# Q13. How can stagnation be reduced?
# A13. Increase evaporation, cap pheromone bounds, add randomization, or use MMAS/ACS variants.
#
# Q14. Why use multiple ants instead of one?
# A14. Parallel candidate construction improves exploration and reduces bias/noise.
#
# Q15. What is the role of heuristic eta=1/distance?
# A15. It biases ants toward nearer cities, improving local greedy quality while pheromone
#      captures learned global route quality.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) TSP BASICS
#    - Visit each city exactly once and return to start.
#    - Objective: minimum total tour length.
#
# 2) ACO DECISION RULE
#    - Next city selected probabilistically using pheromone and heuristic.
#    - Not purely greedy, so exploration remains possible.
#
# 3) PHEROMONE DYNAMICS
#    - Evaporation: weakens old path memory.
#    - Deposition: reinforces edges from short tours.
#
# 4) CONVERGENCE PATTERN
#    - Initially random/diverse tours.
#    - Later iterations bias toward stronger route patterns.
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "Each ant builds a full tour using pheromone+distance probabilities, then global
#     pheromone update reinforces shorter tours; repeated cycles approximate optimal TSP route."
# ============================================================
