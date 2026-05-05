# ============================================================
# FILE: PartA_Practical3_FuzzySet.py
# STANDALONE FILE — No other files needed.
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# STEP 1 — Install required libraries (only once):
#   Open terminal and run:
#   pip install numpy matplotlib
#   (These are the only two dependencies needed)
#
# OPTION A — Jupyter Notebook / Google Colab (RECOMMENDED):
#   1. Open Jupyter: run "jupyter notebook" in terminal, then open a new notebook
#      OR go to colab.research.google.com and create a new notebook
#   2. Paste the entire code into a cell
#   3. Press Shift+Enter to run
#   4. All output tables will print below the cell
#   5. A figure with 6 subplots will appear inline showing all operations
#   6. File 'fuzzy_operations.png' will be saved in the current directory
#
# OPTION B — PyCharm:
#   1. Open PyCharm → Open this file
#   2. If numpy/matplotlib not installed: PyCharm will show a yellow warning bar
#      Click "Install requirements" OR open terminal inside PyCharm and run:
#      pip install numpy matplotlib
#   3. Click ▶ Run — output prints in Run console at bottom
#   4. Matplotlib figure opens as a separate popup window
#   5. PNG file saves in the project root directory
#
# OPTION C — Terminal:
#   python PartA_Practical3_FuzzySet.py
#   → Output prints to terminal, plot opens in a window, PNG saved locally
#
# EXPECTED OUTPUT:
#   - Table showing U, μA, μB, A∪B, A∩B, Ā, A-B values for all elements
#   - Cartesian product matrix R (3×3)
#   - Max-Min composition matrix T (3×2)
#   - Figure with 6 plots (saved as fuzzy_operations.png)
# ============================================================

# Import numpy for numerical computations and array operations
import numpy as np
# Import matplotlib for plotting fuzzy set membership graphs
import matplotlib.pyplot as plt

# ============================================================
# PART 1: DEFINE THE UNIVERSE OF DISCOURSE AND FUZZY SETS
# ============================================================

# Define the Universe of Discourse U — all possible values (0 to 10)
U = np.arange(0, 11, 1)  # Universe: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Universe of Discourse U:", U)

# Define Fuzzy Set A with membership values for each element in U
# Membership values must be between 0.0 and 1.0
# Example: A represents "small numbers" — higher membership for smaller values
A = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
print("\nFuzzy Set A (membership values):", A)

# Define Fuzzy Set B with membership values for each element in U
# Example: B represents "medium numbers" — higher membership for middle values
B = np.array([0.0, 0.0, 0.2, 0.4, 0.6, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
print("Fuzzy Set B (membership values):", B)

# ============================================================
# PART 2: FUZZY SET OPERATIONS
# ============================================================

# --- UNION (A ∪ B) ---
# Union takes the MAXIMUM membership value at each point
# Formula: μ(A∪B)(x) = max(μA(x), μB(x))
A_union_B = np.maximum(A, B)  # Element-wise maximum of A and B
print("\n--- UNION (A ∪ B) ---")
print("Formula: max(μA(x), μB(x))")
print("Result:", A_union_B)

# --- INTERSECTION (A ∩ B) ---
# Intersection takes the MINIMUM membership value at each point
# Formula: μ(A∩B)(x) = min(μA(x), μB(x))
A_intersect_B = np.minimum(A, B)  # Element-wise minimum of A and B
print("\n--- INTERSECTION (A ∩ B) ---")
print("Formula: min(μA(x), μB(x))")
print("Result:", A_intersect_B)

# --- COMPLEMENT (NOT A) ---
# Complement inverts the membership value
# Formula: μ(Ā)(x) = 1 - μA(x)
A_complement = 1 - A  # Subtract each membership value from 1
print("\n--- COMPLEMENT (NOT A = Ā) ---")
print("Formula: 1 - μA(x)")
print("Result:", A_complement)

B_complement = 1 - B  # Also compute complement of B (needed for difference)
print("\nComplement of B:", B_complement)

# --- DIFFERENCE (A - B) ---
# Difference = A AND (NOT B) = Intersection of A and complement of B
# Formula: μ(A\B)(x) = min(μA(x), 1 - μB(x))
A_minus_B = np.minimum(A, B_complement)  # min(μA, 1-μB) at each point
print("\n--- DIFFERENCE (A - B = A ∩ B̄) ---")
print("Formula: min(μA(x), 1 - μB(x))")
print("Result:", A_minus_B)

# ============================================================
# PART 3: DISPLAY TABLE OF ALL OPERATIONS
# ============================================================
print("\n" + "="*75)
print(f"{'x':>3} | {'μA':>6} | {'μB':>6} | {'A∪B':>6} | {'A∩B':>6} | {'Ā':>6} | {'A-B':>6}")
print("="*75)
for i in range(len(U)):
    # Print each element and its membership values under each operation
    print(f"{U[i]:>3} | {A[i]:>6.2f} | {B[i]:>6.2f} | {A_union_B[i]:>6.2f} | "
          f"{A_intersect_B[i]:>6.2f} | {A_complement[i]:>6.2f} | {A_minus_B[i]:>6.2f}")
print("="*75)

# ============================================================
# PART 4: FUZZY RELATIONS — CARTESIAN PRODUCT
# ============================================================
print("\n\n" + "="*50)
print("PART 4: FUZZY RELATIONS — CARTESIAN PRODUCT (A × B)")
print("="*50)

# Define two new smaller fuzzy sets for relation computation
# Universe V for set P (for demonstration)
P = np.array([0.2, 0.5, 0.8])  # Fuzzy set P on universe {1, 2, 3}
Q = np.array([0.4, 0.7, 1.0])  # Fuzzy set Q on universe {a, b, c}

print("Fuzzy Set P:", P)
print("Fuzzy Set Q:", Q)

# Compute Cartesian Product: creates a relation matrix R
# For each pair (p, q), μR(p, q) = min(μP(p), μQ(q))
# This creates a matrix where rows correspond to P and columns to Q
R = np.zeros((len(P), len(Q)))  # Initialize relation matrix with zeros

# Fill the relation matrix using nested loops
for i in range(len(P)):         # Iterate over each element of P
    for j in range(len(Q)):     # Iterate over each element of Q
        # Membership of relation = minimum of the two memberships (AND-like operation)
        R[i][j] = min(P[i], Q[j])

print("\nFuzzy Relation R = P × Q (Cartesian Product):")
print("Formula: μR(p,q) = min(μP(p), μQ(q))")
print("Relation Matrix R:")
print(R)

# ============================================================
# PART 5: MAX-MIN COMPOSITION
# ============================================================
print("\n" + "="*50)
print("PART 5: MAX-MIN COMPOSITION (R ∘ S)")
print("="*50)

# Define second fuzzy relation S on universes Q × W
# R is on P×Q, S is on Q×W, composition T = R∘S will be on P×W
S = np.array([
    [0.3, 0.6],   # Row for q1: membership values for each w in W
    [0.8, 0.4],   # Row for q2
    [0.5, 0.9]    # Row for q3
])

print("Relation R (P × Q):\n", R)
print("\nRelation S (Q × W):\n", S)

# Compute Max-Min Composition: T = R ∘ S
# For each pair (p, w):
#   1. For each intermediate v in Q: compute min(μR(p,v), μS(v,w))
#   2. Take max over all v: μT(p,w) = max_v { min(μR(p,v), μS(v,w)) }
rows_R = R.shape[0]  # Number of rows in R (size of P)
cols_S = S.shape[1]  # Number of columns in S (size of W)
len_Q = len(Q)       # Length of intermediate domain Q

# Initialize the composition result matrix T
T = np.zeros((rows_R, cols_S))

for i in range(rows_R):         # For each element p in P
    for j in range(cols_S):     # For each element w in W
        # Compute intermediate min values for all v in Q
        min_values = []
        for k in range(len_Q):  # For each intermediate element v in Q
            # Step 1: Compute min of R[p,v] and S[v,w]
            min_val = min(R[i][k], S[k][j])
            min_values.append(min_val)  # Store intermediate result
        # Step 2: Take maximum over all intermediate min values
        T[i][j] = max(min_values)

print("\nMax-Min Composition T = R ∘ S (P × W):")
print("Formula: μT(p,w) = max_v { min(μR(p,v), μS(v,w)) }")
print("Result Matrix T:\n", T)

# ============================================================
# PART 6: VISUALIZATION — PLOT FUZZY SET OPERATIONS
# ============================================================
# Create a figure with multiple subplots for visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns of subplots

# Plot Fuzzy Set A and B on first subplot
axes[0, 0].plot(U, A, 'b-o', label='Set A', linewidth=2)  # Blue line with circles
axes[0, 0].plot(U, B, 'r-s', label='Set B', linewidth=2)  # Red line with squares
axes[0, 0].set_title('Fuzzy Sets A and B')
axes[0, 0].set_xlabel('Universe U')
axes[0, 0].set_ylabel('Membership μ')
axes[0, 0].legend()
axes[0, 0].grid(True)  # Add grid for readability
axes[0, 0].set_ylim([-0.1, 1.1])  # Set y-axis limits

# Plot Union
axes[0, 1].fill_between(U, A_union_B, alpha=0.3, color='green')  # Shade under curve
axes[0, 1].plot(U, A_union_B, 'g-^', linewidth=2)
axes[0, 1].set_title('Union A ∪ B (max)')
axes[0, 1].set_xlabel('Universe U')
axes[0, 1].set_ylabel('Membership μ')
axes[0, 1].grid(True)
axes[0, 1].set_ylim([-0.1, 1.1])

# Plot Intersection
axes[0, 2].fill_between(U, A_intersect_B, alpha=0.3, color='purple')
axes[0, 2].plot(U, A_intersect_B, 'm-v', linewidth=2)
axes[0, 2].set_title('Intersection A ∩ B (min)')
axes[0, 2].set_xlabel('Universe U')
axes[0, 2].set_ylabel('Membership μ')
axes[0, 2].grid(True)
axes[0, 2].set_ylim([-0.1, 1.1])

# Plot Complement
axes[1, 0].fill_between(U, A_complement, alpha=0.3, color='orange')
axes[1, 0].plot(U, A_complement, 'y-D', linewidth=2, color='darkorange')
axes[1, 0].set_title('Complement Ā (1 - μA)')
axes[1, 0].set_xlabel('Universe U')
axes[1, 0].set_ylabel('Membership μ')
axes[1, 0].grid(True)
axes[1, 0].set_ylim([-0.1, 1.1])

# Plot Difference
axes[1, 1].fill_between(U, A_minus_B, alpha=0.3, color='red')
axes[1, 1].plot(U, A_minus_B, 'r-P', linewidth=2)
axes[1, 1].set_title('Difference A - B (min(A, B̄))')
axes[1, 1].set_xlabel('Universe U')
axes[1, 1].set_ylabel('Membership μ')
axes[1, 1].grid(True)
axes[1, 1].set_ylim([-0.1, 1.1])

# Plot Cartesian Product (as heatmap)
im = axes[1, 2].imshow(R, cmap='Blues', aspect='auto')  # Display as color matrix
axes[1, 2].set_title('Cartesian Product R = P × Q')
axes[1, 2].set_xlabel('Q elements')
axes[1, 2].set_ylabel('P elements')
plt.colorbar(im, ax=axes[1, 2])  # Add color scale bar
# Add values inside each cell of the heatmap
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        axes[1, 2].text(j, i, f'{R[i,j]:.2f}', ha='center', va='center', color='black')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('fuzzy_operations.png', dpi=150, bbox_inches='tight')  # Save as PNG file
plt.show()  # Display the plot
print("\nPlot saved as 'fuzzy_operations.png'")


# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Define Universe of Discourse U = {0,1,...,10} and two fuzzy sets A and B
# 2. Union: Apply np.maximum() element-wise — gives the OR of two fuzzy sets
# 3. Intersection: Apply np.minimum() element-wise — gives the AND of two fuzzy sets
# 4. Complement: Subtract each membership from 1 — inverts the set
# 5. Difference: Compute A AND (NOT B) = min(A, 1-B) element-wise
# 6. Cartesian Product: Build relation matrix R where R[i][j] = min(P[i], Q[j])
#    This creates a 2D fuzzy relation representing all (p,q) pairs
# 7. Max-Min Composition: Compose R and S matrices using max-of-min rule
#    For each (p,w) pair: find max over all intermediate v of min(R[p,v], S[v,w])
# 8. Visualize all operations using matplotlib subplots
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: Fuzzy Sets and Fuzzy Relations
# Fuzzy Logic extends classical binary logic to handle partial truth.
# In classical sets: an element either belongs (1) or doesn't (0).
# In fuzzy sets: membership is a real number between 0 and 1.
# Key operations:
#   Union: OR — take maximum (most belongs)
#   Intersection: AND — take minimum (least certain belonging)
#   Complement: NOT — invert the degree of belonging
#   Difference: A except B — elements belonging to A but not B
# Fuzzy Relations extend fuzzy sets to multiple dimensions.
# Cartesian Product creates 2D relations from two 1D fuzzy sets.
# Max-Min Composition chains two relations to create a third.
# Applications: Control systems, image processing, decision making, AI.
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. What is a Fuzzy Set? How is it different from a classical set?
# A1. A Fuzzy Set allows partial membership — each element has a degree of
#     membership between 0 and 1. Classical set: element is either IN (1) or OUT (0).
#     Fuzzy set: element can be "partly in" (e.g., 0.7).
#     Example: "tall" — a 5'10" person may have membership 0.7 in the "tall" set.
#
# Q2. How is Union of two fuzzy sets performed?
# A2. μ(A∪B)(x) = max(μA(x), μB(x)) for each element x.
#     Takes the maximum membership at each point.
#     Represents the OR operation in fuzzy logic.
#     Example: if μA(5)=0.6 and μB(5)=0.8, then μ(A∪B)(5) = 0.8
#
# Q3. How is Intersection of two fuzzy sets performed?
# A3. μ(A∩B)(x) = min(μA(x), μB(x)) for each element x.
#     Takes the minimum membership at each point.
#     Represents the AND operation in fuzzy logic.
#     Example: if μA(5)=0.6 and μB(5)=0.8, then μ(A∩B)(5) = 0.6
#
# Q4. How do you calculate the Complement of a fuzzy set?
# A4. μ(Ā)(x) = 1 - μA(x) for each element x.
#     Inverts the membership degree. Represents NOT in fuzzy logic.
#     Example: if μA(5) = 0.7, then μ(Ā)(5) = 1 - 0.7 = 0.3
#
# Q5. What is the Difference operation (A - B) in fuzzy sets?
# A5. A - B represents elements belonging to A but not B.
#     Formula: μ(A\B)(x) = min(μA(x), 1 - μB(x))
#     = Intersection of A with the complement of B.
#     Elements strongly in A but weakly in B get high membership in A-B.
#
# Q6. What is a Fuzzy Relation?
# A6. A Fuzzy Relation is a fuzzy set defined on a Cartesian product of universes.
#     It captures the degree of relationship between elements from different universes.
#     Represented as a matrix where R[i][j] = degree that element i relates to element j.
#     Example: "is close to" relation on numbers.
#
# Q7. How is a Fuzzy Relation created using Cartesian Product?
# A7. For fuzzy sets A on U and B on V:
#     R = A × B (Cartesian product)
#     μR(u, v) = min(μA(u), μB(v)) for each pair (u, v) in U × V
#     Creates a matrix capturing the relationship strength for every element pair.
#
# Q8. Explain Max-Min Composition.
# A8. Combines two relations R (on U×V) and S (on V×W) to get T (on U×W).
#     For each pair (u, w):
#       Step 1: For each intermediate v: compute min(μR(u,v), μS(v,w))
#       Step 2: Take maximum of all these min values
#     μT(u,w) = max_v { min(μR(u,v), μS(v,w)) }
#     The min finds the "bottleneck" of each path; max finds the "best path".
#
# Q9. Why do we use max and min in Max-Min composition?
# A9. Min: Represents the weakest link — the connection strength along a path
#         is limited by the weakest step (like a chain).
#     Max: Takes the best alternative path — if multiple paths exist,
#         we choose the one with the strongest overall connection.
#     Together they model how well two things are indirectly related via an intermediate.
#
# Q10. What are practical applications of Fuzzy Logic?
# A10. (i) Fuzzy controllers: washing machines, air conditioners, camera focusing
#      (ii) Medical diagnosis: decision support under uncertainty
#      (iii) Image processing: edge detection, noise reduction
#      (iv) Natural language processing: handling ambiguous terms
#      (v) Traffic control systems, elevator control, stock market prediction
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# Fuzzy logic solves "partial truth" problems where crisp yes/no logic is too rigid.
# Example: "temperature is hot" is not binary; it changes gradually.
#
# Three foundational pieces:
#   1) Fuzzification: crisp input -> membership values.
#   2) Inference: apply fuzzy IF-THEN rules.
#   3) Defuzzification: fuzzy output -> crisp actionable value.
#
# Membership function types:
#   - Triangular, trapezoidal, Gaussian, sigmoid.
#   Choice depends on smoothness and domain behavior.
#
# Operators:
#   - AND often min/product
#   - OR often max/probabilistic sum
#   - NOT usually 1-mu
#
# Why max-min composition matters:
#   It combines chained fuzzy relations and is heavily used in fuzzy inference systems.
#
# Important distinction:
#   - Probability handles uncertainty of occurrence.
#   - Fuzziness handles vagueness/degree of belonging.
#
# One-line viva point:
# "Fuzzy systems are interpretable and robust for linguistic control problems."
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. What is a membership function?
# A11. A curve mapping each input value to a membership degree in [0,1] for a fuzzy concept.
#
# Q12. What is defuzzification and why needed?
# A12. It converts fuzzy output sets into a single crisp value for real control/action.
#      Common method: centroid (center of area).
#
# Q13. What is the centroid defuzzification formula idea?
# A13. Weighted average of output universe values by membership strengths.
#      It gives a balanced crisp output.
#
# Q14. Why are fuzzy systems used in control engineering?
# A14. They model expert linguistic rules and handle noisy/nonlinear systems without
#      requiring precise mathematical models.
#
# Q15. What is the difference between Mamdani and Sugeno fuzzy systems?
# A15. Mamdani outputs fuzzy sets (then defuzzify); Sugeno outputs mathematical
#      functions/constants and is computationally efficient for optimization/control.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) WHY FUZZY LOGIC WAS CREATED
#    - Real-world terms (hot, tall, fast, risky) are vague.
#    - Binary logic (0/1) cannot model gradual transitions naturally.
#
# 2) MEMBERSHIP FUNCTION INTERPRETATION
#    - mu(x)=0 means no membership.
#    - mu(x)=1 means full membership.
#    - mu(x)=0.4 means partial membership, not probability.
#
# 3) OPERATION INTUITION
#    - Union max: choose strongest belonging among sets.
#    - Intersection min: choose weakest agreement among sets.
#    - Complement 1-mu: inverse degree.
#
# 4) RELATIONS AND COMPOSITION
#    - Fuzzy relation matrix stores degree of relation for each pair.
#    - Max-min composition chains two fuzzy relations (like two-step reasoning).
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "This program defines fuzzy sets, performs standard fuzzy operations, constructs
#     fuzzy relation matrices using min operator, and composes relations with max-min
#     rule to demonstrate fuzzy inference fundamentals."
# ============================================================
