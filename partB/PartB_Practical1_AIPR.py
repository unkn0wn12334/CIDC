# ============================================================
# FILE: PartB_Practical1_AIPR_StructureDamage.py
# STANDALONE FILE — No other files needed.
# SUBJECT: Computational Intelligence (CI)
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# STEP 1 — Install required libraries (only once):
#   pip install numpy matplotlib scikit-learn
#
# OPTION A — Jupyter Notebook / Google Colab (RECOMMENDED):
#   1. Open colab.research.google.com → New Notebook
#   2. In first cell run: !pip install numpy matplotlib scikit-learn
#   3. Paste entire code into next cell
#   4. Press Shift+Enter
#   5. Output: Training logs, classification report, confusion matrix plot
#   6. 'aipr_results.png' saved in session directory
#
# OPTION B — PyCharm:
#   1. Open this file → install libraries if prompted
#   2. pip install numpy matplotlib scikit-learn
#   3. Click ▶ Run
#   4. Output in console, plot opens as popup window
#
# OPTION C — Terminal:
#   python PartB_Practical1_AIPR_StructureDamage.py
#
# EXPECTED OUTPUT:
#   - Synthetic dataset info (200 samples, 4 features, 3 damage classes)
#   - Training progress: accuracy improving each generation
#   - Classification Report: precision, recall, F1-score per class
#   - Confusion Matrix heatmap saved as aipr_results.png
#   - Final test accuracy (should be ~75-90%)
# ============================================================

# Import numpy for numerical operations and array handling
import numpy as np
# Import matplotlib for plotting confusion matrix and results
import matplotlib.pyplot as plt
# Import pandas for loading real CSV datasets
import pandas as pd
# Import sklearn for dataset generation, splitting, and evaluation metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
# Import random for stochastic antibody operations
import random
# Import copy for deep copying antibody objects
import copy

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================
# STEP 1: DATASET LOADING (REAL DATA FIRST, SYNTHETIC FALLBACK)
# ============================================================
print("="*60)
print("AIPR — Structural Damage Classification")
print("="*60)

def load_real_dataset():
    """
    REAL DATASET SECTION (preferred in exam/practical):
    Loads the Iris dataset from a public CSV URL and maps species labels to
    structural-damage-style classes for AIPR demonstration.
    """
    # Primary source: URL-based dataset loading
    dataset_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(dataset_url)

    # Local fallback path (keep commented; use only if URL fails)
    # df = pd.read_csv(r"U:\ci_dc_practical\data\iris.csv")

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_real = df[feature_cols].values

    # Map botanical labels to practical-friendly damage labels
    label_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
    y_real = df["species"].map(label_map).values
    return X_real, y_real

def load_synthetic_dataset():
    """
    SYNTHETIC DATASET SECTION (backup / lab simulation):
    Preserved from your original code for offline/demo use.
    """
    return make_classification(
        n_samples=200,       # 200 structural measurement samples
        n_features=4,        # 4 measurement features per sample
        n_classes=3,         # 3 damage severity classes
        n_clusters_per_class=1,
        n_informative=4,
        n_redundant=0,
        random_state=42
    )

try:
    X, y = load_real_dataset()
    print("Dataset source: REAL dataset from URL (Iris CSV)")
except Exception as e:
    print(f"URL dataset load failed ({e}). Falling back to synthetic dataset...")
    X, y = load_synthetic_dataset()
    print("Dataset source: SYNTHETIC generated dataset")

# Define class label names for clarity in reports
CLASS_NAMES = ['No Damage', 'Minor Damage', 'Severe Damage']
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {len(CLASS_NAMES)} classes")
print(f"Classes: {CLASS_NAMES}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================
# Normalize features to [0, 1] range using Min-Max scaling
# This ensures all features contribute equally (no feature dominates due to scale)
scaler = MinMaxScaler()  # Create scaler object
X_normalized = scaler.fit_transform(X)  # Fit on data and transform
print(f"\nData normalized to [0, 1] range using Min-Max scaling")

# Split dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

# ============================================================
# STEP 3: ANTIBODY REPRESENTATION
# ============================================================
class Antibody:
    """
    Represents a candidate classifier (antibody) in the AIPR system.
    Each antibody is a real-valued vector (same dimension as input features)
    paired with a class label it represents.
    Antibody vector ≈ a prototype/centroid for one class.
    """
    def __init__(self, vector, label):
        # vector: numpy array of feature values (same length as input features)
        self.vector = vector.copy()
        # label: which damage class this antibody represents (0, 1, or 2)
        self.label = label
        # affinity: how well this antibody matches its training antigens
        self.affinity = 0.0

    def clone(self):
        """Creates an independent copy of this antibody"""
        new_ab = Antibody(self.vector.copy(), self.label)
        new_ab.affinity = self.affinity
        return new_ab

# ============================================================
# STEP 4: AFFINITY (SIMILARITY) COMPUTATION
# ============================================================
def compute_affinity(antibody_vector, antigen_vector):
    """
    Computes the affinity (similarity) between an antibody and an antigen.
    Uses inverse Euclidean distance: closer = higher affinity.
    antigen: a data sample (structural measurement vector)
    antibody: a candidate classifier vector
    """
    # Compute Euclidean distance between antibody and antigen vectors
    distance = np.linalg.norm(antibody_vector - antigen_vector)
    # Convert distance to affinity: affinity = 1 / (1 + distance)
    # Higher affinity → antibody is a better match for this antigen
    affinity = 1.0 / (1.0 + distance)
    return affinity  # Returns value in (0, 1]

def classify(antibodies, antigen):
    """
    Classifies an antigen (sample) by finding the antibody with highest affinity.
    The class label of the winning antibody is the predicted class.
    This is a nearest-prototype classification approach.
    """
    best_affinity = -1      # Track highest affinity seen
    best_label = -1         # Track label of best-matching antibody

    for ab in antibodies:
        # Compute affinity of this antibody to the antigen
        aff = compute_affinity(ab.vector, antigen)
        if aff > best_affinity:
            best_affinity = aff   # Update best affinity
            best_label = ab.label  # Update predicted label

    return best_label  # Return the predicted class label

# ============================================================
# STEP 5: INITIALIZE ANTIBODY POPULATION
# ============================================================
def initialize_antibodies(X_train, y_train, n_antibodies_per_class=5):
    """
    Initializes antibody population using random training samples as starting points.
    Each class gets n_antibodies_per_class antibodies (one per prototype).
    Starting from real data points helps antibodies converge faster.
    """
    antibodies = []  # List to hold all antibodies
    n_classes = len(np.unique(y_train))  # Number of unique damage classes

    for class_label in range(n_classes):
        # Get all training samples belonging to this class
        class_samples = X_train[y_train == class_label]

        # Randomly select n_antibodies_per_class samples as initial antibodies
        selected_indices = np.random.choice(
            len(class_samples),
            size=min(n_antibodies_per_class, len(class_samples)),
            replace=False
        )
        for idx in selected_indices:
            # Create antibody from selected training sample
            ab = Antibody(class_samples[idx], class_label)
            antibodies.append(ab)

    print(f"\nInitialized {len(antibodies)} antibodies "
          f"({n_antibodies_per_class} per class × {n_classes} classes)")
    return antibodies

# ============================================================
# STEP 6: EVALUATE ANTIBODY POPULATION FITNESS
# ============================================================
def evaluate_population(antibodies, X_train, y_train):
    """
    Evaluates each antibody's affinity/fitness based on training accuracy.
    For each antibody: count how many training samples of its class it correctly
    recognizes (i.e., has higher affinity for than antibodies of other classes).
    """
    for ab in antibodies:
        # Get training samples of the same class as this antibody
        same_class_mask = (y_train == ab.label)
        same_class_samples = X_train[same_class_mask]

        if len(same_class_samples) == 0:
            ab.affinity = 0.0
            continue

        # Compute average affinity of this antibody to its own class samples
        affinities = [compute_affinity(ab.vector, sample)
                      for sample in same_class_samples]
        ab.affinity = np.mean(affinities)  # Average affinity = fitness

    return antibodies

# ============================================================
# STEP 7: CLONAL SELECTION AND HYPERMUTATION FOR AIPR
# ============================================================
def aipr_train(X_train, y_train, n_antibodies_per_class=5,
               n_generations=30, clone_factor=3, mutation_rate=0.1):
    """
    Main AIPR training loop.
    Evolves antibodies through cloning, mutation, and selection to
    build an accurate classifier for structural damage classes.
    """
    print("\n" + "="*60)
    print("AIPR TRAINING — Evolving Antibody Classifiers")
    print(f"Antibodies per class: {n_antibodies_per_class}")
    print(f"Generations: {n_generations}, Clone Factor: {clone_factor}")
    print("="*60)

    # INITIALIZATION: Create initial antibody population from training data
    antibodies = initialize_antibodies(X_train, y_train, n_antibodies_per_class)

    # Track training accuracy history for convergence plot
    accuracy_history = []

    # MAIN AIPR EVOLUTION LOOP
    for generation in range(n_generations):

        # EVALUATION: Compute affinity for all antibodies on training data
        antibodies = evaluate_population(antibodies, X_train, y_train)

        # Compute current training accuracy for monitoring
        predictions = [classify(antibodies, sample) for sample in X_train]
        accuracy = np.mean(np.array(predictions) == y_train)
        accuracy_history.append(accuracy)

        # Print progress every 5 generations
        if generation % 5 == 0 or generation == n_generations - 1:
            avg_aff = np.mean([ab.affinity for ab in antibodies])
            print(f"Gen {generation+1:>3}: Train Accuracy={accuracy*100:.1f}%, "
                  f"Avg Affinity={avg_aff:.4f}")

        # CLONING: Clone antibodies proportional to their affinity
        clones = []
        for ab in sorted(antibodies, key=lambda x: x.affinity, reverse=True):
            # Number of clones proportional to affinity rank
            n_clones = max(1, int(clone_factor * ab.affinity * 10))
            for _ in range(n_clones):
                clones.append(ab.clone())  # Add clones to list

        # HYPERMUTATION: Mutate clones — higher affinity = lower mutation
        for clone in clones:
            # Mutation rate inversely proportional to affinity
            mut_strength = mutation_rate * (1.0 - clone.affinity)
            # Add Gaussian noise to each feature of the clone vector
            noise = np.random.normal(0, mut_strength, size=clone.vector.shape)
            clone.vector = clone.vector + noise  # Apply perturbation
            # Clip to valid normalized range [0, 1]
            clone.vector = np.clip(clone.vector, 0, 1)

        # SELECTION: Evaluate clones and keep best per class
        all_candidates = antibodies + clones  # Combine original + clones
        # Re-evaluate all candidates
        all_candidates = evaluate_population(all_candidates, X_train, y_train)

        # Keep top n_antibodies_per_class antibodies per class
        new_antibodies = []
        n_classes = len(np.unique(y_train))
        for class_label in range(n_classes):
            # Get antibodies for this class
            class_abs = [ab for ab in all_candidates if ab.label == class_label]
            # Sort by affinity descending and keep top ones
            class_abs.sort(key=lambda x: x.affinity, reverse=True)
            new_antibodies.extend(class_abs[:n_antibodies_per_class])

        antibodies = new_antibodies  # Update population

    print(f"\nTraining complete! Final antibody count: {len(antibodies)}")
    return antibodies, accuracy_history

# ============================================================
# STEP 8: TESTING AND EVALUATION
# ============================================================
def evaluate_classifier(antibodies, X_test, y_test):
    """
    Tests the trained AIPR classifier on unseen test data.
    Computes accuracy, classification report, and confusion matrix.
    """
    # Classify all test samples using trained antibodies
    predictions = [classify(antibodies, sample) for sample in X_test]
    predictions = np.array(predictions)

    # Compute overall test accuracy
    accuracy = np.mean(predictions == y_test) * 100
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Print detailed classification report (precision, recall, F1)
    print(f"\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES))

    # Compute confusion matrix
    cm = confusion_matrix(y_test, predictions)
    return predictions, cm, accuracy

# ============================================================
# MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    # TRAIN the AIPR classifier
    trained_antibodies, acc_history = aipr_train(
        X_train, y_train,
        n_antibodies_per_class=5,   # 5 antibodies per damage class
        n_generations=30,           # 30 evolution cycles
        clone_factor=3,             # Clone factor for reproduction
        mutation_rate=0.15          # Mutation strength parameter
    )

    # TEST on unseen data
    preds, conf_matrix, test_acc = evaluate_classifier(
        trained_antibodies, X_test, y_test)

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training accuracy convergence over generations
    axes[0].plot(range(1, len(acc_history)+1), [a*100 for a in acc_history],
                 'b-o', linewidth=2, markersize=4)
    axes[0].set_title('AIPR Training Accuracy over Generations', fontsize=13)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Training Accuracy (%)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 105])

    # Plot 2: Confusion Matrix heatmap
    im = axes[1].imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title(f'Confusion Matrix (Test Acc: {test_acc:.1f}%)', fontsize=13)
    axes[1].set_xticks(range(len(CLASS_NAMES)))
    axes[1].set_yticks(range(len(CLASS_NAMES)))
    axes[1].set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    axes[1].set_yticklabels(CLASS_NAMES)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    plt.colorbar(im, ax=axes[1])
    # Add count values inside confusion matrix cells
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            axes[1].text(j, i, str(conf_matrix[i, j]),
                        ha='center', va='center',
                        color='white' if conf_matrix[i,j] > conf_matrix.max()/2 else 'black',
                        fontsize=14, fontweight='bold')

    plt.suptitle('AIPR — Structural Damage Classification Results',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('aipr_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResults saved as 'aipr_results.png'")


# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Synthetic dataset: 200 samples, 4 features, 3 damage classes generated
# 2. Preprocessing: Min-Max scaling normalizes all features to [0, 1]
# 3. Antibody Initialization: Random training samples chosen as initial antibodies
#    (5 antibodies per class = 15 total starting antibodies)
# 4. Affinity: Inverse Euclidean distance — closer antibody = higher affinity
# 5. Evaluation: Average affinity of each antibody to its own class samples
# 6. Cloning: High-affinity antibodies create more clones (better solutions replicate)
# 7. Hypermutation: Clones perturbed with Gaussian noise inversely proportional to affinity
#    — good antibodies mutate less (fine-tune), bad ones mutate more (explore)
# 8. Selection: Top-k antibodies per class kept for next generation
# 9. Classification: New sample classified by finding antibody with max affinity
# 10. Evaluation: Accuracy, precision, recall, F1, confusion matrix on test set
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: Artificial Immune Pattern Recognition (AIPR)
# Inspired by how the human immune system recognizes and responds to pathogens.
# Key biological concepts mapped to classification:
#   Antigen = Input data sample (structural measurement)
#   Antibody = Classifier prototype (learned representative of a class)
#   Affinity = Similarity between antibody and antigen (inverse distance)
#   Clonal Selection = Best classifiers reproduce more
#   Hypermutation = Random variations to explore better classifiers
#   Affinity Maturation = Gradual improvement of classifier quality
# Application: Classify structural damage (no/minor/severe) from sensor readings.
# Real use: bridges, buildings, aircraft, pipelines — structural health monitoring.
# AIS (Artificial Immune Systems) also used for: anomaly detection, spam filtering,
# fault detection in industrial systems, computer security (intrusion detection).
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. Describe AIS (Artificial Immune System).
# A1. AIS is a class of computational intelligence algorithms inspired by the
#     vertebrate immune system. Key characteristics:
#     - Pattern recognition without prior knowledge of all patterns
#     - Self/non-self discrimination (normal vs anomalous)
#     - Memory: remembers previously encountered patterns for faster future response
#     - Distributed: no central controller — many immune cells work together
#     - Adaptive: learns and improves over time through exposure
#     Three main AIS paradigms: Clonal Selection, Negative Selection, Immune Networks.
#
# Q2. Explain Data Representation in AIS.
# A2. In AIS for pattern recognition:
#     Antigen: Input data sample represented as a real-valued feature vector.
#              Each element = one measurement feature (e.g., vibration frequency).
#     Antibody: Candidate classifier stored as a real-valued vector of same dimension.
#              Represents a learned prototype for one class.
#     Affinity: Numerical measure of how well antibody matches antigen.
#              Common metrics: Euclidean distance, Hamming distance, cosine similarity.
#     Shape Space: The high-dimensional space where both antigens and antibodies live.
#
# Q3. Define Data Cleaning and Feature Extraction.
# A3. Data Cleaning: Process of fixing/removing incorrect, incomplete, or noisy data.
#     Steps: Remove duplicates, handle missing values (imputation/removal),
#     filter outliers, correct inconsistent formats, remove irrelevant columns.
#     Feature Extraction: Transforming raw data into informative numerical features.
#     For structural damage: extract vibration frequency components, statistical
#     measures (mean, variance, RMS), frequency domain features via FFT,
#     strain gauge readings, acceleration measurements.
#
# Q4. Why is there a need for AIS?
# A4. AIS is needed because:
#     (i) Handles high-dimensional pattern recognition without labeled training data
#     (ii) Self-organizing — doesn't need predefined decision boundaries
#     (iii) Robust to noise and partial data (like the immune system)
#     (iv) Handles evolving patterns — can learn new patterns over time
#     (v) Intrinsically multi-class — naturally handles multiple categories
#     (vi) Anomaly detection without knowing what anomalies look like in advance
#     Traditional ML needs lots of labeled data; AIS can work with less.
#
# Q5. What is affinity maturation in the context of AIPR?
# A5. Affinity maturation is the iterative improvement process where:
#     - Antibodies that match the target pattern (antigen) well get cloned
#     - Clones undergo hypermutation (random small changes)
#     - Mutated clones with higher affinity are selected to survive
#     - Over generations, antibody affinity for the target pattern increases
#     In AIPR: classifier prototypes gradually move closer to cluster centers
#     of each damage class, improving classification accuracy over generations.
#
# Q6. What is Negative Selection in AIS and how does it differ from Clonal Selection?
# A6. Negative Selection: Inspired by T-cell maturation in the thymus.
#     Generates detectors that do NOT match self (normal patterns).
#     Used for anomaly detection — any pattern a detector matches = anomaly.
#     Clonal Selection: Amplifies detectors that DO match antigens (target patterns).
#     Used for classification and optimization.
#     Key difference: Negative selection learns "what is normal" and flags deviations.
#     Clonal selection learns "what each class looks like" for multi-class classification.
#
# Q7. What evaluation metrics are used for AIPR classification?
# A7. Standard classification metrics:
#     Accuracy: (TP+TN)/(TP+TN+FP+FN) — overall correct predictions
#     Precision: TP/(TP+FP) — of predicted positive, how many actually positive
#     Recall: TP/(TP+FN) — of actual positive, how many correctly predicted
#     F1-Score: 2*(Precision*Recall)/(Precision+Recall) — harmonic mean
#     Confusion Matrix: Table showing actual vs predicted for all classes
#     For structural damage: recall is critical — missing severe damage is dangerous.
#
# Q8. How does AIPR compare to traditional classifiers like SVM or KNN?
# A8. AIPR vs KNN: Both use distance-based classification. AIPR uses evolved
#     prototypes (fewer reference points) while KNN uses all training samples.
#     AIPR is faster at test time with fewer antibodies than training samples.
#     AIPR vs SVM: SVM finds optimal hyperplane; AIPR uses prototype matching.
#     AIPR is more interpretable and naturally handles incremental learning.
#     AIPR can update prototypes without full retraining (immune memory).
#
# Q9. What preprocessing steps are important for AIPR?
# A9. (i) Normalization: Scale all features to [0,1] or [-1,1] so distance
#         computation isn't dominated by large-scale features
#     (ii) Feature Extraction: Select/derive informative features from raw data
#     (iii) Dimensionality Reduction: PCA to reduce computational cost
#     (iv) Data Balancing: Ensure equal class representation (oversampling/SMOTE)
#     (v) Noise Removal: Filter outliers that could corrupt antibody evolution
#
# Q10. What are real-world applications of AIPR in structural damage classification?
# A10. (i) Bridge health monitoring: Detect cracks from strain sensor data
#      (ii) Aircraft structural integrity: Vibration signature analysis
#      (iii) Building damage post-earthquake: Accelerometer data classification
#      (iv) Pipeline leak detection: Pressure and flow anomaly classification
#      (v) Wind turbine blade damage: Acoustic emission pattern recognition
#      (vi) Railway track defect detection: Wheel-rail interaction signatures
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# AIPR (Artificial Immune Pattern Recognition):
#   Uses immune-inspired adaptive prototypes (antibodies) to classify patterns (antigens).
#
# Working intuition:
#   - Each class has antibodies as representative patterns.
#   - New sample is assigned to antibody with highest affinity.
#   - Training evolves antibodies by cloning strong ones and mutating copies.
#
# Why it is useful:
#   - Adaptive learning under noisy measurements.
#   - Prototype-based interpretability.
#   - Good for evolving environments and anomaly-aware systems.
#
# Pipeline in this practical:
#   dataset -> normalize -> initialize class-wise antibodies ->
#   evaluate affinity -> clone/mutate/select -> test metrics.
#
# Structural monitoring relevance:
#   sensor streams are noisy/nonlinear; immune-inspired adaptive classifiers can
#   update and remain robust when signal distributions drift.
#
# CI perspective:
#   AIPR sits under Artificial Immune Systems with clonal selection and negative selection.
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. Why normalize features before affinity computation?
# A11. Distance-based affinity is scale-sensitive; normalization prevents one large-scale
#      feature from dominating similarity.
#
# Q12. What is prototype-based classification advantage?
# A12. It is interpretable, memory-efficient, and fast at inference compared to storing all data.
#
# Q13. How does class imbalance affect AIPR?
# A13. Majority classes may dominate antibody evolution; use stratified splits, balanced
#      initialization, or weighted sampling.
#
# Q14. What if two antibodies from different classes have similar affinity?
# A14. Tie-breakers or confidence thresholds can be added; otherwise nearest affinity decides.
#
# Q15. How can this model be made online/incremental?
# A15. Periodically update antibody pool with new labeled samples, re-evaluate affinity,
#      and perform bounded clone-mutation updates without full retraining.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) AIPR CLASSIFICATION INTUITION
#    - Each antibody is a class prototype.
#    - Sample gets label of highest-affinity antibody.
#
# 2) TRAINING MECHANISM
#    - Evaluate antibody affinity on class data.
#    - Clone stronger antibodies.
#    - Mutate clones to search better prototype locations.
#    - Keep best per class to form next generation.
#
# 3) WHY THIS IS ROBUST
#    - Prototype evolution adapts to noisy or shifting patterns.
#    - Controlled mutation helps balance exploration and stability.
#
# 4) METRICS TO EXPLAIN IN VIVA
#    - Accuracy: overall correctness.
#    - Precision: correctness of positive predictions.
#    - Recall: ability to detect true class instances.
#    - F1-score: balance of precision and recall.
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "AIPR models immune learning by evolving class prototypes (antibodies) via
#     clonal selection and hypermutation, then classifies using highest affinity."
# ============================================================
