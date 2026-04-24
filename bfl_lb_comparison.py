"""
Title 4: BFL-LB vs Centralized Scheduler
Fault-Tolerant and Secure Dynamic Load Balancing Using Blockchain-Enabled Federated Learning

Primary Algorithm: BFL-LB (Blockchain Federated Learning-Based Load Balancer)
Secondary Algorithm: Centralized Scheduler
Additional Baselines: Non-Secure Distributed, Heuristic Security

Dataset: Load Balancing Improved.csv (10,678 rows)
Methodology: 10-fold stratified cross-validation
Metric: Accuracy (binary classification)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Load Balancing Improved.csv')
print(f"Dataset loaded: {len(df)} rows")

# Prepare features and target
# Drop timestamp column if present
if 'timestamp' in df.columns:
    df = df.drop('timestamp', axis=1)

X = df.iloc[:, :-1]  # All columns except last
y = df.iloc[:, -1]   # Last column as target

# Convert target to binary if needed
y = pd.to_numeric(y, errors='coerce').fillna(0)
if y.nunique() > 2:
    y = (y > y.median()).astype(int)
else:
    y = y.astype(int)

# Initialize cross-validation and storage
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
bfl_scores = []
centralized_scores = []
nonsecure_scores = []
heuristic_scores = []

print("Starting 10-fold evaluation...\n")

# 10-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Fold {fold}: ", end="", flush=True)
    
    # ===== BFL-LB: Blockchain Federated Learning-Based Load Balancer =====
    # MLP as proxy for federated learning with blockchain consensus
    print("BFL-LB... ", end="", flush=True)
    bfl_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu", solver="adam", learning_rate_init=0.001,
        max_iter=150, early_stopping=True, n_iter_no_change=10,
        random_state=RANDOM_STATE + fold
    )
    bfl_model.fit(X_train_scaled, y_train)
    bfl_pred = bfl_model.predict(X_test_scaled)
    bfl_acc = np.mean(bfl_pred == y_test) * 100
    bfl_scores.append(bfl_acc)
    
    # ===== Centralized Scheduler =====
    # Gradient-based linear classifier (centralized decision making)
    print("Centralized... ", end="", flush=True)
    centralized_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE + fold)
    centralized_model.fit(X_train_scaled, y_train)
    centralized_pred = centralized_model.predict(X_test_scaled)
    centralized_acc = np.mean(centralized_pred == y_test) * 100
    centralized_scores.append(centralized_acc)
    
    # ===== Non-Secure Distributed Scheduler =====
    # RandomForest without security considerations
    print("Non-Secure... ", end="", flush=True)
    nonsecure_model = RandomForestClassifier(
        n_estimators=60, max_depth=8, n_jobs=-1,
        random_state=RANDOM_STATE + fold
    )
    nonsecure_model.fit(X_train, y_train)
    nonsecure_pred = nonsecure_model.predict(X_test)
    nonsecure_acc = np.mean(nonsecure_pred == y_test) * 100
    nonsecure_scores.append(nonsecure_acc)
    
    # ===== Heuristic Security Scheduler =====
    # Rule-based approach: median thresholds on task and network features
    print("Heuristic-Security... ", end="", flush=True)
    
    cpu_threshold = X_train.iloc[:, 1].median()
    mem_threshold = X_train.iloc[:, 2].median()
    latency_threshold = X_train.iloc[:, 3].median()
    
    # Heuristic: High CPU + High Memory + Low Latency → 1, else 0
    heuristic_pred = np.where(
        (X_test.iloc[:, 1] > cpu_threshold) & 
        (X_test.iloc[:, 2] > mem_threshold) & 
        (X_test.iloc[:, 3] <= latency_threshold),
        1, 0
    )
    heuristic_acc = np.mean(heuristic_pred == y_test) * 100
    heuristic_scores.append(heuristic_acc)
    
    print("✓")

print("\n" + "="*80)
print("ALL 4 ALGORITHMS: 10-Fold Accuracy Values")
print("="*80 + "\n")

# Display all algorithms
print("BFL-LB (Blockchain Federated Learning-Based Load Balancer):")
print(f"  10 Fold Accuracies: {[f'{s:.1f}' for s in bfl_scores]}")
print(f"  Mean Accuracy: {np.mean(bfl_scores):.1f}")
print(f"  Std Dev: {np.std(bfl_scores):.1f}\n")

print("Centralized Scheduler:")
print(f"  10 Fold Accuracies: {[f'{s:.1f}' for s in centralized_scores]}")
print(f"  Mean Accuracy: {np.mean(centralized_scores):.1f}")
print(f"  Std Dev: {np.std(centralized_scores):.1f}\n")

print("Non-Secure Distributed Scheduler:")
print(f"  10 Fold Accuracies: {[f'{s:.1f}' for s in nonsecure_scores]}")
print(f"  Mean Accuracy: {np.mean(nonsecure_scores):.1f}")
print(f"  Std Dev: {np.std(nonsecure_scores):.1f}\n")

print("Heuristic Security Scheduler:")
print(f"  10 Fold Accuracies: {[f'{s:.1f}' for s in heuristic_scores]}")
print(f"  Mean Accuracy: {np.mean(heuristic_scores):.1f}")
print(f"  Std Dev: {np.std(heuristic_scores):.1f}\n")

# Primary vs Secondary comparison
print("="*80)
print("PRIMARY vs SECONDARY COMPARISON")
print("="*80 + "\n")

bfl_mean = np.mean(bfl_scores)
centralized_mean = np.mean(centralized_scores)
difference = bfl_mean - centralized_mean

print("[PRIMARY] BFL-LB (Blockchain Federated Learning-Based Load Balancer):")
print(f"  10 Fold Accuracies: {[f'{s:.1f}' for s in bfl_scores]}")
print(f"  Mean Accuracy: {bfl_mean:.1f}\n")

print("[SECONDARY] Centralized Scheduler:")
print(f"  10 Fold Accuracies: {[f'{s:.1f}' for s in centralized_scores]}")
print(f"  Mean Accuracy: {centralized_mean:.1f}\n")

if difference > 0:
    print(f"  ✓ PRIMARY outperforms SECONDARY by: {difference:.1f}\n")
else:
    print(f"  ✗ SECONDARY outperforms PRIMARY by: {abs(difference):.1f}\n")

# Best algorithm
print("="*80)
print("BEST ALGORITHM FROM DATASET")
print("="*80 + "\n")

all_means = {
    'BFL-LB': bfl_mean,
    'Centralized Scheduler': centralized_mean,
    'Non-Secure Distributed': np.mean(nonsecure_scores),
    'Heuristic Security': np.mean(heuristic_scores)
}

best_algo = max(all_means, key=all_means.get)
best_score = all_means[best_algo]

if best_algo == 'BFL-LB':
    print(f"🏆 BEST ALGORITHM: {best_algo}")
    print(f"   10 Fold Accuracies: {[f'{s:.1f}' for s in bfl_scores]}")
    print(f"   Mean Accuracy: {best_score:.1f}")
    print(f"   Std Dev (Stability): {np.std(bfl_scores):.1f}\n")
elif best_algo == 'Centralized Scheduler':
    print(f"🏆 BEST ALGORITHM: {best_algo}")
    print(f"   10 Fold Accuracies: {[f'{s:.1f}' for s in centralized_scores]}")
    print(f"   Mean Accuracy: {best_score:.1f}")
    print(f"   Std Dev (Stability): {np.std(centralized_scores):.1f}\n")
elif best_algo == 'Non-Secure Distributed':
    print(f"🏆 BEST ALGORITHM: {best_algo}")
    print(f"   10 Fold Accuracies: {[f'{s:.1f}' for s in nonsecure_scores]}")
    print(f"   Mean Accuracy: {best_score:.1f}")
    print(f"   Std Dev (Stability): {np.std(nonsecure_scores):.1f}\n")
else:
    print(f"🏆 BEST ALGORITHM: {best_algo}")
    print(f"   10 Fold Accuracies: {[f'{s:.1f}' for s in heuristic_scores]}")
    print(f"   Mean Accuracy: {best_score:.1f}")
    print(f"   Std Dev (Stability): {np.std(heuristic_scores):.1f}\n")

# Ranking all algorithms
print("="*80)
print("RANKING ALL ALGORITHMS")
print("="*80 + "\n")

sorted_algos = sorted(all_means.items(), key=lambda x: x[1], reverse=True)
for rank, (algo, score) in enumerate(sorted_algos, 1):
    if algo == 'BFL-LB':
        std = np.std(bfl_scores)
    elif algo == 'Centralized Scheduler':
        std = np.std(centralized_scores)
    elif algo == 'Non-Secure Distributed':
        std = np.std(nonsecure_scores)
    else:
        std = np.std(heuristic_scores)
    
    print(f"{rank}. {algo}")
    print(f"   Mean Accuracy: {score:.1f}")
    print(f"   Std Dev: {std:.1f}\n")

print("="*80)
print("Evaluation complete.")
print("="*80)
