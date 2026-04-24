import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DATA_FILE = "Load Balancing Improved.csv"
N_SPLITS = 10
RANDOM_STATE = 42


@dataclass
class FoldResult:
    gnn_lb: float
    topology_agnostic: float
    centralized_scheduler: float
    traditional_distributed: float


def evaluate(df: pd.DataFrame, target_col: str = "target"):
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col, "timestamp"])

    # Extract numeric columns only
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"  Fold {fold}: GNN-LB...", end=" ", flush=True)

        # GNN-LB: Graph Neural Network-Based Load Balancer (MLP proxy)
        gnn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=150,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=RANDOM_STATE + fold,
        )
        gnn_model.fit(X_train_scaled, y_train)
        gnn_preds = gnn_model.predict(X_test_scaled)

        print("Topo-Agnostic...", end=" ", flush=True)

        # Topology-Agnostic: Simple median-based heuristic
        cpu_threshold = X_train.iloc[:, 0].median()
        mem_threshold = X_train.iloc[:, 1].median()
        ta_preds = np.where((X_test.iloc[:, 0] > cpu_threshold) & (X_test.iloc[:, 1] > mem_threshold), 1, 0)

        print("Centralized...", end=" ", flush=True)

        # Centralized Scheduler: Logistic Regression
        cs_model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE + fold, n_jobs=-1)
        cs_model.fit(X_train_scaled, y_train)
        cs_preds = cs_model.predict(X_test_scaled)

        print("Distributed...", end=" ", flush=True)

        # Traditional Distributed: Random Forest (simpler ensemble)
        tda_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )
        tda_model.fit(X_train_scaled, y_train)
        tda_preds = tda_model.predict(X_test_scaled)

        print("✓")

        fold_results.append(
            FoldResult(
                gnn_lb=accuracy_score(y_test, gnn_preds),
                topology_agnostic=accuracy_score(y_test, ta_preds),
                centralized_scheduler=accuracy_score(y_test, cs_preds),
                traditional_distributed=accuracy_score(y_test, tda_preds),
            )
        )

    return fold_results


def print_results(fold_results: list[FoldResult]):
    algo_to_scores = defaultdict(list)
    for result in fold_results:
        algo_to_scores["GNN-LB"].append(result.gnn_lb)
        algo_to_scores["Topology-Agnostic"].append(result.topology_agnostic)
        algo_to_scores["Centralized Scheduler"].append(result.centralized_scheduler)
        algo_to_scores["Traditional Distributed"].append(result.traditional_distributed)

    print("\n" + "=" * 80)
    print("ALL 4 ALGORITHMS: 10-Fold Accuracy Values")
    print("=" * 80)

    algo_stats = {}
    for algo, scores in algo_to_scores.items():
        percent_scores = [round(s * 100, 1) for s in scores]
        mean_percent = np.mean(scores) * 100
        std_percent = np.std(scores) * 100
        algo_stats[algo] = {"scores": percent_scores, "mean": mean_percent, "std": std_percent}
        print(f"\n{algo}:")
        print(f"  10 Fold Accuracies: {percent_scores}")
        print(f"  Mean Accuracy: {mean_percent:.1f}")
        print(f"  Std Dev: {std_percent:.1f}")

    # PRIMARY vs SECONDARY
    print("\n" + "=" * 80)
    print("PRIMARY vs SECONDARY COMPARISON")
    print("=" * 80)

    primary_algo = "GNN-LB"
    secondary_algo = "Topology-Agnostic"

    primary_scores = algo_stats[primary_algo]["scores"]
    primary_mean = algo_stats[primary_algo]["mean"]
    secondary_scores = algo_stats[secondary_algo]["scores"]
    secondary_mean = algo_stats[secondary_algo]["mean"]

    print(f"\n[PRIMARY] {primary_algo} (Graph Neural Network-Based Load Balancer):")
    print(f"  10 Fold Accuracies: {primary_scores}")
    print(f"  Mean Accuracy: {primary_mean:.1f}")

    print(f"\n[SECONDARY] {secondary_algo} (Topology-Agnostic Baseline):")
    print(f"  10 Fold Accuracies: {secondary_scores}")
    print(f"  Mean Accuracy: {secondary_mean:.1f}")

    difference = primary_mean - secondary_mean
    if difference > 0:
        print(f"\n  ✓ PRIMARY outperforms SECONDARY by: {difference:.1f}")
    elif difference < 0:
        print(f"\n  ✗ PRIMARY underperforms SECONDARY by: {abs(difference):.1f}")
    else:
        print(f"\n  ≈ PRIMARY and SECONDARY are equal: {difference:.1f}")

    # BEST ALGORITHM
    print("\n" + "=" * 80)
    print("BEST ALGORITHM FROM DATASET")
    print("=" * 80)

    best_algo = max(algo_stats, key=lambda x: (algo_stats[x]["mean"], -algo_stats[x]["std"]))
    best_stats = algo_stats[best_algo]

    print(f"\n🏆 BEST ALGORITHM: {best_algo}")
    print(f"   10 Fold Accuracies: {best_stats['scores']}")
    print(f"   Mean Accuracy: {best_stats['mean']:.1f}")
    print(f"   Std Dev (Stability): {best_stats['std']:.1f}")

    print("\n" + "=" * 80)
    print("RANKING ALL ALGORITHMS")
    print("=" * 80)
    sorted_algos = sorted(algo_stats.items(), key=lambda x: (-x[1]["mean"], x[1]["std"]))
    for rank, (algo, stats) in enumerate(sorted_algos, 1):
        print(f"\n{rank}. {algo}")
        print(f"   Mean Accuracy: {stats['mean']:.1f}")
        print(f"   Std Dev: {stats['std']:.1f}")


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset loaded: {len(df)} rows")
    print("Starting 10-fold evaluation...\n")
    results = evaluate(df)
    print_results(results)
    print("Evaluation complete. Printing results...")
    print_results(results)
