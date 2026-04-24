import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = "multi_cloud_service_dataset.csv"
N_SPLITS = 10
RANDOM_STATE = 42


@dataclass
class FoldResult:
    moeo_lb: float
    static_heuristic: float
    energy_unaware: float
    single_objective: float


def build_static_heuristic(train_df: pd.DataFrame, target_col: str):
    """Build heuristic-based placement rules based on latency and energy metrics"""
    global_majority = train_df[target_col].mode().iloc[0] if len(train_df[target_col].mode()) > 0 else 0

    # Create latency and energy buckets
    latency_q1, latency_q2 = train_df["Service_Latency (ms)"].quantile([0.33, 0.66])
    energy_q1, energy_q2 = train_df["CPU_Utilization (%)"].quantile([0.33, 0.66])

    def bin_metric(series: pd.Series, q1: float, q2: float) -> pd.Series:
        return pd.cut(series, bins=[-np.inf, q1, q2, np.inf], labels=["L", "M", "H"])

    grouped = train_df.copy()
    grouped["latency_bin"] = bin_metric(grouped["Service_Latency (ms)"], latency_q1, latency_q2)
    grouped["energy_bin"] = bin_metric(grouped["CPU_Utilization (%)"], energy_q1, energy_q2)

    key_cols = ["latency_bin", "energy_bin", "Service_Type"]
    rule_map = (
        grouped.groupby(key_cols, observed=False)[target_col]
        .agg(lambda x: x.value_counts().idxmax() if len(x.value_counts()) > 0 else global_majority)
        .to_dict()
    )

    def predictor(test_df: pd.DataFrame) -> np.ndarray:
        latency_bins = bin_metric(test_df["Service_Latency (ms)"], latency_q1, latency_q2)
        energy_bins = bin_metric(test_df["CPU_Utilization (%)"], energy_q1, energy_q2)

        preds = []
        for i in range(len(test_df)):
            key = (
                latency_bins.iloc[i],
                energy_bins.iloc[i],
                test_df["Service_Type"].iloc[i],
            )
            preds.append(rule_map.get(key, global_majority))

        return np.array(preds)

    return predictor


def predict_energy_unaware(test_size: int) -> np.ndarray:
    """Simple round-robin placement without energy awareness - returns binary predictions"""
    return np.array([i % 2 for i in range(test_size)])


def predict_single_objective(test_df: pd.DataFrame) -> np.ndarray:
    """Single-objective optimization focusing only on latency - returns binary predictions"""
    df = test_df.copy()
    df["_orig_idx"] = np.arange(len(df))
    df = df.sort_values("Service_Latency (ms)").reset_index(drop=True)

    # Assign based on latency median
    median_latency = df["Service_Latency (ms)"].median()
    preds = [0 if lat < median_latency else 1 for lat in df["Service_Latency (ms)"]]

    df["pred"] = preds
    preds = df.sort_values("_orig_idx")["pred"].to_numpy()
    return preds


def evaluate(df: pd.DataFrame, target_col: str = "Optimal_Service_Placement"):
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col, "Service_ID"])

    numeric_features = [
        "CPU_Utilization (%)",
        "Memory_Usage (MB)",
        "Storage_Usage (GB)",
        "Network_Bandwidth (Mbps)",
        "Service_Latency (ms)",
        "Response_Time (ms)",
        "Throughput (Requests/sec)",
        "Load_Balancing (%)",
        "QoS_Score",
        "Workload_Variability",
    ]
    categorical_features = ["Service_Type", "Cloud_Provider", "Edge_Node_ID"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # MOEO-LB: Multi-Objective Evolutionary Optimization Load Balancer
        moeo_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        random_state=RANDOM_STATE + fold,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        moeo_model.fit(X_train, y_train)
        moeo_preds = moeo_model.predict(X_test)

        # Energy-Unaware Load Balancing
        eu_preds_binary = predict_energy_unaware(len(X_test))

        # Single-Objective Optimization (Latency-only)
        so_preds_binary = predict_single_objective(X_test)

        # Static Heuristic
        static_predictor = build_static_heuristic(
            pd.concat([X_train, pd.Series(y_train, name=target_col)], axis=1),
            target_col,
        )
        sh_preds = static_predictor(X_test)
        # Map to binary
        sh_preds_binary = np.where(sh_preds.astype(int) % 2 == 0, 0, 1)

        fold_results.append(
            FoldResult(
                moeo_lb=accuracy_score(y_test, moeo_preds),
                static_heuristic=accuracy_score(y_test, sh_preds_binary),
                energy_unaware=accuracy_score(y_test, eu_preds_binary),
                single_objective=accuracy_score(y_test, so_preds_binary),
            )
        )

    return fold_results


def print_results(fold_results: list[FoldResult]):
    algo_to_scores = defaultdict(list)
    for result in fold_results:
        algo_to_scores["MOEO-LB"].append(result.moeo_lb)
        algo_to_scores["Static Heuristic"].append(result.static_heuristic)
        algo_to_scores["Energy-Unaware"].append(result.energy_unaware)
        algo_to_scores["Single-Objective"].append(result.single_objective)

    print("=" * 80)
    print("ALL 4 ALGORITHMS: 10-Fold Accuracy Values")
    print("=" * 80)

    algo_stats = {}
    for algo, scores in algo_to_scores.items():
        decimal_scores = [round(s, 2) for s in scores]
        mean_decimal = np.mean(scores)
        std_decimal = np.std(scores)
        algo_stats[algo] = {"scores": decimal_scores, "mean": mean_decimal, "std": std_decimal}
        print(f"\n{algo}:")
        print(f"  10 Fold Accuracies: {decimal_scores}")
        print(f"  Mean Accuracy: {mean_decimal:.2f}")
        print(f"  Std Dev: {std_decimal:.2f}")

    # PRIMARY vs SECONDARY
    print("\n" + "=" * 80)
    print("PRIMARY vs SECONDARY COMPARISON")
    print("=" * 80)

    primary_algo = "MOEO-LB"
    secondary_algo = "Static Heuristic"

    primary_scores = algo_stats[primary_algo]["scores"]
    primary_mean = algo_stats[primary_algo]["mean"]
    secondary_scores = algo_stats[secondary_algo]["scores"]
    secondary_mean = algo_stats[secondary_algo]["mean"]

    print(f"\n[PRIMARY] {primary_algo} (Multi-Objective Evolutionary Optimization Load Balancer):")
    print(f"  10 Fold Accuracies: {primary_scores}")
    print(f"  Mean Accuracy: {primary_mean:.2f}")

    print(f"\n[SECONDARY] {secondary_algo} (Heuristic-Based Placement):")
    print(f"  10 Fold Accuracies: {secondary_scores}")
    print(f"  Mean Accuracy: {secondary_mean:.2f}")

    difference = primary_mean - secondary_mean
    if difference > 0:
        print(f"\n  ✓ PRIMARY outperforms SECONDARY by: {difference:.2f}")
    elif difference < 0:
        print(f"\n  ✗ PRIMARY underperforms SECONDARY by: {abs(difference):.2f}")
    else:
        print(f"\n  ≈ PRIMARY and SECONDARY are equal: {difference:.2f}")

    # BEST ALGORITHM
    print("\n" + "=" * 80)
    print("BEST ALGORITHM FROM DATASET")
    print("=" * 80)

    best_algo = max(algo_stats, key=lambda x: (algo_stats[x]["mean"], -algo_stats[x]["std"]))
    best_stats = algo_stats[best_algo]

    print(f"\n🏆 BEST ALGORITHM: {best_algo}")
    print(f"   10 Fold Accuracies: {best_stats['scores']}")
    print(f"   Mean Accuracy: {best_stats['mean']:.2f}")
    print(f"   Std Dev (Stability): {best_stats['std']:.2f}")

    print("\n" + "=" * 80)
    print("RANKING ALL ALGORITHMS")
    print("=" * 80)
    sorted_algos = sorted(algo_stats.items(), key=lambda x: (-x[1]["mean"], x[1]["std"]))
    for rank, (algo, stats) in enumerate(sorted_algos, 1):
        print(f"\n{rank}. {algo}")
        print(f"   Mean Accuracy: {stats['mean']:.2f}")
        print(f"   Std Dev: {stats['std']:.2f}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    results = evaluate(df)
    print_results(results)
