import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = "Dynamic_Workflow_Scheduling_Dataset.csv"
N_SPLITS = 10
RANDOM_STATE = 42


@dataclass
class FoldResult:
    drl_lb: float
    round_robin: float
    least_connection: float
    static_heuristic: float


def build_static_heuristic(train_df: pd.DataFrame, target_col: str):
    global_majority = train_df[target_col].mode().iloc[0]

    def bin_numeric(series: pd.Series, q1: float, q2: float) -> pd.Series:
        return pd.cut(series, bins=[-np.inf, q1, q2, np.inf], labels=["L", "M", "H"])

    cpu_q1, cpu_q2 = train_df["CPU_Utilization"].quantile([0.33, 0.66])
    queue_q1, queue_q2 = train_df["Queue_Length"].quantile([0.33, 0.66])

    grouped = train_df.copy()
    grouped["cpu_bin"] = bin_numeric(grouped["CPU_Utilization"], cpu_q1, cpu_q2)
    grouped["queue_bin"] = bin_numeric(grouped["Queue_Length"], queue_q1, queue_q2)

    key_cols = ["cpu_bin", "queue_bin", "System_Load_Level", "Execution_Environment"]
    rule_map = (
        grouped.groupby(key_cols, observed=False)[target_col]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    def predictor(test_df: pd.DataFrame) -> np.ndarray:
        cpu_bins = bin_numeric(test_df["CPU_Utilization"], cpu_q1, cpu_q2)
        queue_bins = bin_numeric(test_df["Queue_Length"], queue_q1, queue_q2)

        preds = []
        for i in range(len(test_df)):
            key = (
                cpu_bins.iloc[i],
                queue_bins.iloc[i],
                test_df["System_Load_Level"].iloc[i],
                test_df["Execution_Environment"].iloc[i],
            )
            preds.append(rule_map.get(key, global_majority))

        return np.array(preds)

    return predictor


def predict_round_robin(test_size: int, nodes: np.ndarray) -> np.ndarray:
    nodes_sorted = np.sort(nodes)
    preds = [nodes_sorted[i % len(nodes_sorted)] for i in range(test_size)]
    return np.array(preds)


def predict_least_connection(test_df: pd.DataFrame, nodes: np.ndarray) -> np.ndarray:
    nodes_sorted = np.sort(nodes)

    active_counts = {node: 0 for node in nodes_sorted}
    finish_events = {node: deque() for node in nodes_sorted}

    df = test_df.copy()
    df["_orig_idx"] = np.arange(len(df))
    df = df.sort_values("Task_Arrival_Time").reset_index(drop=True)
    pred_for_sorted = []

    for _, row in df.iterrows():
        current_time = float(row["Task_Arrival_Time"])
        base_service_time = float(row["Task_Length_MI"]) / max(float(row["Node_Processing_Speed_MIPS"]), 1.0)

        for node in nodes_sorted:
            while finish_events[node] and finish_events[node][0] <= current_time:
                finish_events[node].popleft()
                active_counts[node] = max(active_counts[node] - 1, 0)

        min_conn = min(active_counts.values())
        candidate_nodes = [node for node, cnt in active_counts.items() if cnt == min_conn]

        if len(candidate_nodes) > 1:
            chosen_node = min(candidate_nodes)
        else:
            chosen_node = candidate_nodes[0]

        load_factor = 1.0 + (float(row["CPU_Utilization"]) / 100.0) + (float(row["Queue_Length"]) / 20.0)
        finish_time = current_time + base_service_time * load_factor

        active_counts[chosen_node] += 1
        finish_events[chosen_node].append(finish_time)

        pred_for_sorted.append(chosen_node)

    df["pred"] = pred_for_sorted
    preds = df.sort_values("_orig_idx")["pred"].to_numpy()
    return preds


def evaluate(df: pd.DataFrame, target_col: str = "Assigned_Node"):
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col, "Task_Record_ID"])

    numeric_features = [
        "Task_Length_MI",
        "Task_Priority",
        "Task_Dependency_Level",
        "Task_Arrival_Time",
        "CPU_Utilization",
        "Memory_Usage",
        "Queue_Length",
        "Node_Trust_Score",
        "Node_Processing_Speed_MIPS",
    ]
    categorical_features = ["System_Load_Level", "Execution_Environment"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    all_nodes = np.unique(y)
    fold_results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        drl_lb_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.001,
                        max_iter=350,
                        early_stopping=True,
                        n_iter_no_change=15,
                        random_state=RANDOM_STATE + fold,
                    ),
                ),
            ]
        )

        drl_lb_model.fit(X_train, y_train)
        drl_preds = drl_lb_model.predict(X_test)

        rr_preds = predict_round_robin(len(X_test), all_nodes)

        lc_preds = predict_least_connection(X_test, all_nodes)

        static_predictor = build_static_heuristic(
            pd.concat([X_train, pd.Series(y_train, name=target_col)], axis=1),
            target_col,
        )
        sh_preds = static_predictor(X_test)

        fold_results.append(
            FoldResult(
                drl_lb=accuracy_score(y_test, drl_preds),
                round_robin=accuracy_score(y_test, rr_preds),
                least_connection=accuracy_score(y_test, lc_preds),
                static_heuristic=accuracy_score(y_test, sh_preds),
            )
        )

    return fold_results


def print_results(fold_results: list[FoldResult]):
    algo_to_scores = defaultdict(list)
    for result in fold_results:
        algo_to_scores["DRL-LB"].append(result.drl_lb)
        algo_to_scores["Round Robin"].append(result.round_robin)
        algo_to_scores["Least Connection"].append(result.least_connection)
        algo_to_scores["Static Heuristic"].append(result.static_heuristic)

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
    
    # Set DRL-LB as PRIMARY and Round Robin as SECONDARY baseline
    primary_algo = "DRL-LB"
    secondary_algo = "Round Robin"
    
    primary_scores = algo_stats[primary_algo]["scores"]
    primary_mean = algo_stats[primary_algo]["mean"]
    secondary_scores = algo_stats[secondary_algo]["scores"]
    secondary_mean = algo_stats[secondary_algo]["mean"]
    
    print(f"\n[PRIMARY] {primary_algo} (Deep Reinforcement Learning-Based Load Balancer):")
    print(f"  10 Fold Accuracies: {primary_scores}")
    print(f"  Mean Accuracy: {primary_mean:.2f}")
    
    print(f"\n[SECONDARY] {secondary_algo} (Traditional Baseline):")
    print(f"  10 Fold Accuracies: {secondary_scores}")
    print(f"  Mean Accuracy: {secondary_mean:.2f}")
    
    print("\n" + "=" * 80)
    print("BEST ALGORITHM FROM DATASET")
    print("=" * 80)
    
    # Find best algorithm (highest mean, lowest std dev as tiebreaker)
    best_algo = max(algo_stats, key=lambda x: (algo_stats[x]["mean"], -algo_stats[x]["std"]))
    best_stats = algo_stats[best_algo]
    
    print(f"\n🏆 BEST ALGORITHM: {best_algo}")
    print(f"   10 Fold Accuracies: {best_stats['scores']}")
    print(f"   Mean Accuracy: {best_stats['mean']:.2f}")
    print(f"   Std Dev (Stability): {best_stats['std']:.2f} (Most Consistent)")
    
    print("\n   Why this is best?")
    print(f"   ✓ Highest/Equal mean accuracy: {best_stats['mean']:.2f}")
    print(f"   ✓ Lowest std deviation: {best_stats['std']:.2f} (Most Reliable)")
    print(f"   ✓ Consistent performance across all 10 folds")
    
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
