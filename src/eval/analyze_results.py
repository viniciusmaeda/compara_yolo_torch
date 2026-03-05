import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve project paths from this file location: src/eval/analyze_results.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

# Metric names used in tables/plots and their column names in Ultralytics results.csv.
METRIC_COLUMNS: Dict[str, str] = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}

CURVE_COLUMNS: Dict[str, str] = {
    "train_box_loss": "train/box_loss",
    "val_box_loss": "val/box_loss",
    "map50_95": "metrics/mAP50-95(B)",
}

RUN_DIR_PATTERN = re.compile(r"^fold_(\d+)_(.+)$")
MODEL_ORDER = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]


def model_order_key(model_name: str) -> int:
    """Return sort index following predefined model-size order."""
    try:
        return MODEL_ORDER.index(model_name)
    except ValueError:
        return len(MODEL_ORDER)


def discover_runs(results_dir: str) -> List[Tuple[int, str, str]]:
    """Return a list of (fold, model, run_dir) discovered from results directory names."""
    runs: List[Tuple[int, str, str]] = []

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for entry in sorted(os.listdir(results_dir)):
        entry_path = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        match = RUN_DIR_PATTERN.match(entry)
        if not match:
            continue

        fold = int(match.group(1))
        model = match.group(2)
        csv_path = os.path.join(entry_path, "results.csv")

        if os.path.isfile(csv_path):
            runs.append((fold, model, entry_path))

    return runs


def load_run_data(runs: List[Tuple[int, str, str]]) -> Tuple[pd.DataFrame, Dict[str, List[pd.DataFrame]]]:
    """Load per-fold summary and full epoch curves from all runs."""
    per_fold_rows = []
    curves_by_model: Dict[str, List[pd.DataFrame]] = {}

    for fold, model, run_dir in runs:
        csv_path = os.path.join(run_dir, "results.csv")
        df = pd.read_csv(csv_path)

        required = [*METRIC_COLUMNS.values(), *CURVE_COLUMNS.values(), "epoch"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            print(f"Skipping {run_dir}: missing columns {missing}")
            continue

        best_idx = df[METRIC_COLUMNS["map50_95"]].idxmax()
        best_row = df.loc[best_idx]

        row = {
            "fold": fold,
            "model": model,
            "run_dir": os.path.basename(run_dir),
            "epochs_ran": int(df["epoch"].max()),
            "best_epoch": int(best_row["epoch"]),
        }

        for metric_name, column_name in METRIC_COLUMNS.items():
            row[metric_name] = float(best_row[column_name])

        per_fold_rows.append(row)

        curve_df = df[["epoch", *CURVE_COLUMNS.values()]].copy()
        curve_df["epoch"] = curve_df["epoch"].astype(int)
        curves_by_model.setdefault(model, []).append(curve_df)

    per_fold_df = pd.DataFrame(per_fold_rows)
    return per_fold_df, curves_by_model


def compute_summary_tables(per_fold_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build summary, ranking, and stability tables aggregated by model."""
    if per_fold_df.empty:
        raise RuntimeError("No valid runs found to summarize.")

    agg_spec = {}
    for metric in METRIC_COLUMNS:
        agg_spec[metric] = ["mean", "std", "min", "max"]

    summary_df = per_fold_df.groupby("model", as_index=False).agg(agg_spec)
    summary_df.columns = ["_".join(col).strip("_") for col in summary_df.columns.to_flat_index()]
    summary_df = summary_df.rename(columns={"model_": "model"})

    fold_counts = per_fold_df.groupby("model", as_index=False).size().rename(columns={"size": "n_folds"})
    summary_df = summary_df.merge(fold_counts, on="model", how="left")

    for metric in METRIC_COLUMNS:
        std_col = f"{metric}_std"
        mean_col = f"{metric}_mean"
        ci_col = f"{metric}_ci95"
        summary_df[std_col] = summary_df[std_col].fillna(0.0)
        summary_df[ci_col] = 1.96 * summary_df[std_col] / np.sqrt(summary_df["n_folds"].clip(lower=1))

    summary_df["_model_order"] = summary_df["model"].map(model_order_key)
    summary_df = summary_df.sort_values(["_model_order", "model"]).drop(columns=["_model_order"]).reset_index(drop=True)

    ranking_df = summary_df[[
        "model",
        "n_folds",
        "map50_95_mean",
        "map50_95_std",
        "map50_mean",
        "recall_mean",
        "precision_mean",
    ]].copy()
    ranking_df["rank"] = ranking_df["map50_95_mean"].rank(method="min", ascending=False).astype(int)
    ranking_df = ranking_df[[
        "rank",
        "model",
        "n_folds",
        "map50_95_mean",
        "map50_95_std",
        "map50_mean",
        "recall_mean",
        "precision_mean",
    ]]
    ranking_df["_model_order"] = ranking_df["model"].map(model_order_key)
    ranking_df = ranking_df.sort_values(["_model_order", "model"]).drop(columns=["_model_order"]).reset_index(drop=True)

    stability_rows = []
    for _, row in summary_df.iterrows():
        stability_row = {"model": row["model"], "n_folds": int(row["n_folds"])}
        for metric in METRIC_COLUMNS:
            mean_val = float(row[f"{metric}_mean"])
            std_val = float(row[f"{metric}_std"])
            stability_row[f"{metric}_cv"] = (std_val / mean_val) if mean_val != 0 else np.nan
        stability_rows.append(stability_row)

    stability_df = pd.DataFrame(stability_rows)
    stability_df["_model_order"] = stability_df["model"].map(model_order_key)
    stability_df = stability_df.sort_values(["_model_order", "model"]).drop(columns=["_model_order"]).reset_index(drop=True)
    return summary_df, ranking_df, stability_df


def save_tables(per_fold_df: pd.DataFrame, summary_df: pd.DataFrame, ranking_df: pd.DataFrame, stability_df: pd.DataFrame) -> None:
    """Persist generated analysis tables to CSV files."""
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    per_fold_sorted = per_fold_df.copy()
    per_fold_sorted["_model_order"] = per_fold_sorted["model"].map(model_order_key)
    per_fold_sorted = per_fold_sorted.sort_values(["_model_order", "model", "fold"]).drop(columns=["_model_order"])
    per_fold_sorted.to_csv(os.path.join(ANALYSIS_DIR, "per_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(ANALYSIS_DIR, "model_summary.csv"), index=False)
    ranking_df.to_csv(os.path.join(ANALYSIS_DIR, "model_ranking.csv"), index=False)
    stability_df.to_csv(os.path.join(ANALYSIS_DIR, "stability_table.csv"), index=False)


def plot_boxplots(per_fold_df: pd.DataFrame) -> None:
    """Create per-metric boxplots across folds."""
    models = sorted(per_fold_df["model"].unique(), key=model_order_key)

    for metric in METRIC_COLUMNS:
        data = [per_fold_df.loc[per_fold_df["model"] == model, metric].values for model in models]

        plt.figure(figsize=(8, 5))
        plt.boxplot(data, tick_labels=models)
        plt.title(f"{metric} by model across folds")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"{metric}_boxplot.png"), dpi=150)
        plt.close()


def plot_mean_std_bars(summary_df: pd.DataFrame) -> None:
    """Create bar charts with mean +/- std per metric."""
    ordered = summary_df.copy()
    ordered["_model_order"] = ordered["model"].map(model_order_key)
    ordered = ordered.sort_values(["_model_order", "model"]).drop(columns=["_model_order"])
    x = np.arange(len(ordered))

    for metric in METRIC_COLUMNS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        plt.figure(figsize=(9, 5))
        plt.bar(x, ordered[mean_col].values, yerr=ordered[std_col].values, capsize=4)
        plt.xticks(x, ordered["model"].values)
        plt.title(f"{metric}: mean +/- std across folds")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"{metric}_mean_std_bar.png"), dpi=150)
        plt.close()


def plot_fold_model_heatmap(per_fold_df: pd.DataFrame) -> None:
    """Create fold x model heatmap for mAP50-95."""
    pivot = per_fold_df.pivot_table(index="fold", columns="model", values="map50_95", aggfunc="mean")
    pivot = pivot.sort_index()
    ordered_columns = sorted(pivot.columns.tolist(), key=model_order_key)
    pivot = pivot[ordered_columns]

    plt.figure(figsize=(10, 5))
    matrix = pivot.values
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="mAP50-95")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel("Model")
    plt.ylabel("Fold")
    plt.title("Fold x model heatmap (mAP50-95)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "map50_95_heatmap.png"), dpi=150)
    plt.close()


def plot_precision_recall_scatter(per_fold_df: pd.DataFrame) -> None:
    """Create precision vs recall scatter with one color per model."""
    plt.figure(figsize=(7, 6))

    for model in sorted(per_fold_df["model"].unique(), key=model_order_key):
        model_df = per_fold_df[per_fold_df["model"] == model]
        plt.scatter(model_df["recall"], model_df["precision"], label=model, s=45, alpha=0.8)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall by fold")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "precision_recall_scatter.png"), dpi=150)
    plt.close()


def plot_ranking(summary_df: pd.DataFrame) -> None:
    """Create model ranking chart using mAP50-95 mean."""
    ranked = summary_df.copy()
    ranked["_model_order"] = ranked["model"].map(model_order_key)
    ranked = ranked.sort_values(["_model_order", "model"], ascending=[True, True]).drop(columns=["_model_order"])

    plt.figure(figsize=(8, 5))
    plt.barh(ranked["model"], ranked["map50_95_mean"])
    plt.xlabel("mAP50-95 (mean)")
    plt.ylabel("Model")
    plt.title("Model ranking by mean mAP50-95")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "model_ranking_barh.png"), dpi=150)
    plt.close()


def plot_training_curves(curves_by_model: Dict[str, List[pd.DataFrame]]) -> None:
    """Create mean +/- std epoch curves per model and metric."""
    for metric_name, column_name in CURVE_COLUMNS.items():
        plt.figure(figsize=(10, 6))

        for model in sorted(curves_by_model, key=model_order_key):
            model_curves = curves_by_model[model]
            combined = pd.concat(model_curves, ignore_index=True)
            grouped = combined.groupby("epoch")[column_name].agg(["mean", "std"]).reset_index()
            grouped["std"] = grouped["std"].fillna(0.0)

            x = grouped["epoch"].to_numpy(dtype=float)
            y = grouped["mean"].to_numpy(dtype=float)
            std = grouped["std"].to_numpy(dtype=float)

            plt.plot(x, y, label=model)
            plt.fill_between(x, y - std, y + std, alpha=0.18)

        plt.title(f"{metric_name} mean +/- std across folds")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"{metric_name}_curve_mean_std.png"), dpi=150)
        plt.close()


def main() -> None:
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("\nReading k-fold training results...\n")
    runs = discover_runs(RESULTS_DIR)
    if not runs:
        raise RuntimeError(f"No fold runs found under: {RESULTS_DIR}")

    per_fold_df, curves_by_model = load_run_data(runs)
    if per_fold_df.empty:
        raise RuntimeError("No valid CSV data found after reading runs.")

    summary_df, ranking_df, stability_df = compute_summary_tables(per_fold_df)
    save_tables(per_fold_df, summary_df, ranking_df, stability_df)

    plot_boxplots(per_fold_df)
    plot_mean_std_bars(summary_df)
    plot_fold_model_heatmap(per_fold_df)
    plot_precision_recall_scatter(per_fold_df)
    plot_ranking(summary_df)
    plot_training_curves(curves_by_model)

    print("Analysis complete.")
    print(f"Tables and charts saved to: {ANALYSIS_DIR}")
    print("Generated tables:")
    print("- per_fold_metrics.csv")
    print("- model_summary.csv")
    print("- model_ranking.csv")
    print("- stability_table.csv")


if __name__ == "__main__":
    main()
