import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.getenv("RESULTS_DIR", "/Data_share/hongyi/DAT/SAE/results/steering")
CSV_PATH = os.getenv("STEERING_CSV", os.path.join(RESULTS_DIR, "ad_steering_targeted_success_results.csv"))
SUMMARY_PATH = os.getenv(
    "STEERING_SUMMARY", os.path.join(RESULTS_DIR, "ad_steering_targeted_success_summary.json")
)
FIG_DPI = int(os.getenv("FIG_DPI", "300"))


def read_rows(csv_path: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) == 0:
        raise RuntimeError("CSV has no rows")
    return rows


def maybe_read_summary(summary_path: str) -> Dict:
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(x: str) -> float:
    return float(x)


def to_int(x: str) -> int:
    return int(x)


def format_feature_list(values: List[int], max_items: int = 12) -> str:
    if not values:
        return "[]"
    if len(values) <= max_items:
        return "[" + ", ".join(str(v) for v in values) + "]"
    head = ", ".join(str(v) for v in values[:max_items])
    return f"[{head}, ...]"


def plot_target_conf_scatter(rows: List[Dict], summary: Dict, out_path: str):
    x = np.array([to_float(r["baseline_target_conf"]) for r in rows], dtype=np.float32)
    y = np.array([to_float(r["steered_target_conf"]) for r in rows], dtype=np.float32)
    changed = np.array([to_int(r["changed_pred"]) for r in rows], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x[changed == 1], y[changed == 1], s=45, c="#e74c3c", alpha=0.85, label="Prediction changed")
    ax.scatter(x[changed == 0], y[changed == 0], s=45, c="#2e86de", alpha=0.85, label="Prediction unchanged")

    m = max(float(x.max()), float(y.max()), 1e-6)
    ax.plot([0, m], [0, m], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(0, m * 1.05)
    ax.set_ylim(0, m * 1.05)

    ax.set_xlabel("Baseline target confidence")
    ax.set_ylabel("Steered target confidence")
    ax.set_title("Target Confidence Shift (Per Image)")
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    zero_features = summary.get("zero_features", []) if summary else []
    boost_features = summary.get("boost_features", []) if summary else []
    boost_value = summary.get("boost_value", "N/A") if summary else "N/A"

    note = (
        f"Zeroed features: {format_feature_list(zero_features)}\n"
        f"Boosted features: {format_feature_list(boost_features)}\n"
        f"Boost value: {boost_value}"
    )
    ax.text(
        0.02,
        0.98,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#999999"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = read_rows(CSV_PATH)
    summary = maybe_read_summary(SUMMARY_PATH)

    p3 = os.path.join(RESULTS_DIR, "target_conf_scatter.png")
    plot_target_conf_scatter(rows, summary, p3)

    print("Done.")
    print(f"Saved: {p3}")


if __name__ == "__main__":
    main()
