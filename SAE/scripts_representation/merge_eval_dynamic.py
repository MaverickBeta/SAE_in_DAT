#!/usr/bin/env python3
"""
Merge eval_20cls.json + dynamic_nearest_results.json into a single CSV + summary.

Usage:
    cd /Data_share/hongyi/DAT/SAE/scripts_representation
    python merge_eval_dynamic.py
"""

import json
import csv
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
EVAL_PATH = Path("/Data_share/hongyi/DAT/SAE/results_representation/eval_20cls/eval_results.json")
DYN_PATH = Path("/Data_share/hongyi/DAT/SAE/results_representation/steering_20cls_dynamic_nearest/dynamic_nearest_results.json")
OUT_DIR = Path("/Data_share/hongyi/DAT/SAE/results_representation")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def pct(v):
    """Format 0-1 float as percentage string, or N/A."""
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%"


def pp(v):
    """Format percentage-point value, or N/A."""
    if v is None:
        return "N/A"
    return f"{v:+.1f}"


def qval(q, k):
    return q.get(k) if q else None


def build_per_class_rows(eval_data, dyn_data):
    eval_by_name = {r["name"]: r for r in eval_data["per_class"]}
    dyn_by_name = {r["name"]: r for r in dyn_data["per_class"]}
    names = [r["name"] for r in eval_data["per_class"]]

    rows = []
    for name in names:
        e = eval_by_name[name]
        d = dyn_by_name.get(name, {})

        row = {
            # Basic
            "name": name,
            "wnid": e["wnid"],
            "class_idx": e["class_idx"],
            "n_all_adv": e["n_all_adv"],
            "n_succ_adv": e["n_succ_adv"],
            "n_clean": d.get("n_clean"),
            # Eval baseline
            "all_control_acc": e.get("all_control_acc"),
            "all_sae_only_acc": e.get("all_sae_only_acc"),
            "succ_control_acc": e.get("succ_control_acc"),
            "succ_sae_only_acc": e.get("succ_sae_only_acc"),
            "all_sae_vs_control_pp": e.get("all_sae_vs_control"),
            "succ_sae_vs_control_pp": e.get("succ_sae_vs_control"),
            # Dynamic accuracy
            "succ_v1_acc": d.get("succ_v1_acc"),
            "succ_v2_acc": d.get("succ_v2_acc"),
            "all_v1_acc": d.get("all_v1_acc"),
            "all_v2_acc": d.get("all_v2_acc"),
            "clean_control_acc": d.get("clean_control_acc"),
            "clean_v1_acc": d.get("clean_v1_acc"),
            "clean_v2_acc": d.get("clean_v2_acc"),
            # Dynamic gain vs Control (pp)
            "succ_v1_vs_control_pp": d.get("succ_v1_vs_control"),
            "succ_v2_vs_control_pp": d.get("succ_v2_vs_control"),
            "all_v1_vs_control_pp": d.get("all_v1_vs_control"),
            "all_v2_vs_control_pp": d.get("all_v2_vs_control"),
            "clean_v1_vs_control_pp": d.get("clean_v1_vs_control"),
            "clean_v2_vs_control_pp": d.get("clean_v2_vs_control"),
            # Dynamic gain vs SAE-only (pp)
            "succ_v1_vs_sae_pp": (
                (d.get("succ_v1_acc") - e.get("succ_sae_only_acc")) * 100
                if d.get("succ_v1_acc") is not None and e.get("succ_sae_only_acc") is not None
                else None
            ),
            "succ_v2_vs_sae_pp": (
                (d.get("succ_v2_acc") - e.get("succ_sae_only_acc")) * 100
                if d.get("succ_v2_acc") is not None and e.get("succ_sae_only_acc") is not None
                else None
            ),
            "all_v1_vs_sae_pp": (
                (d.get("all_v1_acc") - e.get("all_sae_only_acc")) * 100
                if d.get("all_v1_acc") is not None and e.get("all_sae_only_acc") is not None
                else None
            ),
            "all_v2_vs_sae_pp": (
                (d.get("all_v2_acc") - e.get("all_sae_only_acc")) * 100
                if d.get("all_v2_acc") is not None and e.get("all_sae_only_acc") is not None
                else None
            ),
            # Quadrant analysis
            "succ_v1_recovery": qval(d.get("succ_v1_quadrants"), "recovery"),
            "succ_v1_regression": qval(d.get("succ_v1_quadrants"), "regression"),
            "succ_v1_net_gain": qval(d.get("succ_v1_quadrants"), "net_gain"),
            "succ_v2_recovery": qval(d.get("succ_v2_quadrants"), "recovery"),
            "succ_v2_regression": qval(d.get("succ_v2_quadrants"), "regression"),
            "succ_v2_net_gain": qval(d.get("succ_v2_quadrants"), "net_gain"),
            "all_v1_recovery": qval(d.get("all_v1_quadrants"), "recovery"),
            "all_v1_regression": qval(d.get("all_v1_quadrants"), "regression"),
            "all_v1_net_gain": qval(d.get("all_v1_quadrants"), "net_gain"),
            "all_v2_recovery": qval(d.get("all_v2_quadrants"), "recovery"),
            "all_v2_regression": qval(d.get("all_v2_quadrants"), "regression"),
            "all_v2_net_gain": qval(d.get("all_v2_quadrants"), "net_gain"),
        }
        rows.append(row)
    return rows


def write_per_class_csv(rows, out_path):
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-class CSV saved: {out_path}")


def write_overall_csv(overall_rows, out_path):
    fieldnames = list(overall_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overall_rows)
    print(f"Overall CSV saved: {out_path}")


def print_overall_table(eval_ov, dyn_ov):
    """Print a clean overall average table."""
    print("\n" + "=" * 95)
    print("OVERALL AVERAGE SUMMARY")
    print("=" * 95)

    # Row format helper
    def row(label, e_val, d_v1, d_v2):
        print(f"  {label:<35} {e_val:>12} {d_v1:>12} {d_v2:>12}")

    print(f"  {'Metric':<35} {'Eval (SAE)':>12} {'Dynamic V1':>12} {'Dynamic V2':>12}")
    print("-" * 95)

    row("Succ Adv - Control",   pct(eval_ov.get("avg_succ_control_acc")),   pct(dyn_ov.get("avg_succ_control_acc")),   pct(dyn_ov.get("avg_succ_control_acc")))
    row("Succ Adv - SAE-only",  pct(eval_ov.get("avg_succ_sae_only_acc")),  "—",                                       "—")
    row("Succ Adv - Dynamic",   "—",                                        pct(dyn_ov.get("avg_succ_v1_acc")),        pct(dyn_ov.get("avg_succ_v2_acc")))
    row("Succ Adv - Δ vs Ctrl", pp(eval_ov.get("avg_succ_sae_vs_control")), pp(dyn_ov.get("avg_succ_v1_vs_control")),  pp(dyn_ov.get("avg_succ_v2_vs_control")))
    row("Succ Adv - Δ vs SAE",  "—",                                        pp(dyn_ov.get("avg_succ_v1_vs_control") - eval_ov.get("avg_succ_sae_vs_control", 0)), pp(dyn_ov.get("avg_succ_v2_vs_control") - eval_ov.get("avg_succ_sae_vs_control", 0)))
    print("-" * 95)
    row("All Adv - Control",    pct(eval_ov.get("avg_all_control_acc")),    pct(dyn_ov.get("avg_all_control_acc")),    pct(dyn_ov.get("avg_all_control_acc")))
    row("All Adv - SAE-only",   pct(eval_ov.get("avg_all_sae_only_acc")),   "—",                                       "—")
    row("All Adv - Dynamic",    "—",                                        pct(dyn_ov.get("avg_all_v1_acc")),         pct(dyn_ov.get("avg_all_v2_acc")))
    row("All Adv - Δ vs Ctrl",  pp(eval_ov.get("avg_all_sae_vs_control")),  pp(dyn_ov.get("avg_all_v1_vs_control")),   pp(dyn_ov.get("avg_all_v2_vs_control")))
    print("-" * 95)
    row("Clean - Control",      "—",                                        pct(dyn_ov.get("avg_clean_control_acc")),  pct(dyn_ov.get("avg_clean_control_acc")))
    row("Clean - Dynamic",      "—",                                        pct(dyn_ov.get("avg_clean_v1_acc")),       pct(dyn_ov.get("avg_clean_v2_acc")))
    row("Clean - Δ vs Ctrl",    "—",                                        pp(dyn_ov.get("avg_clean_v1_vs_control")), pp(dyn_ov.get("avg_clean_v2_vs_control")))
    print("=" * 95)


def build_overall_rows(eval_ov, dyn_ov):
    """Build rows for overall-average CSV."""
    return [
        {
            "metric": "succ_control_acc",
            "eval": eval_ov.get("avg_succ_control_acc"),
            "dynamic_v1": dyn_ov.get("avg_succ_control_acc"),
            "dynamic_v2": dyn_ov.get("avg_succ_control_acc"),
        },
        {
            "metric": "succ_sae_only_acc",
            "eval": eval_ov.get("avg_succ_sae_only_acc"),
            "dynamic_v1": None,
            "dynamic_v2": None,
        },
        {
            "metric": "succ_dynamic_acc",
            "eval": None,
            "dynamic_v1": dyn_ov.get("avg_succ_v1_acc"),
            "dynamic_v2": dyn_ov.get("avg_succ_v2_acc"),
        },
        {
            "metric": "succ_vs_control_pp",
            "eval": eval_ov.get("avg_succ_sae_vs_control"),
            "dynamic_v1": dyn_ov.get("avg_succ_v1_vs_control"),
            "dynamic_v2": dyn_ov.get("avg_succ_v2_vs_control"),
        },
        {
            "metric": "all_control_acc",
            "eval": eval_ov.get("avg_all_control_acc"),
            "dynamic_v1": dyn_ov.get("avg_all_control_acc"),
            "dynamic_v2": dyn_ov.get("avg_all_control_acc"),
        },
        {
            "metric": "all_sae_only_acc",
            "eval": eval_ov.get("avg_all_sae_only_acc"),
            "dynamic_v1": None,
            "dynamic_v2": None,
        },
        {
            "metric": "all_dynamic_acc",
            "eval": None,
            "dynamic_v1": dyn_ov.get("avg_all_v1_acc"),
            "dynamic_v2": dyn_ov.get("avg_all_v2_acc"),
        },
        {
            "metric": "all_vs_control_pp",
            "eval": eval_ov.get("avg_all_sae_vs_control"),
            "dynamic_v1": dyn_ov.get("avg_all_v1_vs_control"),
            "dynamic_v2": dyn_ov.get("avg_all_v2_vs_control"),
        },
        {
            "metric": "clean_control_acc",
            "eval": None,
            "dynamic_v1": dyn_ov.get("avg_clean_control_acc"),
            "dynamic_v2": dyn_ov.get("avg_clean_control_acc"),
        },
        {
            "metric": "clean_dynamic_acc",
            "eval": None,
            "dynamic_v1": dyn_ov.get("avg_clean_v1_acc"),
            "dynamic_v2": dyn_ov.get("avg_clean_v2_acc"),
        },
        {
            "metric": "clean_vs_control_pp",
            "eval": None,
            "dynamic_v1": dyn_ov.get("avg_clean_v1_vs_control"),
            "dynamic_v2": dyn_ov.get("avg_clean_v2_vs_control"),
        },
    ]


def print_preview(rows):
    """Print a compact preview of key columns."""
    cols = [
        "name",
        "n_succ_adv",
        "n_all_adv",
        "n_clean",
        "succ_control_acc",
        "succ_v1_acc",
        "succ_v2_acc",
        "all_control_acc",
        "all_v1_acc",
        "all_v2_acc",
        "clean_control_acc",
        "clean_v1_acc",
        "clean_v2_acc",
    ]
    hdr = "{:>15} {:>6} {:>6} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11} {:>11}"
    print("\n" + "=" * 145)
    print("PER-CLASS PREVIEW (key columns)")
    print("=" * 145)
    print(hdr.format(*[c.replace("_", " ") for c in cols]))
    print("-" * 145)
    for r in rows:
        print(
            hdr.format(
                r["name"],
                r["n_succ_adv"],
                r["n_all_adv"],
                r["n_clean"] if r["n_clean"] is not None else "N/A",
                pct(r["succ_control_acc"]),
                pct(r["succ_v1_acc"]),
                pct(r["succ_v2_acc"]),
                pct(r["all_control_acc"]),
                pct(r["all_v1_acc"]),
                pct(r["all_v2_acc"]),
                pct(r["clean_control_acc"]),
                pct(r["clean_v1_acc"]),
                pct(r["clean_v2_acc"]),
            )
        )
    print("=" * 145)


def main():
    eval_data = load_json(EVAL_PATH)
    dyn_data = load_json(DYN_PATH)

    rows = build_per_class_rows(eval_data, dyn_data)

    # Per-class CSV
    per_class_csv = OUT_DIR / "eval_dynamic_merged.csv"
    write_per_class_csv(rows, per_class_csv)

    # Overall CSV
    overall_rows = build_overall_rows(eval_data["overall_average"], dyn_data["overall_average"])
    overall_csv = OUT_DIR / "eval_dynamic_overall.csv"
    write_overall_csv(overall_rows, overall_csv)

    # Print tables
    print_preview(rows)
    print_overall_table(eval_data["overall_average"], dyn_data["overall_average"])

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
