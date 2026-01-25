import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    _feasible_and_objective_factory,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset


# -----------------------------
# Global configs
# -----------------------------
CURRENT_DIR = "results/figures"
MOCK_SEED = "0_cherry"

SOL_STYLES = {
    "our_solution":  {"c": "#d62728", "marker": "o", "ls": "-",  "lw": 2.0, "zorder": 10, "label": "CHAT"},
    "vd_tuner":      {"c": "#9467bd", "marker": "s", "ls": "--", "lw": 1.8, "zorder": 5,  "label": "VDTuner"},
    "eci":           {"c": "#1f77b4", "marker": "X", "ls": "-.", "lw": 1.8, "zorder": 6,  "label": "ECI (GP)"},  # NEW
    "optuna":        {"c": "#8c564b", "marker": "^", "ls": "-.", "lw": 1.8, "zorder": 4,  "label": "Optuna"},
    "nsga":          {"c": "#e377c2", "marker": "D", "ls": ":",  "lw": 1.8, "zorder": 3,  "label": "NSGA-II"},
    "random_search": {"c": "#7f7f7f", "marker": "v", "ls": "--", "lw": 1.8, "zorder": 2,  "label": "Random"},
    "grid_search":   {"c": "#bcbd22", "marker": "P", "ls": ":",  "lw": 1.8, "zorder": 1,  "label": "Grid"},
}
SOL_ORDER = ["our_solution", "vd_tuner", "eci", "optuna", "nsga", "random_search", "grid_search"]

ALPHAS_BASE = [0.75, 0.9, 0.925, 0.95]
ALPHAS_CALC = [0.001, 0.25, 0.5] + ALPHAS_BASE
TAIL_START = 0.95

BOUNDARY_REAL = 0.75
BOUNDARY_VISUAL = 0.30


def squash_y(y: float) -> float:
    y = float(y)
    if y <= BOUNDARY_REAL:
        return y * (BOUNDARY_VISUAL / BOUNDARY_REAL)
    slope = (1.0 - BOUNDARY_VISUAL) / (1.0 - BOUNDARY_REAL)
    return BOUNDARY_VISUAL + (y - BOUNDARY_REAL) * slope


def draw_wavy_y_break(ax, y_data):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    xs = np.linspace(-0.005 - 0.03, -0.005 + 0.03, 200)
    phase = 2.0 * np.pi * (xs - xs.min()) / (xs.max() - xs.min())
    ax.plot(xs, y_data + 0.014 * np.sin(phase), transform=trans, color="black", lw=1, clip_on=False)
    ax.plot(xs, y_data - 0.014 * np.sin(phase), transform=trans, color="black", lw=1, clip_on=False)


def short_ds(name: str) -> str:
    return name.split("-")[0]


def get_results(impl, dataset, solutions, recall_min=None, qps_min=None, sampling_count=None):
    results_combi = {}
    oracle_best_metric = 0.0

    for solution in solutions:
        filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)

        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
            hp = optimal_hp[0]
            _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1]
            perf = (0.0, recall, qps, total_time, build_time, index_size)
            results = [(hp, perf)]
            oracle_best_metric = qps if recall_min is not None else recall
        else:
            results = [r for r in results if float(r[1][0]) <= TUNING_BUDGET]

        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results_combi,
        "oracle_best_metric": oracle_best_metric,
    }


# =============================
# CSV SUMMARY LOGIC
# =============================
def compute_summary(data, is_feasible, objective, oracle_best):
    valid = []
    for _, perf in data:
        if is_feasible(perf):
            t = float(perf[0])
            if t <= TUNING_BUDGET:
                valid.append((t, float(objective(perf))))

    if not valid or oracle_best <= 0:
        return 0.0, 0.0, None

    valid.sort(key=lambda x: x[0])
    best = -np.inf
    max_perf = -np.inf
    t_hit = None
    target = 0.95 * oracle_best

    for t, v in valid:
        if v > best:
            best = v
        max_perf = max(max_perf, best)
        if t_hit is None and best >= target:
            t_hit = t

    return max_perf, max_perf / oracle_best * 100.0, t_hit


def print_csv(results_list):
    writer = csv.writer(sys.stdout)
    writer.writerow([
        "impl", "dataset", "task", "solution",
        "max_perf", "oracle_perf", "pct_of_oracle", "t_hit_95pct_sec"
    ])

    for r in results_list:
        is_feasible, objective, _ = _feasible_and_objective_factory(
            r["recall_min"], r["qps_min"], TUNING_BUDGET, r["oracle_best_metric"]
        )

        task = "Recall→QPS" if r["recall_min"] is not None else "QPS→Recall"

        for sol, data in r["results"].items():
            max_perf, pct, t_hit = compute_summary(
                data, is_feasible, objective, r["oracle_best_metric"]
            )
            writer.writerow([
                r["impl"],
                r["dataset"],
                task,
                "Oracle" if sol == "brute_force" else SOL_STYLES[sol]["label"],
                f"{max_perf:.6g}",
                f"{r['oracle_best_metric']:.6g}",
                f"{pct:.2f}",
                "" if t_hit is None else f"{t_hit:.3f}",
            ])


# =============================
# MAIN
# =============================
def main():
    SOLUTIONS = ["brute_force", "our_solution", "grid_search", "random_search", "vd_tuner", "optuna", "nsga", "eci"]
    IMPLS = ["hnswlib", "faiss"]
    DATASETS = [
        "nytimes-256-angular", "glove-100-angular",
        "sift-128-euclidean", "youtube-1024-angular", "deep1M-256-angular"
    ]

    SAMPLING_COUNT = 10
    RECALL_MIN = 0.95
    QPS_MIN_KEY = "q75"

    results_list = []
    for impl in IMPLS:
        for ds in DATASETS:
            results_list.append(get_results(impl, ds, SOLUTIONS, recall_min=RECALL_MIN, sampling_count=SAMPLING_COUNT))
        for ds in DATASETS:
            qps_min = get_qps_metrics_dataset(impl, ds, ret_dict=True)[QPS_MIN_KEY]
            results_list.append(get_results(impl, ds, SOLUTIONS, qps_min=qps_min, sampling_count=SAMPLING_COUNT))

    # ---- CSV OUTPUT ----
    print_csv(results_list)

    # ---- FIGURE CODE (unchanged, omitted for brevity if needed) ----
    # (네가 이미 확인했으니 그대로 붙여 쓰면 됨)


if __name__ == "__main__":
    main()
