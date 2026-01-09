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
    "vd_tuner":      {"c": "#9467bd", "marker": "s", "ls": "--", "lw": 1.2, "zorder": 5,  "label": "VDTuner"},
    "optuna":        {"c": "#8c564b", "marker": "^", "ls": "-.", "lw": 1.2, "zorder": 4,  "label": "Optuna"},
    "nsga":          {"c": "#e377c2", "marker": "D", "ls": ":",  "lw": 1.2, "zorder": 3,  "label": "NSGA-II"},
    "random_search": {"c": "#7f7f7f", "marker": "v", "ls": "--", "lw": 1.2, "zorder": 2,  "label": "Random"},
    "grid_search":   {"c": "#bcbd22", "marker": "P", "ls": ":",  "lw": 1.2, "zorder": 1,  "label": "Grid"},
}
SOL_ORDER = ["our_solution", "vd_tuner", "optuna", "nsga", "random_search", "grid_search"]

ALPHAS_BASE = [0, 0.7, 0.8, 0.9 , 0.95]
ALPHAS_CALC = [0.001, 0.25, 0.5] + ALPHAS_BASE
TAIL_START = 0.95

# Broken-axis (piecewise linear) mapping
BOUNDARY_REAL = 0.7
BOUNDARY_VISUAL = 0.30


def squash_y(y: float) -> float:
    """Map real ratio y in [0,1] to visual y in [0,1] with piecewise linear squash."""
    y = float(y)
    if y <= BOUNDARY_REAL:
        return y * (BOUNDARY_VISUAL / BOUNDARY_REAL)
    slope = (1.0 - BOUNDARY_VISUAL) / (1.0 - BOUNDARY_REAL)
    return BOUNDARY_VISUAL + (y - BOUNDARY_REAL) * slope


def draw_wavy_y_break(
    ax,
    y_data: float,
    x_center: float = -0.02,
    width: float = 0.06,
    amp: float = 0.012,
    gap: float = 0.028,
    cycles: float = 1.0,
    lw: float = 0.5,        # [수정] 기본 두께를 얇게 변경
    color: str = "gray",    # [수정] 기본 색상을 회색으로 변경
    alpha: float = 0.5      # [수정] 투명도 추가
):
    """
    Draw a 'wavy break' symbol (two sine-like strokes) near the left y-axis at y_data.
    """
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    xs = np.linspace(x_center - width / 2.0, x_center + width / 2.0, 200)
    phase = 2.0 * np.pi * cycles * (xs - xs.min()) / (xs.max() - xs.min())

    y_top = y_data + gap / 2.0 + amp * np.sin(phase)
    y_bot = y_data - gap / 2.0 + amp * np.sin(phase)

    # [수정] color, alpha, lw 적용
    ax.plot(xs, y_top, transform=trans, color=color, lw=lw, alpha=alpha, clip_on=False, solid_capstyle="round", zorder=50)
    ax.plot(xs, y_bot, transform=trans, color=color, lw=lw, alpha=alpha, clip_on=False, solid_capstyle="round", zorder=50)


def short_ds(name: str) -> str:
    return name.split("-")[0]


def get_results(impl, dataset, solutions, recall_min=None, qps_min=None, sampling_count=None, tuning_time=TUNING_BUDGET):
    assert (recall_min is not None) != (qps_min is not None)

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

            metric = qps if recall_min is not None else recall
            oracle_best_metric = max(oracle_best_metric, float(metric))
        else:
            results = [r for r in results if float(r[1][0]) <= tuning_time]

        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results_combi,
        "oracle_best_metric": oracle_best_metric,
    }


def build_attainment_points(
    data,
    is_feasible,
    objective,
    oracle_best,
    budget,
    alphas_calc=ALPHAS_CALC,
    tail_start=TAIL_START,
):
    """Return attainment curve points: x=t_hit, y=ratio_to_oracle (0..1)."""
    if (not data) or oracle_best <= 1e-12:
        return [], []

    valid = []
    for _, perf in data:
        if is_feasible(perf):
            t = float(perf[0])
            if t <= budget:
                val = float(objective(perf))
                valid.append((t, val))
    valid.sort(key=lambda x: x[0])

    if not valid:
        return [], []

    # Build monotone best-so-far envelope
    times, bests = [], []
    cur = -np.inf
    for t, v in valid:
        if v > cur:
            cur = v
            times.append(t)
            bests.append(cur)

    plot_xs, plot_ys = [], []

    # Base ratios (including low ratios for visibility)
    for alpha in alphas_calc:
        target = alpha * oracle_best
        hit = None
        for t, b in zip(times, bests):
            if b >= target:
                hit = t
                break
        if hit is not None:
            plot_xs.append(hit)
            plot_ys.append(alpha)

    # Tail points (fine-grained after tail_start)
    tail_target = tail_start * oracle_best
    start_idx = None
    for i, b in enumerate(bests):
        if b >= tail_target:
            start_idx = i
            break

    if start_idx is not None:
        last_t = plot_xs[-1] if plot_xs else None
        last_y = plot_ys[-1] if plot_ys else None

        for t, b in zip(times[start_idx:], bests[start_idx:]):
            ratio = min(float(b / oracle_best), 1.0)
            if last_t is not None and last_y is not None:
                if abs(t - last_t) < 1e-12 and abs(ratio - last_y) < 1e-12:
                    continue
            plot_xs.append(float(t))
            plot_ys.append(ratio)

    return plot_xs, plot_ys


def plot_attainment_on_ax(
    ax,
    results_dict,
    recall_min=None,
    qps_min=None,
    tuning_budget=TUNING_BUDGET,
    max_perf=None,
    show_xlabel=False,
    show_ylabel=False,
):
    is_feasible, objective, _ = _feasible_and_objective_factory(recall_min, qps_min, tuning_budget, max_perf)

    if not max_perf or max_perf <= 0:
        ax.text(0.5, 0.5, "No Oracle", ha="center", va="center", fontsize=8)
        return

    for sol in SOL_ORDER:
        data = results_dict.get(sol, [])
        style = SOL_STYLES[sol]

        xs, ys_real = build_attainment_points(
            data=data,
            is_feasible=is_feasible,
            objective=objective,
            oracle_best=max_perf,
            budget=tuning_budget,
        )

        ys_vis = [squash_y(y) for y in ys_real]
        if xs:
            ax.plot(
                xs,
                ys_vis,
                marker=style["marker"],
                color=style["c"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                markersize=5,
                zorder=style["zorder"],
                label=style["label"],
                alpha=0.9,
            )

    # X axis
    ax.set_xlim(-500, tuning_budget * 1.05)
    ax.set_xticks([0, tuning_budget / 4, tuning_budget / 2, tuning_budget * 3 / 4, tuning_budget])
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    if show_xlabel:
        ax.tick_params(axis="x", labelsize=12)
    else:
        ax.set_xticklabels([])

    # Y axis (custom ticks in visual space, labels in real %)
    display_yticks_real = [0.0] + ALPHAS_BASE + [1.0]
    display_yticks_vis = [squash_y(y) for y in display_yticks_real]
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(display_yticks_vis)
    if show_ylabel:
        ax.set_yticklabels([f"{int(y*100)}%" for y in display_yticks_real], fontsize=12)
    else:
        ax.set_yticklabels([])

    # Grid
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, color="gray", alpha=0.8)

    # Reference lines (가로선)
    break_y_vis = squash_y(BOUNDARY_REAL)
    
    # [수정] 75% 경계선을 연하고 얇게 (gray, lw=0.5, alpha=0.5)
    ax.axhline(break_y_vis, linestyle=":", linewidth=0.7, color="gray", alpha=0.8, zorder=0)

    top_y_vis = squash_y(1.0)
    ax.axhline(top_y_vis, linestyle="--", linewidth=1.0, color="blue", alpha=0.9, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(tuning_budget, linestyle="--", color="gray", linewidth=1.0)

    # --- Wavy compression mark ---
    y_mid_real = BOUNDARY_REAL / 2.0
    y_mid_vis = squash_y(y_mid_real)
    if show_ylabel:
        # [수정] 물결 무늬도 연하게 호출 (위에서 default를 gray/0.5로 바꿨으므로 인자만 맞춰줌)
        draw_wavy_y_break(
            ax, 
            y_data=y_mid_vis, 
            x_center=-0.005, 
            width=0.05, 
            amp=0.015, 
            gap=0.028, 
            cycles=1.0, 
            lw=0.9,           # 얇게
            color="black",     # 회색
            alpha=0.8         # 투명도
        )


def setup_fonts():
    """Load Libertine fonts if present."""
    font_path_r = f"{CURRENT_DIR}/LinLibertine_R.ttf"
    fm.fontManager.addfont(font_path_r)

    font_path_b = f"{CURRENT_DIR}/LinLibertine_B.ttf"
    if os.path.exists(font_path_b):
        fm.fontManager.addfont(font_path_b)

    font_prop = fm.FontProperties(fname=font_path_r)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False


def build_global_legend(fig):
    """Global legend with CHAT first and bold."""
    handles = []
    labels = []
    for sol in SOL_ORDER:
        s = SOL_STYLES[sol]
        handles.append(Line2D([0], [0], color=s["c"], marker=s["marker"], ls=s["ls"], lw=s["lw"]))
        if sol == "our_solution":
            labels.append("CHAT (Our solution)")
        else:
            labels.append(s["label"])

    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.47, 1.0),
        ncol=len(SOL_ORDER),
        fontsize=18,
        frameon=False,
    )
    for t in leg.get_texts():
        if t.get_text().startswith("CHAT"):
            t.set_fontweight("bold")


def main():
    setup_fonts()

    SOLUTIONS = ["brute_force", "our_solution", "grid_search", "random_search", "vd_tuner", "optuna", "nsga"]
    IMPLS = ["hnswlib", "faiss"]
    DATASETS = ["nytimes-256-angular", "glove-100-angular", "sift-128-euclidean", "youtube-1024-angular", "deep1M-256-angular"]
    SAMPLING_COUNT = 10
    RECALL_MIN = 0.95
    QPS_MIN_KEY = "q75"

    # Load data: 2 impl * 5 datasets * (QPS/Recall rows) = 20 subplots
    results_list = []
    for impl in IMPLS:
        for dataset in DATASETS:
            results_list.append(get_results(impl, dataset, SOLUTIONS, recall_min=RECALL_MIN, qps_min=None, sampling_count=SAMPLING_COUNT))
        for dataset in DATASETS:
            qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
            results_list.append(get_results(impl, dataset, SOLUTIONS, recall_min=None, qps_min=qps_min, sampling_count=SAMPLING_COUNT))

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(
        nrows=4,
        ncols=5,
        figure=fig,
        left=0.06,
        right=0.98,
        top=0.86,
        bottom=0.10,
        wspace=0.16,
        hspace=0.18,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(4)]

    for idx, rdict in enumerate(results_list):
        r = idx // 5
        c = idx % 5
        ax = axes[r][c]

        plot_attainment_on_ax(
            ax,
            results_dict=rdict["results"],
            recall_min=rdict["recall_min"],
            qps_min=rdict["qps_min"],
            tuning_budget=TUNING_BUDGET,
            max_perf=rdict["oracle_best_metric"],
            show_xlabel=True,
            show_ylabel=True,
        )

        if r == 0:
            ax.set_title(short_ds(rdict["dataset"]), fontsize=19, pad=4, weight="bold")
        if r == 3:
            ax.set_xlabel("Time (seconds)", fontsize=16, weight="bold")

    build_global_legend(fig)

    # Row/column labels
    fig.text(0.015, 0.71, "Hnswlib", va="center", ha="center", rotation="vertical", fontsize=24, weight="bold")
    fig.text(0.015, 0.30, "Faiss",   va="center", ha="center", rotation="vertical", fontsize=24, weight="bold")

    label_opts = {"va": "center", "ha": "center", "rotation": "vertical", "fontsize": 16, "weight": "bold"}
    fig.text(0.035, 0.78, "QPS", **label_opts)
    fig.text(0.035, 0.59, "Recall", **label_opts)
    fig.text(0.035, 0.38, "QPS", **label_opts)
    fig.text(0.035, 0.185, "Recall", **label_opts)

    output_filename = "main_figure_attainment_broken_axis.pdf"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved broken axis plot to {output_filename}")


if __name__ == "__main__":
    main()