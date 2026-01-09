import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import numpy as np
import os

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    _feasible_and_objective_factory,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"
MOCK_SEED = "0_cherry"

# --- Styles per solution ---
SOL_STYLES = {
    "our_solution":  {"c": "#d62728", "marker": "o", "ls": "-",  "lw": 2.0, "zorder": 10, "label": "CHAT"},
    "vd_tuner":      {"c": "#9467bd", "marker": "s", "ls": "--", "lw": 1.8, "zorder": 5,  "label": "VDTuner"},
    "optuna":        {"c": "#8c564b", "marker": "^", "ls": "-.", "lw": 1.8, "zorder": 4,  "label": "Optuna"},
    "nsga":          {"c": "#e377c2", "marker": "D", "ls": ":",  "lw": 1.8, "zorder": 3,  "label": "NSGA-II"},
    "random_search": {"c": "#7f7f7f", "marker": "v", "ls": "--", "lw": 1.8, "zorder": 2,  "label": "Random"},
    "grid_search":   {"c": "#bcbd22", "marker": "P", "ls": ":",  "lw": 1.8, "zorder": 1,  "label": "Grid"},
}
SOL_ORDER = ["our_solution", "vd_tuner", "optuna", "nsga", "random_search", "grid_search"]

# Base target levels
ALPHAS_BASE = [0.7, 0.8, 0.9, 0.925, 0.95]
TAIL_START = 0.95  

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
            if metric > oracle_best_metric:
                oracle_best_metric = metric
        else:
            results = [r for r in results if r[1][0] <= tuning_time]

        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results_combi,
        "oracle_best_metric": oracle_best_metric,
    }

def build_attainment_points_linear(
    data,
    is_feasible,
    objective,
    oracle_best,
    budget,
    alphas_base=ALPHAS_BASE,
    tail_start=TAIL_START,
):
    if (not data) or oracle_best <= 1e-12:
        return [], []

    valid_points = []
    for _, perf in data:
        if is_feasible(perf):
            t = float(perf[0])
            if t <= budget:
                val = float(objective(perf))
                valid_points.append((t, val))
    valid_points.sort(key=lambda x: x[0])

    if not valid_points:
        return [], []

    times = []
    best_perfs = []
    current_max = -np.inf
    for t, val in valid_points:
        if val > current_max:
            current_max = val
            times.append(t)
            best_perfs.append(current_max)

    plot_xs = []
    plot_ys = []

    for alpha in alphas_base:
        target = alpha * oracle_best
        found_t = None
        for t, b in zip(times, best_perfs):
            if b >= target:
                found_t = t
                break
        if found_t is not None:
            plot_xs.append(found_t)
            plot_ys.append(alpha)

    tail_target = tail_start * oracle_best
    start_idx = None
    for i, b in enumerate(best_perfs):
        if b >= tail_target:
            start_idx = i
            break

    if start_idx is not None:
        last_base_t = plot_xs[-1] if plot_xs else None
        last_base_y = plot_ys[-1] if plot_ys else None

        for t, b in zip(times[start_idx:], best_perfs[start_idx:]):
            ratio = min(float(b / oracle_best), 1.0)
            if last_base_t is not None and last_base_y is not None:
                if abs(t - last_base_t) < 1e-12 and abs(ratio - last_base_y) < 1e-12:
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

    if max_perf is None or max_perf <= 0:
        ax.text(0.5, 0.5, "No Oracle", ha="center", va="center", fontsize=8)
        return

    for sol in SOL_ORDER:
        data = results_dict.get(sol, [])
        style = SOL_STYLES[sol]

        plot_xs, plot_ys = build_attainment_points_linear(
            data=data,
            is_feasible=is_feasible,
            objective=objective,
            oracle_best=max_perf,
            budget=tuning_budget,
            alphas_base=ALPHAS_BASE,
            tail_start=TAIL_START,
        )

        if plot_xs:
            ax.plot(
                plot_xs,
                plot_ys,
                marker=style["marker"],
                color=style["c"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                markersize=5,
                zorder=style["zorder"],
                label=style["label"],
                alpha=0.9,
            )

    ax.set_xlim(0.0, tuning_budget * 1.05)
    major_ticks = [0, tuning_budget / 4, tuning_budget / 2, tuning_budget * 3 / 4, tuning_budget]
    ax.set_xticks(major_ticks)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.get_xaxis().set_major_formatter(formatter)

    # show_xlabel이 True면 눈금(tick)만 표시, 라벨은 main함수에서 직접 제어
    if show_xlabel:
        ax.tick_params(axis="x", labelsize=12)
    else:
        ax.set_xticklabels([])

    yticks = ALPHAS_BASE + [1.0]
    ax.set_ylim(0.7, 1.02)
    ax.set_yticks(yticks)

    if show_ylabel:
        ax.set_yticklabels([f"{int(a*100)}%" for a in yticks], fontsize=12)
    else:
        ax.set_yticklabels([])

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="gray", alpha=0.8)
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="blue", alpha=0.9, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(tuning_budget, linestyle=":", color="gray", linewidth=1.0)


def main():
    # -------------------------------------------------------------------------
    # 1. 폰트 설정
    # -------------------------------------------------------------------------
    font_path_r = f"{current_dir}/LinLibertine_R.ttf"
    fm.fontManager.addfont(font_path_r)

    # ★★★ 파일명을 확인해주세요: LinLibertine_RB.ttf 또는 LinLibertine_B.ttf ★★★
    # font_path_b = f"{current_dir}/LinLibertine_RB.ttf" 
    font_path_b = f"{current_dir}/LinLibertine_B.ttf" 
    
    if os.path.exists(font_path_b):
        fm.fontManager.addfont(font_path_b)
        print(f"Bold font loaded: {font_path_b}")
    else:
        print(f"Warning: Bold font not found at {font_path_b}. Text may not appear bold.")

    font_prop = fm.FontProperties(fname=font_path_r)
    font_name = font_prop.get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    # -------------------------------------------------------------------------

    SOLUTIONS = ["brute_force", "our_solution", "grid_search", "random_search", "vd_tuner", "optuna", "nsga"]
    IMPLS = ["hnswlib", "faiss"]
    DATASETS = ["nytimes-256-angular", "glove-100-angular", "sift-128-euclidean", "youtube-1024-angular", "deep1M-256-angular"]
    SAMPLING_COUNT = 10
    RECALL_MIN = 0.95
    QPS_MIN_KEY = "q75"

    # --- Data loading ---
    results_list = []
    for impl in IMPLS:
        for dataset in DATASETS:
            results_list.append(get_results(impl, dataset, SOLUTIONS, RECALL_MIN, None, SAMPLING_COUNT))
        for dataset in DATASETS:
            qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
            results_list.append(get_results(impl, dataset, SOLUTIONS, None, qps_min, SAMPLING_COUNT))

    # ---- Layout ----
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

    axes = []
    for r in range(4):
        row_axes = []
        for c in range(5):
            row_axes.append(fig.add_subplot(gs[r, c]))
        axes.append(row_axes)

    # ---- Plotting ----
    for idx, rdict in enumerate(results_list):
        r = idx // 5
        c = idx % 5
        ax = axes[r][c]

        show_x = True 
        show_y = True

        plot_attainment_on_ax(
            ax,
            results_dict=rdict["results"],
            recall_min=rdict["recall_min"],
            qps_min=rdict["qps_min"],
            tuning_budget=TUNING_BUDGET,
            max_perf=rdict["oracle_best_metric"],
            show_xlabel=show_x,
            show_ylabel=show_y,
        )

        if r == 0:
            ax.set_title(short_ds(rdict["dataset"]), fontsize=19, pad=4, weight="bold")
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼ [수정된 부분] 맨 아랫줄의 각 Subplot에 라벨 추가 ▼▼▼▼▼▼▼▼▼▼▼▼▼
        # r이 3일 때 (맨 마지막 행), 각각의 ax에 대해 x축 라벨을 설정합니다.
        if r == 3:
            ax.set_xlabel("Time (seconds)", fontsize=16, weight="bold")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # ---- Global Legend ----
    legend_elements = []
    for sol in SOL_ORDER:
        s = SOL_STYLES[sol]
        legend_elements.append(
            Line2D([0], [0], color=s["c"], marker=s["marker"], ls=s["ls"], lw=s["lw"], label=s["label"])
        )

    leg = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=7,
        fontsize=20,
        frameon=False,
    )
    
    # CHAT Bold 처리
    for text in leg.get_texts():
        if text.get_text() == "CHAT":
            text.set_fontweight("bold")

    # ---- Row Labels & Axis Labels ----
    fig.text(0.015, 0.71, "Hnswlib", va="center", ha="center", rotation="vertical", fontsize=24, weight="bold")
    fig.text(0.015, 0.30, "Faiss", va="center", ha="center", rotation="vertical", fontsize=24, weight="bold")

    label_opts = {"va": "center", "ha": "center", "rotation": "vertical", "fontsize": 16, "weight": "bold"}
    fig.text(0.035, 0.78, "QPS", **label_opts)
    fig.text(0.035, 0.59, "Recall", **label_opts)
    fig.text(0.035, 0.38, "QPS", **label_opts)
    fig.text(0.035, 0.185, "Recall", **label_opts)

    output_filename = "main_figure_attainment_linear_dense_tail_95p.pdf"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved linear attainment plot to {output_filename}")

if __name__ == "__main__":
    main()
