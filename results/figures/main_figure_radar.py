# ======================= main_figure.py =========================
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import gridspec
import numpy as np

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    _feasible_and_objective_factory,
    _oracle_best,
    get_best_perf_at_time,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"
MOCK_SEED = 0

# --- Color & Style Definitions for 4 Time Steps ---
# 1/4 (Cyan), 2/4 (Gold), 3/4 (Orange), Full (Red)
TIME_COLORS = ["#17becf", "#bcbd22", "#ff7f0e", "#d62728"]
# TIME_ALPHAS는 fill을 안 하므로 사용되지 않지만, 나중을 위해 남겨두거나 삭제해도 됩니다.
TIME_LINESTYLES = [":", "-.", "--", "-"] # Dotted, Dash-dot, Dashed, Solid
TIME_LINEWIDTHS = [1.5, 1.5, 2.0, 2.5]
ORACLE_COLOR = "tab:blue"

def short_ds(name: str) -> str:
    return name.split("-")[0]

def get_results(impl, dataset, solutions, recall_min=None, qps_min=None, sampling_count=None, tuning_time=TUNING_BUDGET):
    assert (recall_min is not None) != (qps_min is not None)
    results_combi = {}
    oracle_best_metric = None

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

def plot_radar_chart_on_ax(
    ax,
    results_dict,
    recall_min=None,
    qps_min=None,
    tuning_budget=TUNING_BUDGET,
    max_perf=None, # Oracle Performance
):
    """
    Draws a radar chart (spider web) on a polar axis.
    - Vertices: 6 Solutions (excluding Oracle)
    - Series: Performance at T/4, 2T/4, 3T/4, and T (Full Budget)
    - Normalized: All values are relative to Oracle (1.0)
    """
    is_feasible, objective, _ = _feasible_and_objective_factory(recall_min, qps_min, tuning_budget, max_perf)

    solutions = ["our_solution", "vd_tuner", "optuna", "nsga", "random_search", "grid_search"]
    labels = ["CHAT", "VDTuner", "Optuna", "NSGA-II", "Random", "Grid"]
    
    # Angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, len(solutions), endpoint=False).tolist()
    angles += angles[:1] # Close the loop

    # Time points for quarters
    time_points = [
        tuning_budget / 4,
        (tuning_budget / 4) * 2,
        (tuning_budget / 4) * 3,
        tuning_budget
    ]
    
    # Plot for each time checkpoint
    for i, t_limit in enumerate(time_points):
        values = []
        for sol in solutions:
            sol_data = results_dict.get(sol, [])
            best_val = get_best_perf_at_time(sol_data, is_feasible, objective, t_limit)
            
            # Normalize against Oracle
            norm_val = (best_val / max_perf) if max_perf > 1e-9 else 0.0
            values.append(norm_val)
        
        values += values[:1] # Close the loop
        
        # Plot Line ONLY (No Fill)
        ax.plot(
            angles, 
            values, 
            color=TIME_COLORS[i], 
            linewidth=TIME_LINEWIDTHS[i],
            linestyle=TIME_LINESTYLES[i],
            zorder=i+2 
        )
        
        # [REMOVED] Fill Area
        # ax.fill(angles, values, color=TIME_COLORS[i], alpha=TIME_ALPHAS[i], zorder=i+1)

    # --- Oracle Reference (Blue Dashed Line at 1.0) ---
    oracle_values = [1.0] * (len(solutions) + 1)
    ax.plot(angles, oracle_values, color=ORACLE_COLOR, linestyle="--", linewidth=1.5, label="Oracle", zorder=10)

    # --- Formatting the Polar Plot ---
    ax.set_theta_offset(np.pi / 2) # Start from top
    ax.set_theta_direction(-1)     # Clockwise
    
    # X-Axis (The Spokes)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, weight='bold')
    
    # Y-Axis (Radial)
    ax.set_rlabel_position(30)
    plt.yticks([0.5, 1.0], ["0.5", "1.0"], color="grey", size=9) 
    plt.ylim(0, 1.05) 
    
    # Style the grid
    ax.grid(color='#BBBBBB', linestyle=':', linewidth=0.8)
    
    # Remove outer spine
    ax.spines['polar'].set_visible(False)

def main():
    # Font Setup
    font_path = f"{current_dir}/LinLibertine_R.ttf"
    if font_path not in [f.fname for f in fm.fontManager.ttflist]:
        fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    SOLUTIONS = ["brute_force", "our_solution", "grid_search", "random_search", "vd_tuner", "optuna", "nsga"]
    IMPLS = ["hnswlib", "faiss"]
    DATASETS = ["nytimes-256-angular", "glove-100-angular", "sift-128-euclidean", "youtube-1024-angular", "deep1M-256-angular"]
    SAMPLING_COUNT = 10
    RECALL_MIN = 0.95
    QPS_MIN_KEY = "q75"

    # --- Data Loading ---
    results_list = []
    for impl in IMPLS:
        for dataset in DATASETS:
            results_list.append(get_results(impl, dataset, SOLUTIONS, RECALL_MIN, None, SAMPLING_COUNT))
        for dataset in DATASETS:
            qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
            results_list.append(get_results(impl, dataset, SOLUTIONS, None, qps_min, SAMPLING_COUNT))

    # ---- Layout Setup ----
    fig = plt.figure(figsize=(20, 15))
    
    gs = gridspec.GridSpec(
        nrows=4, ncols=5, figure=fig,
        left=0.05, right=0.98,
        top=0.90, bottom=0.10,
        wspace=0.35, hspace=0.45 
    )

    axes = []
    for r in range(4):
        row_axes = []
        for c in range(5):
            ax = fig.add_subplot(gs[r, c], projection='polar')
            row_axes.append(ax)
        axes.append(row_axes)

    # ---- Plotting ----
    for idx, rdict in enumerate(results_list):
        r = idx // 5
        c = idx % 5
        ax = axes[r][c]

        plot_radar_chart_on_ax(
            ax,
            results_dict=rdict["results"],
            recall_min=rdict["recall_min"],
            qps_min=rdict["qps_min"],
            tuning_budget=TUNING_BUDGET,
            max_perf=rdict["oracle_best_metric"]
        )

        if r == 0:
            ax.set_title(short_ds(rdict["dataset"]), fontsize=18, pad=25, weight='bold')

    # ---- Global Legend ----
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color=TIME_COLORS[0], linestyle=TIME_LINESTYLES[0], linewidth=2, label='1/4 Budget'),
        Line2D([0], [0], color=TIME_COLORS[1], linestyle=TIME_LINESTYLES[1], linewidth=2, label='2/4 Budget'),
        Line2D([0], [0], color=TIME_COLORS[2], linestyle=TIME_LINESTYLES[2], linewidth=2, label='3/4 Budget'),
        Line2D([0], [0], color=TIME_COLORS[3], linestyle=TIME_LINESTYLES[3], linewidth=3, label='Full Budget'),
        Line2D([0], [0], color=ORACLE_COLOR, linestyle="--", linewidth=2, label='Oracle (normalized to 1.0)'),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        fontsize=15,
        frameon=False,
    )

    # ---- Row Labels ----
    fig.text(0.01, 0.71, "hnswlib", va="center", ha="center", rotation="vertical", fontsize=24, weight='bold')
    fig.text(0.01, 0.29, "faiss", va="center", ha="center", rotation="vertical", fontsize=24, weight='bold')
    
    label_opts = {'va': 'center', 'ha': 'center', 'rotation': 'vertical', 'fontsize': 15, 'weight': 'bold'}
    fig.text(0.028, 0.815, "Max QPS\n(Recall≥0.95)", **label_opts)
    fig.text(0.028, 0.605, "Max Recall\n(QPS≥Q75)", **label_opts)
    fig.text(0.028, 0.395, "Max QPS\n(Recall≥0.95)", **label_opts)
    fig.text(0.028, 0.185, "Max Recall\n(QPS≥Q75)", **label_opts)

    output_filename = "main_figure_radar.pdf"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved no-fill radar chart to {output_filename}")

if __name__ == "__main__":
    main()