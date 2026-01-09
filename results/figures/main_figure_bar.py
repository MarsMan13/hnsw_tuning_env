import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import gridspec

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    plot_gradient_bar_with_oracle_on_ax,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"
MOCK_SEED = 0


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


def main():
    # Font
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

    tasks = []
    for impl in IMPLS:
        # Row 0/2: recall constraint (maximize QPS)
        for dataset in DATASETS:
            tasks.append((impl, dataset, SOLUTIONS, RECALL_MIN, None, SAMPLING_COUNT))
        # Row 1/3: qps constraint (maximize Recall)
        for dataset in DATASETS:
            qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
            tasks.append((impl, dataset, SOLUTIONS, None, qps_min, SAMPLING_COUNT))

    # Arrange into 4 rows x 5 cols
    # Row order: hnswlib(recall), hnswlib(qps), faiss(recall), faiss(qps)
    results = []
    for impl in IMPLS:
        # recall row
        for dataset in DATASETS:
            results.append(get_results(impl, dataset, SOLUTIONS, RECALL_MIN, None, SAMPLING_COUNT))
        # qps row
        for dataset in DATASETS:
            qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
            results.append(get_results(impl, dataset, SOLUTIONS, None, qps_min, SAMPLING_COUNT))

    # ---- Layout: 4 rows x (5 plots + 1 colorbar column) ----
    fig = plt.figure(figsize=(22, 10))
    
    # 1. Main Plots GridSpec (5 columns)
    # right=0.915 까지 사용하여 데이터 플롯을 배치
    gs_main = gridspec.GridSpec(
        nrows=4, ncols=5, figure=fig,
        left=0.075, right=0.915, 
        top=0.93, bottom=0.10,
        wspace=0.2, hspace=0.4
    )

    # 2. Colorbar GridSpec (1 column)
    # left=0.925 부터 시작하여 gs_main과 0.01 만큼의 아주 좁은 간격만 둠
    gs_cbar = gridspec.GridSpec(
        nrows=4, ncols=1, figure=fig,
        left=0.925, right=0.935, # 컬러바 너비 조절 (0.925 ~ 0.940)
        top=0.93, bottom=0.10,
        hspace=0.55
    )

    axes = [[None] * 5 for _ in range(4)]
    cax = [None] * 4  # colorbar axes per row
    row_sm = [None] * 4

    # Create axes
    for r in range(4):
        for c in range(5):
            axes[r][c] = fig.add_subplot(gs_main[r, c]) # gs -> gs_main
        cax[r] = fig.add_subplot(gs_cbar[r, 0])         # gs[r, 5] -> gs_cbar[r, 0]
    # Plot
    for idx, rdict in enumerate(results):
        r = idx // 5
        c = idx % 5
        ax = axes[r][c]

        sm = plot_gradient_bar_with_oracle_on_ax(
            ax,
            results_dict=rdict["results"],
            recall_min=rdict["recall_min"],
            qps_min=rdict["qps_min"],
            tuning_budget=TUNING_BUDGET,
            max_perf=rdict["oracle_best_metric"],
            cmap_name="Greys",
            y_bins=260,
            show_xticklabels=True,
            # row=r,
        )
        if r == 0:
            ax.set_title(short_ds(rdict["dataset"]), fontsize=16)
        if row_sm[r] is None:
            row_sm[r] = sm

        # Optional: pull ylabel closer so it doesn't eat margin
        ax.yaxis.labelpad = 2

    # Colorbars (fixed column, never overlaps plots)
    for r in range(4):
        sm = row_sm[r]
        if sm is None:
            continue
        cb = fig.colorbar(sm, cax=cax[r])
        cb.set_label("Time (s)", fontsize=10)
        cb.ax.tick_params(labelsize=8)
        cb.set_ticks([0, TUNING_BUDGET / 4, TUNING_BUDGET / 2, TUNING_BUDGET / 4 * 3, TUNING_BUDGET])


    # Global legend (Oracle only)
    handles, labels = [], []
    for r in range(4):
        for c in range(5):
            h, l = axes[r][c].get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
    by_label = dict(zip(labels, handles))
    if "Oracle" in by_label:
        fig.legend(
            [by_label["Oracle"]], ["Oracle"],
            loc="upper center", bbox_to_anchor=(0.5, 1.01),
            ncol=1, fontsize=14, frameon=False,
        )

    # Row labels (optional)
    fig.text(0.035, 0.73, "hnswlib", va="center", ha="center", rotation="vertical", fontsize=24)
    fig.text(0.035, 0.29, "faiss", va="center", ha="center", rotation="vertical", fontsize=24)
    
    fig.text(0.05, 0.85, "QPS", va="center", ha="center", rotation="vertical", fontsize=16)
    fig.text(0.05, 0.63, "Recall", va="center", ha="center", rotation="vertical", fontsize=16)
    fig.text(0.05, 0.4, "QPS", va="center", ha="center", rotation="vertical", fontsize=16)
    fig.text(0.05, 0.18, "Recall", va="center", ha="center", rotation="vertical", fontsize=16)


    fig.savefig("main_figure_bar.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
