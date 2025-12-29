import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
    results_combi = {}
    for solution in solutions:
        filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
        results = load_search_results(solution, filename, seed=MOCK_SEED, sampling_count=sampling_count)

        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
            hp = optimal_hp[0]
            _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1]
            perf = (0.0, recall, qps, total_time, build_time, index_size)
            results = [(hp, perf)]
        else:
            results = [r for r in results if r[1][0] <= tuning_time]

        results_combi[solution] = results

    return {"impl": impl, "dataset": dataset, "recall_min": recall_min, "qps_min": qps_min, "results": results_combi}


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
    SAMPLING_COUNT = [10]
    RECALL_MINS = [0.95]
    QPS_MIN_KEY = "q75"

    tasks = []
    for impl in IMPLS:
        for metric in ["recall_min", "qps_min"]:
            for dataset in DATASETS:
                for sampling_count in SAMPLING_COUNT:
                    if metric == "recall_min":
                        for recall_min in RECALL_MINS:
                            tasks.append((impl, dataset, SOLUTIONS, recall_min, None, sampling_count))
                    else:
                        qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
                        tasks.append((impl, dataset, SOLUTIONS, None, qps_min, sampling_count))

    results = [get_results(*t) for t in tasks]

    fig, axes = plt.subplots(4, 5, figsize=(22, 10))

    row_sm = [None] * 4

    for idx, (ax, r) in enumerate(zip(axes.flat, results)):
        sm = plot_gradient_bar_with_oracle_on_ax(
            ax,
            results_dict=r["results"],
            recall_min=r["recall_min"],
            qps_min=r["qps_min"],
            tuning_budget=TUNING_BUDGET,
            cmap_name="Greys",
            y_bins=260,
            show_xticklabels=True,
        )
        ax.set_title(short_ds(r["dataset"]), fontsize=11)
        row = idx // 5
        if row_sm[row] is None:
            row_sm[row] = sm

    # One colorbar per row
    for row in range(4):
        sm = row_sm[row]
        if sm is None:
            continue
        cb = fig.colorbar(sm, ax=axes[row, :], fraction=0.02, pad=0.01)
        cb.set_label("Time (s)", fontsize=10)
        cb.ax.tick_params(labelsize=8)
        cb.set_ticks([0, TUNING_BUDGET / 2, TUNING_BUDGET])

    # Global legend (Oracle only)
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))
    if "Oracle" in by_label:
        fig.legend([by_label["Oracle"]], ["Oracle"], loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=1, fontsize=14, frameon=False)

    # Row labels
    fig.text(0.015, 0.78, "hnswlib\n(Recall constraint)", va="center", ha="center", rotation="vertical", fontsize=16)
    fig.text(0.015, 0.53, "hnswlib\n(QPS constraint)", va="center", ha="center", rotation="vertical", fontsize=16)
    fig.text(0.015, 0.28, "faiss\n(Recall constraint)", va="center", ha="center", rotation="vertical", fontsize=16)
    fig.text(0.015, 0.03, "faiss\n(QPS constraint)", va="center", ha="center", rotation="vertical", fontsize=16)

    plt.subplots_adjust(left=0.06, bottom=0.09, top=0.92, wspace=0.35, hspace=0.55)
    fig.savefig("main_figure.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
